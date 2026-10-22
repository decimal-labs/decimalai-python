"""Keyless ``agent_config()``: an agent that runs before there is a DecimalAI account.

With no ``DECIMAL_API_KEY`` (or ``DECIMALAI_API_KEY``) set and ``decimalai.init()``
never called, every ``agent_config()`` runs keyless:

- the prompt is the ``fallback=`` text, or none (the instructions are then the skill
  list alone); ``prompt_source`` is ``"fallback (no key)"``;
- the skill list is the project's: the registry skills ``.decimal/skills.toml`` lists
  under ``[skills]`` (installed in ``.agents/skills``) and the skills you wrote that it
  lists under ``[authored]`` (`decimalai._authored`). ``decimalai skills add`` puts a
  skill on it. Nothing is searched and nothing is added on the agent's behalf: a folder
  in ``.agents/skills`` that is not listed (a Codex or Cursor skill, say) is left out;
- every skill's text is read from disk, so nothing is sent: no traces, no routing
  decisions, no prompt reads, no registry requests.

Adding a key connects the project (:func:`connect`, on the first keyed
``agent_config()`` of each agent): the agent is created when the workspace has none
by that name, and the project's list is copied onto it once: registry skills are
assigned at their pinned versions, and skills you wrote are uploaded to the
organization (not published) and added to it. ``.decimal/link.toml`` records the
workspace and what was copied. From then on the agent's list lives on DecimalAI: the
dashboard and ``decimalai skills add`` change it, and a run never does. A later keyed
start reads only local files and says when the project disagrees with what was copied
(an entry added to ``skills.toml`` by hand, a skill you wrote edited since its upload).
A run with no key in that project raises (:func:`check_link`) rather than quietly running
keyless: that file, committed, is what protects a production deploy that lost its key.
The prompt in the code is never saved to DecimalAI: a saved prompt overrides the
code's, which is exactly what the code's author did not ask for.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import uuid4

from . import _authored

logger = logging.getLogger("decimalai")

#: The link file, in the project's ``.decimal/`` beside ``skills.toml``.
LINK_FILE = "link.toml"

DEFAULT_BASE_URL = "https://api.decimal.ai"

#: The project's skills listed on a run, the platform's menu cap.
_MAX_PROJECT_SKILLS = 20
_MAX_DESCRIPTION_CHARS = 200
#: Routing decisions kept for loads to name (one per run).
_MAX_DECISIONS = 256
#: The body guardrail the platform's router applies per run.
_MAX_LOADED_BODIES = 3
_BODY_CHAR_LIMIT = 24_000

#: The platform's menu, byte for byte, so a keyless model sees what a keyed one does
#: (``_render_stable_menu`` and ``MENU_INSTRUCTION`` in the backend's skill_service).
_MENU_HEADER = ("## Available Skills", "", "| Skill | Description |", "|-------|-------------|")
MENU_INSTRUCTION = (
    "When a skill above applies, follow its instructions and then answer the "
    "user directly. Do not reply with only a statement about which skill you "
    "are using."
)


def base_url() -> str:
    """The registry a keyless process reads: ``DECIMAL_BASE_URL``, as the CLI reads it."""
    url = (os.environ.get("DECIMAL_BASE_URL", "").strip()
           or os.environ.get("DECIMALAI_BASE_URL", "").strip()
           or DEFAULT_BASE_URL)
    return url.rstrip("/")


def _say(line: str) -> None:
    """One line on stderr: what keyless mode just did to the project, or must be told."""
    try:
        sys.stderr.write(f"decimalai: {line}\n")
        sys.stderr.flush()
    except Exception:  # a closed stderr must never cost a run
        pass


# ── relevance: the platform's word-overlap scorer ──────────────────────────────
#
# A port of `_query_terms` / `_lexical_relevance` in the backend's skill_service.py,
# which the platform's own menus use as their word-overlap floor. Ported rather than
# invented so the two agree on what "shares a word" means. With load_skill off, it
# picks the project skill whose text goes into the run's instructions.

_STOPWORDS = frozenset({
    "a", "about", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "does", "for", "from", "get", "give", "had", "has", "have", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "me", "my", "need", "needs",
    "not", "of", "on", "one", "or", "our", "out", "please", "so", "some",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "up", "us", "use", "using", "want", "wants", "was", "we",
    "were", "what", "when", "which", "who", "why", "will", "with", "would",
    "you", "your",
})
_MIN_TERM_LEN = 3
_PREFIX_FLOOR = 5


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxzh":
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: Optional[str]) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    for ch in (text or "").lower():
        if ch.isalnum():
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return [_stem(t) for t in out if len(t) >= _MIN_TERM_LEN]


def query_terms(query: Optional[str]) -> List[str]:
    """The message's distinct content words, stemmed, in order."""
    seen: Set[str] = set()
    terms: List[str] = []
    for t in _tokens(query):
        if t not in _STOPWORDS and t not in seen:
            seen.add(t)
            terms.append(t)
    return terms


def _terms_match(a: str, b: str) -> bool:
    """Equal stems, or a long shared prefix ('cancel' and 'cancellation')."""
    if a == b:
        return True
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n >= _PREFIX_FLOOR and n >= min(len(a), len(b)) - 2


def word_overlap(terms: Sequence[str], name: Optional[str], description: Optional[str]) -> float:
    """How much of the message a skill's name and description cover: 2 a name hit, 1 a description hit."""
    name_tokens, desc_tokens = _tokens(name), _tokens(description)
    score = 0.0
    for term in terms:
        if any(_terms_match(term, t) for t in name_tokens):
            score += 2.0
        elif any(_terms_match(term, t) for t in desc_tokens):
            score += 1.0
    return score


# ── the project's skills ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Offer:
    """One skill a run lists: a registry skill installed in the project, or one you wrote."""

    name: str
    description: str
    path: Optional[Path] = None          # its SKILL.md on disk
    version: Optional[int] = None        # the installed registry version; None for one you wrote
    authored: bool = False               # listed under [authored]: a skill you wrote


def _one_line(text: Any) -> str:
    line = " ".join(str(text or "").split()).replace("|", "/")
    if len(line) > _MAX_DESCRIPTION_CHARS:
        line = line[: _MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
    return line


def project_skills(project: Any = None) -> Dict[str, Offer]:
    """The project's skills, as ``.decimal/skills.toml`` lists them, each read from disk.

    Registry skills: each ``[skills]`` entry the lock has installed, keyed by registry
    name, the name the manifest and the menu use. A skill listed but not installed
    (``decimalai skills install`` not run yet) is left out: there is no text to serve.
    Skills you wrote: each ``[authored]`` name whose ``.agents/skills/<name>/SKILL.md``
    exists and was not installed from the registry, keyed by folder name. Nothing else
    in ``.agents/skills`` is listed.
    """
    from .skillpkg import lock as _lock
    from .skillpkg import manifest as _manifest
    from .skillpkg.project import Project
    from .skills import parse_skill_md

    try:
        project = project or Project.find()
        if not project.manifest_path.is_file():
            return {}
        wanted = _manifest.load(project, missing_ok=True)
        entries = _lock.locked_entries(project) or {}
    except Exception:
        logger.debug("keyless: could not read the project's skills", exc_info=True)
        return {}

    def read(path: Path) -> Optional[Dict[str, Any]]:
        try:
            return parse_skill_md(str(path)) if path.is_file() else None
        except Exception:
            return None

    out: Dict[str, Offer] = {}
    for install_name, entry in sorted(entries.items()):
        name = entry.get("name")
        if not isinstance(name, str) or name not in wanted:
            continue
        path = project.skill_dir(install_name) / "SKILL.md"
        parsed = read(path)
        if not parsed:
            continue
        out[name] = Offer(name=name, description=_one_line(parsed.get("description")), path=path,
                          version=entry.get("version") if isinstance(entry.get("version"), int) else None)
        if len(out) >= _MAX_PROJECT_SKILLS:
            return out
    for name in _authored.listed(project):
        if name in out or name in entries:
            continue  # a registry install of that folder name: the lock's entry is the skill
        path = _authored.skill_md(project, name)
        parsed = read(path)
        if not parsed:
            continue
        out[name] = Offer(name=name, description=_one_line(parsed.get("description")), path=path,
                          authored=True)
        if len(out) >= _MAX_PROJECT_SKILLS:
            break
    return out


def _body(offer: Offer) -> Optional[str]:
    from .skills import parse_skill_md

    try:
        parsed = parse_skill_md(str(offer.path)) if offer.path is not None else None
    except Exception:
        logger.debug("keyless: could not read %s", offer.path, exc_info=True)
        return None
    body = (parsed or {}).get("body")
    return body.strip() if isinstance(body, str) and body.strip() else None


def render_menu(offers: Sequence[Offer]) -> str:
    """The skill list, as the platform renders it, with its instruction."""
    if not offers:
        return ""
    lines = list(_MENU_HEADER)
    lines.extend(f"| {o.name} | {o.description} |" for o in offers)
    return "\n".join(lines) + "\n\n" + MENU_INSTRUCTION


# ── the router a keyless binding uses ────────────────────────────────────────


class KeylessSkillRouter:
    """Lists and serves a keyless agent's skills: the project's, read from disk.

    It stands where an agent's ``SkillRouter`` stands (`AgentBinding` reads the same few
    methods and attributes), and records nothing: a keyless run is sent nowhere. The
    list is read again on every run, so a skill ``decimalai skills add`` lists while the
    process runs is offered from the next run on.
    """

    def __init__(self, *, project_dir: Optional[str] = None) -> None:
        self.base_url = base_url()
        self._project_dir = project_dir
        self._rail_lock = threading.Lock()
        #: run key -> the routing id its run was given.
        self._run_routing_ids: "OrderedDict[str, str]" = OrderedDict()
        #: run key -> names whose text that run was served.
        self._loads: "OrderedDict[str, Set[str]]" = OrderedDict()

    def __repr__(self) -> str:
        return "<KeylessSkillRouter (the project's skills)>"

    def _project(self) -> Any:
        from .skillpkg.project import Project

        return Project.find(self._project_dir)

    # ── one run's list ──────────────────────────────────────
    def build_prompt_parts(
        self, query: Optional[str] = None, *, agent_name: Optional[str] = None,
        inject_body: Optional[bool] = None, scope: Optional[str] = None, **_ignored: Any,
    ) -> Tuple[str, str, Optional[str]]:
        """``(menu, "", routing id)`` for one run, as `SkillRouter.build_prompt_parts`.

        It reports the list as cut, so the binding asks for it on every run (it is a read
        of local files) and its ``load_skill`` takes any name rather than an enum of one
        run's list: a skill listed while the process runs is served on its next run.
        """
        from .skill_router import (
            _last_delivered_names_ctx,
            _last_menu_truncated_ctx,
            _last_offered_names_ctx,
        )

        local = project_skills(self._project())
        offers = list(local.values())
        menu = render_menu(offers)
        delivered: List[str] = []
        if inject_body and query:
            # load_skill is off: the best-matching project skill's text goes in the list.
            terms = query_terms(query)
            best = max(local.values(), key=lambda o: word_overlap(terms, o.name, o.description),
                       default=None)
            text = _body(best) if best is not None and word_overlap(
                terms, best.name, best.description) >= 1 else None
            if best is not None and text:
                menu = f"{menu}\n\n## Skill: {best.name}\n\n{text}"
                delivered = [best.name]
        routing_id = f"keyless_{uuid4().hex}"
        self._remember_run_routing(scope, routing_id)
        _last_offered_names_ctx.set([o.name for o in offers])
        _last_menu_truncated_ctx.set(True)
        _last_delivered_names_ctx.set(delivered or None)
        return menu, "", routing_id

    # ── the rails a binding reads and releases ─────────────
    def _remember_run_routing(self, scope: Optional[str], routing_id: Optional[str]) -> None:
        if scope is None or not routing_id:
            return
        with self._rail_lock:
            self._run_routing_ids[scope] = routing_id
            self._run_routing_ids.move_to_end(scope)
            while len(self._run_routing_ids) > _MAX_DECISIONS:
                self._run_routing_ids.popitem(last=False)

    def consume_routing_id(self, scope: Optional[str] = None) -> Optional[str]:
        return None

    def consume_offered_names(self, scope: Optional[str] = None) -> List[str]:
        return []

    def consume_delivered_names(self, scope: Optional[str] = None) -> List[str]:
        return []

    def consume_delivered_versions(self, scope: Optional[str] = None) -> List[Dict[str, str]]:
        return []

    def consume_loaded_names(self, scope: Optional[str] = None) -> List[str]:
        return []

    def consume_loaded_hashes(self, scope: Optional[str] = None) -> Dict[str, str]:
        return {}

    # ── load_skill ──────────────────────────────────────────
    def load_skill(self, name: str, *, agent_name: Optional[str] = None,
                   scope: Optional[str] = None) -> str:
        """One listed skill's text for the model, read from disk.

        Always a string the model can act on, as `SkillRouter.load_skill`. A name the
        project does not list gets "no skill named", whatever sits in ``.agents/skills``.
        """
        name = (name or "").strip()
        if not name:
            return "load_skill error: a skill name is required."
        if scope is not None:
            with self._rail_lock:
                loaded = self._loads.setdefault(scope, set())
                self._loads.move_to_end(scope)
                while len(self._loads) > _MAX_DECISIONS:
                    self._loads.popitem(last=False)
                if name not in loaded and len(loaded) >= _MAX_LOADED_BODIES:
                    return (f"load_skill budget exhausted: at most {_MAX_LOADED_BODIES} skills "
                            "load per turn. Proceed with what is already loaded.")
        offer = project_skills(self._project()).get(name)
        if offer is None:
            return f"load_skill: no skill named {name!r} is available. Use the exact name from the skills menu."
        body = _body(offer)
        if body is None:
            return f"load_skill: {name!r} has no instructions on disk ({offer.path}). Proceed without it."
        if len(body) > _BODY_CHAR_LIMIT:
            return (f"load_skill budget exhausted: the complete body of {name!r} exceeds the "
                    f"{_BODY_CHAR_LIMIT}-character per-body limit. Its instructions were not "
                    "loaded. Proceed with what is already loaded.")
        if scope is not None:
            with self._rail_lock:
                self._loads.setdefault(scope, set()).add(name)
        return f"## Skill: {name}\n\n{body}"


_router: Optional[KeylessSkillRouter] = None
_router_lock = threading.Lock()


def keyless_router() -> KeylessSkillRouter:
    """The process's keyless router, built on first use."""
    global _router
    with _router_lock:
        if _router is None:
            _router = KeylessSkillRouter()
        return _router


def _reset() -> None:
    """Drop the keyless router and the notices already given (tests)."""
    global _router
    with _router_lock:
        _router = None
    _announced.clear()
    _told.clear()


# ── starting keyless, and the guard ──────────────────────────────────────────

_announced: Set[str] = set()
#: Notices given once per process: (kind, ...) keys.
_told: Set[Tuple[str, ...]] = set()


def _once(*key: str) -> bool:
    """True the first time ``key`` is asked for in this process."""
    if key in _told:
        return False
    _told.add(key)
    return True


def announce(agent_name: str) -> None:
    """Say once per agent that it runs keyless, what it uses, and how to connect it."""
    if agent_name in _announced:
        return
    _announced.add(agent_name)
    _say(f"no DECIMAL_API_KEY, so agent {agent_name!r} runs keyless: no traces, and its skills "
         "are the ones .decimal/skills.toml lists (`decimalai skills add <name>` adds one). "
         "Add DECIMAL_API_KEY to connect it.")
    _tell_unlisted(agent_name)


def _tell_unlisted(agent_name: str, project: Any = None) -> None:
    """Say once per project which skill folders in ``.agents/skills`` its agents leave out.

    The folder is shared with coding tools (Codex, Cursor and others read it), so a
    folder there is not on an agent's list until ``decimalai skills add`` puts it there.
    One line, so a folder that is there for another tool costs one line per process.
    """
    try:
        from .skillpkg.project import Project

        project = project or Project.find()
        names = _authored.unlisted(project)
        if not names or not _once("unlisted", str(project.directory)):
            return
        shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
        _say(f"{project.rel(project.skills_root)} has skills this project's agents do not use: "
             f"{shown}. To give one to agent {agent_name!r}: decimalai skills add <name>")
    except Exception:
        logger.debug("keyless: could not list the unlisted skill folders", exc_info=True)


def link_path(project: Any = None) -> Path:
    from .skillpkg.project import Project

    project = project or Project.find()
    return project.state_dir / LINK_FILE


def read_link(project: Any = None) -> Optional[Dict[str, Any]]:
    """``.decimal/link.toml`` as a dict, or None when the project has none.

    A link file that cannot be parsed still counts as a link: it was written by a
    connected project, and treating damage as "never connected" is the quiet failure
    the guard exists to stop.
    """
    from .skillpkg.manifest import _toml

    path = link_path(project)
    if not path.is_file():
        return None
    try:
        data = _toml.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("keyless: %s does not parse", path, exc_info=True)
        return {"damaged": True}
    return data if isinstance(data, dict) else {"damaged": True}


def check_link(agent_name: str) -> None:
    """Refuse to run keyless in a project that was connected to a workspace.

    Raises:
        DecimalConfigError: ``.decimal/link.toml`` exists and no key is set.
    """
    from ._config import DecimalConfigError

    path = link_path()
    link = read_link()
    if link is None:
        return
    where = _link_label(link)
    raise DecimalConfigError(
        f"This project is connected to DecimalAI {where} ({path}), but no API key "
        f"is set, so agent_config({agent_name!r}) would run keyless and lose its saved prompt, "
        "skills and traces. Set DECIMAL_API_KEY (or DECIMALAI_API_KEY) to a key of that "
        "organization. "
        f"To run keyless on purpose, delete {path}."
    )


def _link_label(link: Mapping[str, Any]) -> str:
    """How the guard names a link file's organization and workspace."""
    org = link.get("org_slug") or link.get("org_id")
    workspace = link.get("workspace_id")
    if org and workspace:
        return f"organization {org} (workspace {workspace})"
    if org:
        return f"organization {org}"
    if workspace:
        return f"workspace {workspace}"
    return f"at {link.get('base_url') or DEFAULT_BASE_URL}"


#: What `link.toml` records for a skill you wrote that is on an agent (a registry skill
#: records the spec it was assigned at).
AUTHORED = "authored"


def _assigned(link: Optional[Mapping[str, Any]], agent_name: str) -> Dict[str, str]:
    """What ``link.toml`` records as given to ``agent_name`` from this project: a registry
    name -> the ``.decimal/skills.toml`` spec it was assigned at (``"3"`` or ``"latest"``),
    and a skill you wrote -> ``"authored"``. Empty when there is no record."""
    table = (link or {}).get("assigned")
    given = table.get(agent_name) if isinstance(table, dict) else None
    if not isinstance(given, dict):
        return {}
    return {k: v for k, v in given.items() if isinstance(k, str) and isinstance(v, str)}


def _has_record(link: Optional[Mapping[str, Any]], agent_name: str) -> bool:
    """False for a link written by 0.24, which connected with no record of what it assigned."""
    table = (link or {}).get("assigned")
    return isinstance(table, dict) and isinstance(table.get(agent_name), dict)


def _uploaded(link: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """The ``[uploaded]`` table: a skill you wrote -> the SHA-256 of its instructions as this
    project last uploaded them."""
    table = (link or {}).get("uploaded")
    if not isinstance(table, dict):
        return {}
    return {k: v for k, v in table.items() if isinstance(k, str) and isinstance(v, str)}


def _record_of(project: Any, link: Mapping[str, Any], agent_name: str) -> Dict[str, str]:
    """An agent's record, or for a 0.24 link (none kept) what the project lists now: that
    link assigned everything its file listed, so nothing in it is news."""
    if _has_record(link, agent_name):
        return _assigned(link, agent_name)
    record = {name: want.spec for name, want in _wanted(project).items()}
    record.update({name: AUTHORED for name in _authored.listed(project)})
    return record


def _write_link(project: Any, *, url: str, identity: Mapping[str, Optional[str]],
                agents: Sequence[str], assigned: Mapping[str, Mapping[str, str]],
                uploaded: Optional[Mapping[str, str]] = None) -> bool:
    """Write ``.decimal/link.toml``; True when its bytes changed."""
    from .skillpkg.manifest import _quote

    lines = [
        "# Written by decimalai on this project's first run with a DecimalAI API key.",
        "# Commit it: in this project, agent_config() with no API key now raises instead",
        "# of running keyless. Delete it to run keyless again.",
        f"base_url = {_quote(url)}",
    ]
    for key in ("org_id", "org_slug", "workspace_id"):
        if identity.get(key):
            lines.append(f"{key} = {_quote(str(identity[key]))}")
    names = sorted(set(agents))
    lines.append("agents = [" + ", ".join(_quote(a) for a in names) + "]")
    lines += [
        "",
        "# What each agent got from this project: the .decimal/skills.toml skills copied onto",
        "# it when it connected, and what `decimalai skills add` has given it since. Its skills",
        "# live on DecimalAI now: the dashboard and `decimalai skills add` change them, and",
        "# editing skills.toml does not.",
    ]
    for agent in names:
        lines.append(f"[assigned.{_quote(agent)}]")
        for name, spec in sorted((assigned.get(agent) or {}).items()):
            lines.append(f"{_quote(name)} = {_quote(spec)}")
    if uploaded:
        lines += [
            "",
            "# The instructions of each skill you wrote, as this project last uploaded them",
            "# (SHA-256). A run says when the file has changed since: `decimalai skills sync`.",
            "[uploaded]",
        ]
        for name, digest in sorted(uploaded.items()):
            lines.append(f"{_quote(name)} = {_quote(digest)}")
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path = link_path(project)
    if path.is_file() and path.read_bytes() == data:
        return False
    project.makedirs(project.state_dir)
    project.write_file(path, data)
    return True


def record(project: Any, *, agent_name: Optional[str] = None,
           assigned: Optional[Mapping[str, str]] = None,
           uploaded: Optional[Mapping[str, str]] = None) -> bool:
    """Add to a connected project's link file what a ``decimalai skills`` command just did:
    skills it gave ``agent_name``, and the instructions it uploaded. False (nothing written)
    when the project is not connected or its link file cannot be read."""
    link = read_link(project)
    if not link or link.get("damaged"):
        return False
    agents = [a for a in link.get("agents") or [] if isinstance(a, str)]
    if agent_name and agent_name not in agents:
        agents.append(agent_name)
    table = {a: _record_of(project, link, a) for a in agents}
    if agent_name and assigned:
        table[agent_name] = {**table.get(agent_name, {}), **assigned}
    identity = {key: link.get(key) for key in ("org_id", "org_slug", "workspace_id")}
    return _write_link(project, url=link.get("base_url") or base_url(), identity=identity,
                       agents=agents, assigned=table, uploaded={**_uploaded(link), **(uploaded or {})})


# ── the first keyed run ──────────────────────────────────────────────────────


def create_agent(agent_name: str) -> bool:
    """Create the agent the key's workspace has no agent by: ``POST /api/v1/agents``.

    No prompt goes with it (the code's prompt is never saved). A 409 means another
    process created it first, which is the same outcome. True when this call created it.
    """
    from ._client import DecimalAPIError
    from ._config import _get_client

    try:
        _get_client().create_agent(agent_name)
    except DecimalAPIError as exc:
        if getattr(exc, "status_code", None) != 409:
            raise
        return False
    _say(f"created agent {agent_name!r} in your DecimalAI workspace.")
    return True


def _identity(client: Any) -> Optional[Dict[str, Optional[str]]]:
    """Who the key is: its organization (``GET /api/v1/org``) and workspace
    (``GET /api/v1/auth/verify``, None for an organization-wide key). None when the
    key did not verify."""
    payload = client._request_with_retry("GET", "/api/v1/auth/verify").json()
    if not isinstance(payload, dict) or payload.get("api_key_valid") is not True:
        return None

    def text(value: Any) -> Optional[str]:
        return value if isinstance(value, str) and value else None

    org = client._request_with_retry("GET", "/api/v1/org", no_raise_statuses=frozenset({404}))
    org = org.json() if org.status_code == 200 else {}
    org = org if isinstance(org, dict) else {}
    return {"org_id": text(org.get("id")), "org_slug": text(org.get("slug")),
            "workspace_id": text(payload.get("workspace_id"))}


def connect(agent_name: str, *, router: Any, created: bool = False) -> None:
    """Connect this project to the key's workspace for ``agent_name`` once, then watch it.

    Runs for a project with something to connect: one with a ``.decimal/skills.toml``
    (it lists the project's skills), or whose agent this run just created. The first
    time (the agent is not in ``.decimal/link.toml`` yet) it copies the project's list
    onto the agent: each registry skill is assigned at the version the lock pins (a
    ``"latest"`` entry follows the latest), and each skill you wrote (``[authored]``) is
    uploaded to the organization, not published, and added to the agent. It then records
    the agent, the key's organization and workspace, and what it copied in
    ``.decimal/link.toml``. From then on the agent's skills live on DecimalAI: every later
    keyed start reads local files only, makes no request, changes nothing, and says once
    per process when the project lists a skill the agent was not given or a skill you
    wrote has changed since its upload. ``router`` is the SkillRouter to copy with, or a
    callable returning it, called only when there is something to copy. Only in a git
    checkout, the one place the file can be committed (a container built from the
    repository carries the committed file). Never raises: a copy that fails is said, and
    the agent runs either way.
    """
    from .skillpkg.project import Project

    try:
        project = Project.find()
        top = project.directory
        if not any(os.path.lexists(d / ".git") for d in (top, *top.parents)):
            return
        if not created and not project.manifest_path.is_file():
            return
        link = read_link(project)
        linked = link.get("agents") if isinstance(link, dict) else None
        if isinstance(linked, list) and agent_name in linked:
            _check(project, agent_name, link)
        else:
            _connect(project, agent_name, link if isinstance(link, dict) else None, router)
    except Exception:
        logger.warning("decimalai: could not connect this project for agent %r (continuing); "
                       "the next run tries again.", agent_name, exc_info=True)


@dataclass(frozen=True)
class _Wanted:
    """One ``.decimal/skills.toml`` skill as an assignment: what to send, and what to record."""

    skill_id: str
    pin_version_id: Optional[str]    # None follows the latest
    spec: str                        # the manifest's spec, "3" or "latest"
    label: str                       # how a message names it


def _wanted(project: Any) -> Dict[str, _Wanted]:
    """The project's skills by registry name: each manifest entry the lock has resolved."""
    from .skillpkg import lock as _lock
    from .skillpkg import manifest as _manifest

    if not project.manifest_path.is_file():
        return {}
    wanted = _manifest.load(project, missing_ok=True)
    out: Dict[str, _Wanted] = {}
    for entry in (_lock.locked_entries(project) or {}).values():
        name = entry.get("name")
        if not isinstance(name, str) or name not in wanted:
            continue
        spec = wanted[name].spec
        pinned = not wanted[name].latest
        out[name] = _Wanted(
            skill_id=str(entry["skill_id"]),
            pin_version_id=str(entry["version_id"]) if pinned else None,
            spec=spec,
            label=f"{name} v{entry.get('version')}" if pinned else f"{name} (latest)",
        )
    return out


def _elsewhere(link: Mapping[str, Any], url: str, identity: Mapping[str, Optional[str]]) -> bool:
    """Whether ``link`` names another platform, organization or workspace than this key's."""
    return link.get("base_url") not in (None, url) or any(
        link.get(key) and identity.get(key) and link.get(key) != identity.get(key)
        for key in ("org_id", "workspace_id"))


def _leave_elsewhere(project: Any, link: Mapping[str, Any], url: str,
                     identity: Mapping[str, Optional[str]]) -> None:
    logger.warning(
        "decimalai: %s connects this project to %s at %s; this key is %s at %s. Left "
        "as it is: delete the file to connect the project to this one.",
        link_path(project), _link_label(link), link.get("base_url") or "?",
        _link_label(identity), url)


def _assign(router: Any, agent_name: str, todo: Mapping[str, _Wanted]
            ) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Assign each of ``todo`` to the agent: ``(assigned name -> spec, labels, failures)``.

    A failure is not recorded, so the next keyed start tries it again."""
    done: Dict[str, str] = {}
    labels: List[str] = []
    failed: List[str] = []
    for name, want in sorted(todo.items()):
        try:
            router.assign_skill(agent_name, want.skill_id, pin_version_id=want.pin_version_id)
        except Exception as exc:
            logger.debug("assign_skill failed", exc_info=True)
            failed.append(f"{name} ({str(exc)[:120]})")
            continue
        done[name] = want.spec
        labels.append(want.label)
    return done, labels, failed


def _say_failed(agent_name: str, failed: Sequence[str]) -> None:
    """Name what the one-time copy could not put on the agent. Nothing retries it: the
    agent's skills live on DecimalAI from now on, so the person adds it there."""
    if failed:
        _say(f"agent {agent_name!r}: could not copy {'; '.join(failed)} onto it. Add "
             f"{'it' if len(failed) == 1 else 'them'} with `decimalai skills add <name>` or on "
             "the agent's page in the dashboard.")


def _own(project: Any) -> Dict[str, Dict[str, Any]]:
    """The skills you wrote that the project lists, by folder name: each parsed SKILL.md."""
    from .skills import parse_skill_md

    managed = _authored.managed(project)
    out: Dict[str, Dict[str, Any]] = {}
    for name in _authored.listed(project):
        if name in managed:
            continue
        path = _authored.skill_md(project, name)
        try:
            descriptor = parse_skill_md(str(path)) if path.is_file() else None
        except Exception:
            descriptor = None
        if descriptor:
            out[name] = descriptor
    return out


def sync_authored(router: Any, items: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    """``POST /skills/sync`` the skills you wrote, the platform's copy winning a conflict.

    ``conflict_policy="remote_wins_on_conflict"`` (with the checkout's ``install_id``):
    a body only this checkout changed is uploaded; one the platform changed is not
    overwritten. A platform from before that policy answers 422, and is asked once more
    with ``newer_wins``, which it has always taken. Any 422 is asked again: the error the
    router raises keeps only the envelope's message ("Request validation failed"), not
    the field it names, and a request refused for another reason is refused again, with
    that second error raised.
    """
    from .skill_router import SkillRouterError

    try:
        return router.sync_skills(items, conflict_policy="remote_wins_on_conflict",
                                  response_mode="diff", **kwargs)
    except SkillRouterError as exc:
        if getattr(exc, "status_code", None) != 422:
            raise
    return router.sync_skills(items, conflict_policy="newer_wins", response_mode="diff", **kwargs)


def _upload(project: Any, router: Any, agent_name: str, own: Mapping[str, Dict[str, Any]]
            ) -> Tuple[Dict[str, str], List[str], List[str], Dict[str, str]]:
    """Upload the skills you wrote and add them to the agent, in one sync:
    ``(record, labels, failures, uploaded hashes)``."""
    from skillevaluation import skillfmt

    from ._install import get_install_identity
    from .skill_router import _sync_up_items
    from .skills import _with_local_timestamps

    items, _managed = _sync_up_items(_with_local_timestamps(list(own.values())))
    if not items:
        return {}, [], [], {}
    folder_of = {d.get("name"): folder for folder, d in own.items()}
    try:
        ident = get_install_identity(str(project.directory))
    except Exception:
        ident = {}
    try:
        result = sync_authored(router, items, install_id=ident.get("install_id"),
                               install_label=ident.get("install_label"),
                               agent_name=agent_name, add_to_agent="all")
    except Exception as exc:
        logger.debug("sync_skills failed", exc_info=True)
        return {}, [], [f"{', '.join(sorted(own))} ({str(exc)[:120]})"], {}
    if result.get("agent_not_found"):
        why = result.get("agent_problem") or "the agent could not be found"
        return {}, [], [f"{', '.join(sorted(own))} ({why})"], {}
    failed_names = {f.get("name") for f in result.get("failures") or [] if isinstance(f, dict)}
    actions = {a.get("name"): a for a in result.get("actions") or [] if isinstance(a, dict)}
    done: Dict[str, str] = {}
    labels: List[str] = []
    failed: List[str] = []
    uploaded: Dict[str, str] = {}
    for item in items:
        name = item["name"]
        folder = folder_of.get(name, name)
        action = actions.get(name) or {}
        if name in failed_names or action.get("action") == "failed":
            failed.append(f"{folder} (upload failed)")
            continue
        done[folder] = AUTHORED
        if action.get("action") == "pulled":
            # The organization's copy was changed on DecimalAI: it stays, and the agent
            # uses it. Recording its text makes the next start say the file differs.
            uploaded[folder] = skillfmt.body_sha256(action.get("body_markdown") or "")
            labels.append(f"{folder} (DecimalAI's copy, which differs from yours)")
        else:
            uploaded[folder] = skillfmt.body_sha256(item["body_markdown"])
            labels.append(f"{folder} (uploaded)")
    return done, labels, failed, uploaded


def _say_edited(project: Any, agent_name: str, wanted: Mapping[str, _Wanted]) -> None:
    """Name each registry skill whose folder was edited: the agent gets the registry's text."""
    from .skillpkg import fs
    from .skillpkg import lock as _lock

    try:
        entries = _lock.locked_entries(project) or {}
    except Exception:
        return
    for folder, entry in sorted(entries.items()):
        name = entry.get("name")
        if name not in wanted:
            continue
        try:
            state = fs.check_folder(project.skill_dir(folder), entry, name=folder)
        except Exception:
            continue
        if state.state is fs.State.MODIFIED:
            version = entry.get("version")
            text = f"v{version}" if isinstance(version, int) else "version"
            _say(f"{name} was edited in {project.rel(project.skill_dir(folder))}, but agent "
                 f"{agent_name!r} uses the registry's {text}: the edit does not carry over. To "
                 "keep it, fork the skill on DecimalAI (Fork a copy, on its page) and give the "
                 "fork to the agent.")


def _connect(project: Any, agent_name: str, link: Optional[Dict[str, Any]], router: Any) -> None:
    from ._config import _get_client, _get_config

    url = _get_config().base_url.rstrip("/")
    identity = _identity(_get_client())
    if identity is None:
        return
    if link and not link.get("damaged") and _elsewhere(link, url, identity):
        _leave_elsewhere(project, link, url, identity)
        return
    wanted = _wanted(project)
    own = _own(project)
    done: Dict[str, str] = {}
    labels: List[str] = []
    failed: List[str] = []
    uploaded: Dict[str, str] = {}
    if wanted or own:
        router = router() if callable(router) else router
        if router is not None:
            if wanted:
                assigned, said, refused = _assign(router, agent_name, wanted)
                done.update(assigned)
                labels += said
                failed += refused
            if own:
                assigned, said, refused, uploaded = _upload(project, router, agent_name, own)
                done.update(assigned)
                labels += said
                failed += refused
    usable = link if link and not link.get("damaged") else {}
    agents = list(usable.get("agents") or []) if isinstance(usable.get("agents"), list) else []
    agents = [a for a in agents if isinstance(a, str)]
    assigned_all = {a: _record_of(project, usable, a) for a in agents}
    assigned_all[agent_name] = done
    kept = {key: identity.get(key) or usable.get(key) for key in ("org_id", "org_slug", "workspace_id")}
    changed = _write_link(project, url=url, identity=kept, agents=[*agents, agent_name],
                          assigned=assigned_all, uploaded={**_uploaded(usable), **uploaded})
    if changed:
        copied = f"copied {', '.join(labels)} onto it. " if labels else ""
        _say(f"connected agent {agent_name!r} to your DecimalAI workspace: {copied}From now on "
             "its skills live on DecimalAI: change them in the dashboard or with `decimalai "
             f"skills add`; editing {project.rel(project.manifest_path)} no longer changes them. "
             f"Wrote {project.rel(link_path(project))}: commit it.")
    _say_edited(project, agent_name, wanted)
    _say_failed(agent_name, failed)


def _check(project: Any, agent_name: str, link: Dict[str, Any]) -> None:
    """A connected agent's later keyed start: reads local files only, writes nothing,
    asks nothing. Says once per process what the project lists that the agent was not
    given, and which skills you wrote changed since this project uploaded them."""
    if not _has_record(link, agent_name):
        given = _record_of(project, link, agent_name)  # a 0.24 link: the file is the record
    else:
        given = _assigned(link, agent_name)
    wanted = _wanted(project)
    listed = [n for n in _authored.listed(project) if _authored.is_own_folder(project, n)]
    manifest = project.rel(project.manifest_path)
    missing = sorted([n for n in wanted if n not in given] + [n for n in listed if n not in given])
    if missing and _once("missing", agent_name, *missing):
        _say(f"agent {agent_name!r} gets its skills from DecimalAI, and {manifest} lists "
             f"{', '.join(missing)}, which it was not given. Add one with `decimalai skills add "
             "<name>` or in the dashboard; editing the file does not.")
    repinned = sorted(n for n, w in wanted.items()
                      if n in given and given[n] not in (w.spec, AUTHORED))
    for name in repinned:
        spec = wanted[name].spec
        if _once("repinned", agent_name, name, spec):
            pin = "the latest version" if spec == "latest" else f"v{spec}"
            _say(f"{manifest} asks for {name} at {pin}; agent {agent_name!r}'s version of it is "
                 f"set on DecimalAI: `decimalai skills add {name}@{spec}` or the dashboard.")
    uploaded = _uploaded(link)
    edited = sorted(n for n in listed if n in uploaded
                    and _authored.body_hash(project, n) not in (None, uploaded[n]))
    if edited and _once("edited", agent_name, *edited):
        one = len(edited) == 1
        _say(f"{', '.join(edited)} in {project.rel(project.skills_root)} "
             f"{'does' if one else 'do'} not match DecimalAI's copy, which agent {agent_name!r} "
             f"uses: run `decimalai skills sync` to sync {'it' if one else 'them'} (a change "
             "made on DecimalAI wins).")
    _tell_unlisted(agent_name, project)
