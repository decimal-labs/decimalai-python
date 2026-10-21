"""Keyless ``agent_config()``: an agent that runs before there is a DecimalAI account.

With no ``DECIMAL_API_KEY`` (or ``DECIMALAI_API_KEY``) set and ``decimalai.init()``
never called, every ``agent_config()`` runs keyless:

- the prompt is the ``fallback=`` text, or none (the instructions are then the skill
  list alone); ``prompt_source`` is ``"fallback (no key)"``;
- each run's skill list is the project's own skills (``.decimal/skills.toml``) plus at
  most two matches from the public registry, found by searching it anonymously with
  the run's message (:func:`relevant_matches` is the cutoff). Only a skill that passed
  its safety review and has a benchmark is offered;
- when the model loads a registry skill, it is added to the project through the
  engine behind ``decimalai skills add`` (anonymous, refused unless its safety review
  is clean), pinned to the version offered in ``.decimal/skills.toml`` and
  ``skills.lock``. One line on stderr names it. Later runs read it from disk;
- nothing else is sent: no traces, no routing decisions, no prompt reads.

Adding a key connects the project (:func:`connect`, on the first keyed
``agent_config()`` of each agent): the agent is created when the workspace has none
by that name, the project's skills are assigned to it at their pinned versions, and
``.decimal/link.toml`` records the workspace. From then on a run with no key in that
project raises (:func:`check_link`) rather than quietly running keyless: that file,
committed, is what protects a production deploy that lost its key. The prompt in the
code is never saved to DecimalAI: a saved prompt overrides the code's, which is
exactly what the code's author did not ask for.
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

from .skillpkg.client import RegistryClient

logger = logging.getLogger("decimalai")

#: The link file, in the project's ``.decimal/`` beside ``skills.toml``.
LINK_FILE = "link.toml"

DEFAULT_BASE_URL = "https://api.decimal.ai"

#: The anonymous registry search a keyless run makes, and the only route besides the
#: install routes (resolve, bundle) a keyless process calls. The message goes in the
#: request body: a URL is written to access logs and keyed into caches at every hop.
#: A registry from before this route (it answers 404 or 405) is not searched at all:
#: its ``GET /api/v1/registry/skills?q=`` would put each message in a URL, and the
#: registry whose logs that fills is api.decimal.ai itself, rolled back past 2026-10-21.
SEARCH_POST_PATH = "/api/v1/registry/skills/search"

#: How long a run waits for the search before it lists the project's skills alone.
#: Measured 2026-10-21 from a laptop: 1.1-3.5 s for a new message, one cold outlier
#: at 36 s. A run must not wait on that outlier.
_SEARCH_TIMEOUT_S = 5.0
_SEARCH_CONNECT_S = 2.0
#: The search's page size. The cutoff reads only the first `_SEARCH_WINDOW` results;
#: the page is a little larger because the server sizes its semantic pool from it.
_SEARCH_LIMIT = 10
#: The longest message sent as the query. The search ranks on meaning, and the first
#: few sentences carry it; a pasted log or email would only make the request long.
_QUERY_MAX_CHARS = 500
#: Registries (base URLs) that answered the POST search 404 or 405: this process does not
#: search them again.
_POST_UNSUPPORTED: Set[str] = set()

#: The relevance cutoff (see `relevant_matches`).
_SEARCH_WINDOW = 5
_MAX_REGISTRY_OFFERS = 2

#: The project's own skills listed on a run, the platform's menu cap.
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
# invented so the two agree on what "shares a word" means.

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


def _slug(name: str) -> str:
    return name.rsplit("/", 1)[-1].strip().lower()


def eligible(item: Mapping[str, Any]) -> bool:
    """A registry skill keyless mode may offer: it passed its safety review and has a benchmark.

    The public listing deliberately includes skills not yet reviewed (it hides only
    negative verdicts), so the filter has to be explicit here.
    """
    return item.get("skill_safety") == "passed" and bool(item.get("benchmark_summary"))


def relevant_matches(query: Optional[str], items: Sequence[Mapping[str, Any]],
                     *, exclude: Sequence[str] = ()) -> List[Mapping[str, Any]]:
    """The registry skills a run offers, from the search's answer for its message.

    The search answers with no relevance score, so the cutoff is built from what the
    answer does carry. Within the first `_SEARCH_WINDOW` results (the server puts its
    meaning matches first), keep a skill that is `eligible` and shares at least one
    content word with the message (`word_overlap`, the platform's own floor), one per
    slug (an import and its copy are one skill to the model), at most
    `_MAX_REGISTRY_OFFERS`, and none the project already has.

    Measured 2026-10-21 on 12 support messages, hand-labeled, twice (the search's
    order shifts between identical calls): 16-17 of 21 offers were on the message's
    topic (76-81%) and 8-10 of 21 would help answer it (38-48%; many on-topic skills
    are written for the store operator, such as "write a refund policy page"). Small
    talk ("Hi, thanks for your help!") offered nothing. Wider windows and a third
    offer each lowered both numbers; one offer halved the useful ones.
    """
    terms = query_terms(query)
    if not terms:
        return []
    taken = {_slug(n) for n in exclude}
    picked: List[Mapping[str, Any]] = []
    for item in list(items)[:_SEARCH_WINDOW]:
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or not eligible(item):
            continue
        if word_overlap(terms, name, item.get("description")) < 1:
            continue
        slug = _slug(name)
        if slug in taken:
            continue
        taken.add(slug)
        picked.append(item)
        if len(picked) >= _MAX_REGISTRY_OFFERS:
            break
    return picked


def search_registry(query: str, *, url: Optional[str] = None, transport: Any = None) -> List[Dict[str, Any]]:
    """The anonymous registry search for one message: its result items, in order.

    ``POST /api/v1/registry/skills/search`` with the message in the body. A registry
    that answers it 404 or 405 predates the route: it is not searched (nothing found,
    said once), and never asked with the ``GET``, which carries the message in its URL.
    Raises on any other failure; the caller lists the project's skills alone.
    """
    import httpx

    from ._config import sdk_user_agent

    root = (url or base_url()).rstrip("/")
    if root in _POST_UNSUPPORTED:
        return []
    ask = {"q": query[:_QUERY_MAX_CHARS], "limit": _SEARCH_LIMIT}
    with httpx.Client(
        base_url=root,
        timeout=httpx.Timeout(_SEARCH_TIMEOUT_S, connect=_SEARCH_CONNECT_S),
        headers={"User-Agent": sdk_user_agent("keyless")},
        transport=transport,
    ) as client:
        resp = client.post(SEARCH_POST_PATH, json=ask)
        if resp.status_code in (404, 405):
            _POST_UNSUPPORTED.add(root)
            logger.warning(
                "decimalai: the registry at %s has no POST %s (it predates 2026-10-21), so "
                "keyless runs list the project's own skills only: its GET search would put "
                "each message in a URL.", root, SEARCH_POST_PATH)
            return []
        resp.raise_for_status()
        items = resp.json().get("items")
    if not isinstance(items, list):
        raise ValueError("the registry search answered without an items list")
    return [i for i in items if isinstance(i, dict)]


# ── the project's own skills ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Offer:
    """One skill a run lists: the project's own (``path`` set) or a registry match."""

    name: str
    description: str
    path: Optional[Path] = None          # the installed SKILL.md
    version: Optional[int] = None        # the registry version offered or installed


def _one_line(text: Any) -> str:
    line = " ".join(str(text or "").split()).replace("|", "/")
    if len(line) > _MAX_DESCRIPTION_CHARS:
        line = line[: _MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
    return line


def project_skills(project: Any = None) -> Dict[str, Offer]:
    """The project's own skills: each entry of ``.decimal/skills.toml`` installed on disk.

    Keyed by registry name, the name the manifest and the menu use. A skill listed in
    the manifest but not installed (``decimalai skills install`` not run yet) is left
    out: there is no text to serve.
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
    out: Dict[str, Offer] = {}
    for install_name, entry in sorted(entries.items()):
        name = entry.get("name")
        if not isinstance(name, str) or name not in wanted:
            continue
        path = project.skill_dir(install_name) / "SKILL.md"
        try:
            parsed = parse_skill_md(str(path)) if path.is_file() else None
        except Exception:
            parsed = None
        if not parsed:
            continue
        out[name] = Offer(name=name, description=_one_line(parsed.get("description")), path=path,
                          version=entry.get("version") if isinstance(entry.get("version"), int) else None)
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


def _split_frontmatter(text: str) -> Optional[Tuple[str, str]]:
    from skillevaluation import skillfmt

    return skillfmt.split_frontmatter(text)


class _FetchOnce(RegistryClient):
    """A registry client that fetches each rendering once, so the bundle a load checked
    is the bundle the install engine writes (and the registry counts one fetch)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bundles: Dict[Tuple[str, str, str], Any] = {}

    def fetch_resolved(self, resolved: Any, *, purpose: str) -> Any:
        key = (resolved.version_id, resolved.install_name, resolved.render.hash)
        if key not in self._bundles:
            self._bundles[key] = super().fetch_resolved(resolved, purpose=purpose)
        return self._bundles[key]



class KeylessSkillRouter:
    """Lists and serves a keyless agent's skills: the project's, and the registry's on use.

    It stands where an agent's ``SkillRouter`` stands (`AgentBinding` reads the same few
    methods and attributes), and records nothing: a keyless run is sent nowhere.

    A registry skill is served only to a load whose run listed it, as the platform
    serves one only to a load whose routing decision offered it: a message that talks
    the model into loading some other public skill gets "no skill named", and nothing
    is added to the project. The project's own skills are served to any load.
    """

    def __init__(self, *, url: Optional[str] = None, transport: Any = None,
                 project_dir: Optional[str] = None) -> None:
        self.base_url = (url or base_url()).rstrip("/")
        self._transport = transport
        self._project_dir = project_dir
        self._rail_lock = threading.Lock()
        #: run key -> the routing decision its loads name.
        self._run_routing_ids: "OrderedDict[str, str]" = OrderedDict()
        #: routing decision -> the registry skills it offered, by name.
        self._decisions: "OrderedDict[str, Dict[str, Offer]]" = OrderedDict()
        #: run key -> names whose text that run was served.
        self._loads: "OrderedDict[str, Set[str]]" = OrderedDict()
        self._add_lock = threading.Lock()
        #: Registry skills refused at their first load (unsafe, or too long): not listed again.
        self._unusable: Set[str] = set()
        self._search_warned = False

    def __repr__(self) -> str:
        return f"<KeylessSkillRouter {self.base_url}>"

    def _project(self) -> Any:
        from .skillpkg.project import Project

        return Project.find(self._project_dir)

    # ── one run's list ──────────────────────────────────────
    def build_prompt_parts(
        self, query: Optional[str] = None, *, agent_name: Optional[str] = None,
        inject_body: Optional[bool] = None, scope: Optional[str] = None, **_ignored: Any,
    ) -> Tuple[str, str, Optional[str]]:
        """``(menu, "", routing id)`` for one run, as `SkillRouter.build_prompt_parts`.

        The list depends on the message, so it reports itself cut: the binding then
        lists every run's own matches before its model call, and its ``load_skill``
        takes any name rather than an enum of one run's list.
        """
        from .skill_router import (
            _last_delivered_names_ctx,
            _last_menu_truncated_ctx,
            _last_offered_names_ctx,
        )

        local = project_skills(self._project())
        found: List[Mapping[str, Any]] = []
        if query and not inject_body:
            # With load_skill off nothing could ever add a registry skill to the
            # project, so it is not listed.
            with self._rail_lock:
                skip = [*local, *self._unusable]
            found = relevant_matches(query, self._search(query), exclude=skip)
        registry = {
            str(item["name"]): Offer(
                name=str(item["name"]), description=_one_line(item.get("description")),
                version=item.get("latest_version_number")
                if isinstance(item.get("latest_version_number"), int) else None)
            for item in found
        }
        offers = list(local.values()) + list(registry.values())
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
        with self._rail_lock:
            self._decisions[routing_id] = registry
            while len(self._decisions) > _MAX_DECISIONS:
                self._decisions.popitem(last=False)
        self._remember_run_routing(scope, routing_id)
        _last_offered_names_ctx.set([o.name for o in offers])
        _last_menu_truncated_ctx.set(True)
        _last_delivered_names_ctx.set(delivered or None)
        return menu, "", routing_id

    def _search(self, query: str) -> List[Dict[str, Any]]:
        try:
            return search_registry(query, url=self.base_url, transport=self._transport)
        except Exception as exc:
            if not self._search_warned:
                self._search_warned = True
                logger.warning(
                    "decimalai: the public registry search failed (%s); keyless runs list "
                    "the project's own skills until it answers.", exc)
            else:
                logger.debug("keyless: registry search failed", exc_info=True)
            return []

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
        """One skill's text for the model, adding a registry skill to the project first.

        Always a string the model can act on, as `SkillRouter.load_skill`.
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
        project = self._project()
        offer = project_skills(project).get(name)
        if offer is None:
            listed = self._listed(scope, name)
            if listed is not None:
                added = self._add(project, listed, agent_name)
                if isinstance(added, str):
                    return added
                offer = added
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

    def _listed(self, scope: Optional[str], name: str) -> Optional[Offer]:
        """The registry skill ``name`` as this run's list offered it, or None."""
        with self._rail_lock:
            routing_id = self._run_routing_ids.get(scope) if scope is not None else None
            decision = self._decisions.get(routing_id) if routing_id else None
            return (decision or {}).get(name)

    def _add(self, project: Any, offer: Offer, agent_name: Optional[str]) -> Any:
        """Add a listed registry skill to the project, pinned to its current version: its
        `Offer` on disk, or the sentence the model gets instead.

        Checked before anything is written: the version's safety review must be clean
        (the install engine's own gate, without its question: a run has no one to ask)
        and its text must fit the per-skill limit a keyed run's router enforces too. A
        skill that fails either is left out of the project, and of this process's
        later lists. The bundle fetched for the check is the one installed.
        """
        from .skillpkg import ops
        from .skillpkg.client import ResolveRequest, trust_verdict
        from .skillpkg.errors import SkillpkgError

        with self._add_lock:
            # Another run may have added it while this one waited.
            done = project_skills(project).get(offer.name)
            if done is not None:
                return done
            client = _FetchOnce(api_key=None, base_url=self.base_url, channel="sdk",
                                transport=self._transport)
            try:
                resolved = client.resolve_one(ResolveRequest(ref=offer.name), purpose="add")
                refusal, caution = trust_verdict(resolved)
                if refusal or caution:
                    return self._refuse(offer.name, f"its safety review is {refusal or caution}")
                text = client.fetch_resolved(resolved, purpose="add").skill_md.decode("utf-8")
                split = _split_frontmatter(text)
                body = (split[1] if split else text).strip()
                if len(body) > _BODY_CHAR_LIMIT:
                    return self._refuse(offer.name, (
                        f"its instructions are {len(body)} characters, over the {_BODY_CHAR_LIMIT}-"
                        "character limit for one skill"))
                result = ops.add(offer.name, version=resolved.version, yes=False, interactive=False,
                                 directory=str(project.directory), client=client)
            except SkillpkgError as exc:
                result, why = None, str(exc)
            finally:
                client.close()
            if result is not None and int(result.exit_code) != 0:
                why = "; ".join(str(p) for p in result.problems if isinstance(p, SkillpkgError))
                why = why or f"exit code {int(result.exit_code)}"
            if result is None or int(result.exit_code) != 0:
                logger.warning("decimalai: could not add %s to this project: %s", offer.name, why)
                return (f"load_skill: {offer.name!r} could not be added to this project ({why}). "
                        "Proceed without it.")
            added = project_skills(project).get(offer.name)
            if added is None:
                return f"load_skill: {offer.name!r} was added but cannot be read. Proceed without it."
            who = f" (first loaded by agent {agent_name!r})" if agent_name else ""
            _say(f"added skill {offer.name} v{resolved.version} to .decimal/skills.toml{who}.")
            return added

    def _refuse(self, name: str, why: str) -> str:
        """Leave a registry skill out of the project and out of this process's lists."""
        with self._rail_lock:
            self._unusable.add(name)
        logger.warning("decimalai: did not add %s to this project: %s.", name, why)
        return f"load_skill: {name!r} cannot be used here ({why}). Proceed without it."


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
    _POST_UNSUPPORTED.clear()


# ── starting keyless, and the guard ──────────────────────────────────────────

_announced: Set[str] = set()


def announce(agent_name: str) -> None:
    """Say once per agent that it runs keyless, and how to connect it."""
    if agent_name in _announced:
        return
    _announced.add(agent_name)
    _say(f"no DECIMAL_API_KEY, so agent {agent_name!r} runs keyless: no traces, and its skills "
         "come from this project and the public registry. Add DECIMAL_API_KEY to connect it.")


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


def _write_link(project: Any, *, url: str, identity: Mapping[str, Optional[str]],
                agents: Sequence[str]) -> bool:
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
    lines.append("agents = [" + ", ".join(_quote(a) for a in sorted(set(agents))) + "]")
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path = link_path(project)
    if path.is_file() and path.read_bytes() == data:
        return False
    project.makedirs(project.state_dir)
    project.write_file(path, data)
    return True


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
    """Connect this project to the key's workspace for ``agent_name``, once.

    Runs for a project with something to connect: one that lists skills in
    ``.decimal/skills.toml`` (a keyless run adds them there), or whose agent this run
    just created. Assigns each of those skills to the agent at the version the lock
    pins (``assign_skill`` with the lock's version id; a ``"latest"`` entry follows the
    latest), then records the agent, the key's organization and its workspace in
    ``.decimal/link.toml``. ``router`` is the SkillRouter to assign with, or a callable
    returning it, called only when there is something to assign. Only in a git checkout,
    the one place the file can be committed (a container built from the repository
    carries the committed file). Never raises: a project that cannot be connected now
    (the platform did not answer) is connected by a later run, and the agent runs
    either way.
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
            return
        _connect(project, agent_name, link if isinstance(link, dict) else None, router)
    except Exception:
        logger.warning("decimalai: could not connect this project for agent %r (continuing); "
                       "the next run tries again.", agent_name, exc_info=True)


def _connect(project: Any, agent_name: str, link: Optional[Dict[str, Any]], router: Any) -> None:
    from ._config import _get_client, _get_config
    from .skillpkg import lock as _lock
    from .skillpkg import manifest as _manifest

    url = _get_config().base_url.rstrip("/")
    identity = _identity(_get_client())
    if identity is None:
        return
    if link and not link.get("damaged"):
        elsewhere = link.get("base_url") not in (None, url) or any(
            link.get(key) and identity.get(key) and link.get(key) != identity.get(key)
            for key in ("org_id", "workspace_id"))
        if elsewhere:
            logger.warning(
                "decimalai: %s connects this project to %s at %s; this key is %s at %s. Left "
                "as it is: delete the file to connect the project to this one.",
                link_path(project), _link_label(link), link.get("base_url") or "?",
                _link_label(identity), url)
            return
    assigned: List[str] = []
    failed: List[str] = []
    router = router() if callable(router) else router
    if project.manifest_path.is_file() and router is not None:
        wanted = _manifest.load(project, missing_ok=True)
        entries = _lock.locked_entries(project) or {}
        for entry in entries.values():
            name = entry.get("name")
            if not isinstance(name, str) or name not in wanted:
                continue
            pinned = not wanted[name].latest
            try:
                router.assign_skill(agent_name, str(entry["skill_id"]),
                                    pin_version_id=str(entry["version_id"]) if pinned else None)
                assigned.append(f"{name} v{entry.get('version')}" if pinned else f"{name} (latest)")
            except Exception as exc:
                logger.debug("assign_skill failed", exc_info=True)
                failed.append(f"{name} ({str(exc)[:120]})")
    agents = list(link.get("agents") or []) if link and isinstance(link.get("agents"), list) else []
    kept = {key: identity.get(key) or (link or {}).get(key) for key in ("org_id", "org_slug", "workspace_id")}
    changed = _write_link(project, url=url, identity=kept, agents=[*agents, agent_name])
    if not changed:
        return
    done = f"assigned {', '.join(assigned)}; " if assigned else ""
    _say(f"connected agent {agent_name!r} to your workspace: {done}wrote "
         f"{project.rel(link_path(project))}. Commit it.")
    if failed:
        # Said once: the link is written, so no later run tries again on its own.
        _say(f"agent {agent_name!r}: could not assign {'; '.join(failed)}. Assign them from "
             "the agent's Skills tab in the dashboard.")
