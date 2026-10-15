"""Identity on the wire: what a native loader reports, as the registry's skill ids.

A skill ``decimalai skills`` installed sits in ``.agents/skills/<install-name>/``, and a
loader reports it by what it sees there: the install name (``acme-refund-policy``), the
name Claude Code gives a plugin skill (``decimalai-skills:acme-refund-policy``, the prefix
being the ``name`` in ``.agents/.claude-plugin/plugin.json``), or the folder's path. The
platform's name lookups know the registry name (``acme/refund-policy``), so without a
translation those activations resolve to no skill at all.

``ManagedIndex`` maps those spellings, the locked body hash and the skill id to the lock
entry's identity, and ``translate_payload`` rewrites a trace payload with it. The fixtures
are rendered by ``skillevaluation.skillfmt`` and written with the engine's own lock module,
so these tests read exactly what an install leaves on disk.
"""
from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path

import pytest
from skillevaluation import skillfmt

from decimalai.skillpkg import identity as I
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import project as P

INSTALLED_AT = "2026-10-20T10:00:00Z"
ACME_BODY = "# Refunds\n\nThe refund window is thirty days from delivery.\n"
PDF_BODY = "# PDF tools\n\nExtract text with pdftotext before summarising.\n"
REVIEW_BODY = "# Code review\n\nRead the diff twice before commenting.\n"


def _id(n: int) -> str:
    return str(uuid.UUID(int=n))


ACME_ID, ACME_VID = _id(1), _id(101)
PDF_ID, PDF_VID = _id(2), _id(202)
REVIEW_ID, REVIEW_VID = _id(3), _id(303)
#: The public ``pdf-tools`` installed beside a folder that already holds the base name.
PDF_INSTALL = skillfmt.install_name_disambiguated("pdf-tools", PDF_ID)


def _bundle(install_name: str, registry: str, sid: str, vid: str, body: str) -> skillfmt.Bundle:
    fields = skillfmt.RenderFields(
        skill_id=sid, version_id=vid, version=3, registry_name=registry, body=body,
        description=f"Use {registry}.")
    return skillfmt.render_bundle(fields, [], install_name=install_name, variant="foreign")


def _entry(bundle: skillfmt.Bundle, registry: str) -> dict:
    return L.make_entry(
        bundle.install_name,
        name=registry,
        skill_id=bundle.skill_id,
        version_id=bundle.version_id,
        version=3,
        render={"format": bundle.format, "hash": bundle.render_hash, "variant": bundle.variant},
        body_sha256=bundle.body_sha256,
        description_sha256=bundle.description_sha256,
        files={f.path: f.sha256 for f in bundle.files},
        source={"base_url": "https://api.example.test", "org": None},
        invocation=bundle.invocation,
        installed_at=INSTALLED_AT,
    )


SKILLS = {
    "acme-refund-policy": ("acme/refund-policy", ACME_ID, ACME_VID, ACME_BODY),
    PDF_INSTALL: ("pdf-tools", PDF_ID, PDF_VID, PDF_BODY),
    "code-review": ("code-review", REVIEW_ID, REVIEW_VID, REVIEW_BODY),
}


def _install(directory: Path, skills=None, *, plugin_json: bool = True) -> L.Lock:
    """Write the folders, the lock and plugin.json as an install leaves them."""
    project = P.Project.at(directory)
    lock = L.empty()
    for install_name, (registry, sid, vid, body) in (skills or SKILLS).items():
        bundle = _bundle(install_name, registry, sid, vid, body)
        folder = project.skill_dir(install_name)
        for f in bundle.files:
            path = folder / f.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f.content.encode("utf-8"))
        lock = lock.with_entry(install_name, _entry(bundle, registry))
    L.save(project, lock)
    if plugin_json:
        L.write_plugin_json(project, lock.lock_hash)
    return lock


def _hand_authored(directory: Path, name: str, body: str = "# Mine\n\nA skill written by hand.\n"):
    folder = directory / ".agents" / "skills" / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: mine\n---\n{body}",
                                     encoding="utf-8")
    return folder


def _bump_mtime(path: Path, seconds: int = 5) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


@pytest.fixture(autouse=True)
def _fresh_cache():
    I.clear_cache()
    yield
    I.clear_cache()


@pytest.fixture
def project(tmp_path) -> Path:
    (tmp_path / ".git").mkdir()  # the repository: the search for a project stops here
    _hand_authored(tmp_path, "pdf-tools")  # why the public pdf-tools got the hashed name
    _install(tmp_path)
    return tmp_path


@pytest.fixture
def index(project) -> I.ManagedIndex:
    return I.ManagedIndex.load(P.Project.at(project))


def _sha(body: str) -> str:
    return skillfmt.body_sha256(body)


# ── the index ────────────────────────────────────────────────────────────────


class TestLookups:
    def test_by_install_name(self, index):
        skill = index.get("acme-refund-policy")
        assert skill is not None
        assert (skill.install_name, skill.registry_name, skill.skill_id, skill.version_id,
                skill.body_sha256) == ("acme-refund-policy", "acme/refund-policy", ACME_ID,
                                       ACME_VID, _sha(ACME_BODY))
        assert index.by_install_name("acme-refund-policy") is skill

    def test_by_the_name_claude_code_gives_a_plugin_skill(self, index):
        assert index.get("decimalai-skills:acme-refund-policy") is index.get("acme-refund-policy")
        assert index.by_plugin_name("decimalai-skills:code-review").skill_id == REVIEW_ID

    def test_the_marketplace_plugins_namespace_is_not_ours(self, index):
        # `decimalai:<cmd>` is the marketplace plugin (install, search, check), never a
        # skill this project installed.
        assert index.get("decimalai:acme-refund-policy") is None

    def test_the_prefix_is_read_from_plugin_json(self, project):
        path = P.Project.at(project).plugin_json_path
        data = json.loads(path.read_text(encoding="utf-8"))
        data["name"] = "team-skills"
        path.write_text(json.dumps(data), encoding="utf-8")
        index = I.ManagedIndex.load(P.Project.at(project))
        assert index.plugin_name == "team-skills"
        assert index.get("team-skills:acme-refund-policy").skill_id == ACME_ID
        assert index.get("decimalai-skills:acme-refund-policy") is None

    def test_without_plugin_json_there_is_no_prefixed_name(self, tmp_path):
        _install(tmp_path, plugin_json=False)
        index = I.ManagedIndex.load(P.Project.at(tmp_path))
        assert index.plugin_name is None
        assert index.get("decimalai-skills:acme-refund-policy") is None
        assert index.get("acme-refund-policy").skill_id == ACME_ID

    @pytest.mark.parametrize("spelling", [
        lambda h: h, lambda h: f"sha256:{h}", lambda h: h.upper(), lambda h: f" sha256:{h} ",
    ])
    def test_by_body_hash(self, index, spelling):
        assert index.by_hash(spelling(_sha(ACME_BODY))).skill_id == ACME_ID

    def test_a_hash_two_installed_skills_share_names_neither(self, tmp_path):
        _install(tmp_path, {
            "acme-refund-policy": ("acme/refund-policy", ACME_ID, ACME_VID, ACME_BODY),
            "acme-refund-policy-fork": ("bob/acme-refund-policy-fork", _id(9), _id(909),
                                        ACME_BODY),
        })
        index = I.ManagedIndex.load(P.Project.at(tmp_path))
        assert index.by_hash(_sha(ACME_BODY)) is None
        assert index.by_hash(_sha(PDF_BODY)) is None
        assert index.by_hash("not a hash") is None

    def test_by_path(self, project, index, monkeypatch):
        folder = project / ".agents" / "skills" / "acme-refund-policy"
        for path in (str(folder), str(folder) + "/", str(folder / "SKILL.md"),
                     ".agents/skills/acme-refund-policy",
                     ".agents/skills/acme-refund-policy/SKILL.md"):
            assert index.get(path) is not None, path
            assert index.by_path(path).skill_id == ACME_ID, path
        sub = project / "src"
        sub.mkdir()
        monkeypatch.chdir(sub)
        assert index.get("../.agents/skills/acme-refund-policy/SKILL.md").skill_id == ACME_ID

    def test_a_path_through_a_symlink_is_the_folder_it_reaches(self, project, index):
        # The opt-in `.claude/skills/decimalai-skills -> ../../.agents` link.
        link_dir = project / ".claude" / "skills"
        link_dir.mkdir(parents=True)
        (link_dir / "decimalai-skills").symlink_to(Path("..") / ".." / ".agents")
        path = link_dir / "decimalai-skills" / "skills" / "code-review" / "SKILL.md"
        assert index.get(str(path)).skill_id == REVIEW_ID

    def test_a_path_the_lock_does_not_manage_is_nothing(self, project, index):
        assert index.get(str(project / ".agents" / "skills" / "pdf-tools")) is None
        assert index.get(str(project / ".claude" / "skills" / "acme-refund-policy")) is None

    def test_by_skill_id(self, index):
        assert index.by_skill_id(PDF_ID).install_name == PDF_INSTALL
        assert index.by_skill_id(_id(77)) is None

    def test_the_registry_name_is_not_a_key(self, index):
        # A registry name in a trace is the router's report, and may be the org's own
        # namesake; it resolves on the platform's name path, as before.
        assert index.get("acme/refund-policy") is None

    def test_names_are_exact(self, index):
        assert index.get("Acme-Refund-Policy") is None
        assert index.get("") is None
        assert index.get(None) is None  # type: ignore[arg-type]

    def test_the_hashed_install_name_is_reported_as_itself(self, index):
        assert index.get("acme-refund-policy").wire_name == "acme/refund-policy"
        assert index.get("code-review").wire_name == "code-review"
        pdf = index.get(PDF_INSTALL)
        assert pdf.hashed is True and pdf.wire_name == PDF_INSTALL

    def test_a_hashed_folder_stays_hashed_after_an_upstream_rename(self, tmp_path):
        # A locked folder keeps its name when the registry renames the skill, so the
        # suffix, not the current name, says it was disambiguated.
        _install(tmp_path, {PDF_INSTALL: ("acme/pdf-kit", PDF_ID, PDF_VID, PDF_BODY),
                            "acme-refund-policy": ("acme/returns", ACME_ID, ACME_VID,
                                                   ACME_BODY)})
        index = I.ManagedIndex.load(P.Project.at(tmp_path))
        assert index.get(PDF_INSTALL).wire_name == PDF_INSTALL
        assert index.get("acme-refund-policy").wire_name == "acme/returns"

    def test_len_iter_and_truth(self, index):
        assert len(index) == 3 and bool(index)
        assert sorted(s.install_name for s in index) == sorted(SKILLS)
        assert not I.ManagedIndex()


class TestNoIndex:
    """A reader must never fail because of a lock: every refusal is an empty index."""

    def test_no_lock(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert not I.current_index(tmp_path)

    def test_a_version_1_lock(self, tmp_path):
        (tmp_path / ".decimal").mkdir()
        (tmp_path / ".decimal" / "skills.lock").write_text(
            json.dumps({"skills": {"acme-refund-policy": {"name": "acme/refund-policy"}}}))
        assert not I.current_index(tmp_path)

    def test_a_corrupt_lock(self, tmp_path):
        (tmp_path / ".decimal").mkdir()
        (tmp_path / ".decimal" / "skills.lock").write_text("{not json")
        assert not I.current_index(tmp_path)

    def test_entries_an_older_decimalai_merged_in_are_skipped(self, project):
        path = P.Project.at(project).lock_path
        data = json.loads(path.read_text())
        data["skills"]["old-skill"] = {"name": "old-skill", "hash": "sha256:" + "a" * 64}
        path.write_text(json.dumps(data))
        index = I.current_index(project)
        assert len(index) == 3 and index.get("old-skill") is None

    def test_a_hand_edit_that_breaks_an_entry_drops_only_that_entry(self, project):
        path = P.Project.at(project).lock_path
        data = json.loads(path.read_text())
        data["skills"]["code-review"]["version_id"] = ""
        path.write_text(json.dumps(data))
        index = I.current_index(project)
        assert index.get("code-review") is None
        assert index.get("acme-refund-policy") is not None  # a stale lock_hash is fine


class TestCache:
    def test_the_lock_is_read_once_until_it_changes(self, project, monkeypatch):
        calls = []
        real = L.locked_entries
        monkeypatch.setattr(L, "locked_entries", lambda p: calls.append(p) or real(p))
        first = I.current_index(project)
        assert I.current_index(project / ".agents") is first
        assert len(calls) == 1

    def test_the_cache_refreshes_when_the_lock_mtime_changes(self, project):
        before = I.current_index(project)
        assert before.get("extra-skill") is None
        skills = dict(SKILLS, **{"extra-skill": ("acme/extra-skill", _id(4), _id(404),
                                                 "# Extra\n\nOne more skill body.\n")})
        _install(project, skills)
        _bump_mtime(P.Project.at(project).lock_path)
        after = I.current_index(project)
        assert after is not before
        assert after.get("extra-skill").registry_name == "acme/extra-skill"

    def test_a_new_plugin_name_is_seen(self, project):
        I.current_index(project)
        path = P.Project.at(project).plugin_json_path
        path.write_text(json.dumps({"name": "team-skills"}), encoding="utf-8")
        _bump_mtime(path)
        assert I.current_index(project).get("team-skills:code-review") is not None


# ── the translation ──────────────────────────────────────────────────────────


def _payload(**fields) -> dict:
    base = {
        "id": "00000000-0000-0000-0000-00000000abcd",
        "agent_name": "support-bot",
        "active_skills": [],
        "skills_offered_in_prompt": [],
        "skills_loaded_by_agent": [],
        "skills_delivered": [],
        "skills_delivered_versions": [],
    }
    base.update(fields)
    return base


def _locked(name, sid, vid, body, **extra) -> dict:
    return {"name": name, "skill_id": sid, "version_id": vid, "body_sha256": _sha(body), **extra}


class TestNameLists:
    def test_all_four_lists_are_translated(self, index):
        out = I.translate_payload(_payload(
            active_skills=[{"name": "acme-refund-policy"}],
            skills_offered_in_prompt=["acme-refund-policy", "unmanaged-skill"],
            skills_loaded_by_agent=["decimalai-skills:acme-refund-policy"],
            skills_delivered=[".agents/skills/acme-refund-policy/SKILL.md"],
        ), index=index)
        assert out["active_skills"] == [_locked("acme/refund-policy", ACME_ID, ACME_VID,
                                                ACME_BODY)]
        assert out["skills_offered_in_prompt"] == ["acme/refund-policy", "unmanaged-skill"]
        assert out["skills_loaded_by_agent"] == ["acme/refund-policy"]
        assert out["skills_delivered"] == ["acme/refund-policy"]

    def test_the_string_lists_stay_lists_of_strings(self, index):
        out = I.translate_payload(_payload(
            skills_offered_in_prompt=["code-review"],
            skills_loaded_by_agent=["acme-refund-policy", "decimalai-skills:code-review"],
            skills_delivered=["acme-refund-policy"],
        ), index=index)
        for key in ("skills_offered_in_prompt", "skills_loaded_by_agent", "skills_delivered"):
            assert out[key] and all(isinstance(n, str) for n in out[key]), key
        assert out["skills_loaded_by_agent"] == ["acme/refund-policy", "code-review"]

    def test_two_spellings_of_one_skill_become_one_name(self, index):
        out = I.translate_payload(_payload(skills_loaded_by_agent=[
            "acme-refund-policy", "decimalai-skills:acme-refund-policy", "other", "other",
        ]), index=index)
        assert out["skills_loaded_by_agent"] == ["acme/refund-policy", "other"]

    def test_a_hashed_install_name_is_sent_as_itself_everywhere(self, index):
        # The base folder was taken (here by a hand-authored `pdf-tools`, maybe by the
        # org's own skill of that name). Sending the registry name would hand the
        # activation to the org's namesake; the skill id says which skill it is.
        out = I.translate_payload(_payload(
            active_skills=[{"name": f"decimalai-skills:{PDF_INSTALL}"}],
            skills_loaded_by_agent=[f"decimalai-skills:{PDF_INSTALL}"],
            skills_delivered=[PDF_INSTALL],
            skills_offered_in_prompt=[f".agents/skills/{PDF_INSTALL}", "pdf-tools"],
        ), index=index)
        assert out["active_skills"] == [_locked(PDF_INSTALL, PDF_ID, PDF_VID, PDF_BODY)]
        assert out["skills_loaded_by_agent"] == [PDF_INSTALL]
        assert out["skills_delivered"] == [PDF_INSTALL]
        assert out["skills_offered_in_prompt"] == [PDF_INSTALL, "pdf-tools"]

    def test_unknown_names_and_odd_items_pass_through(self, index):
        payload = _payload(
            active_skills=[{"name": "unmanaged"}, {"hash": "sha256:" + "a" * 64}, "unmanaged",
                           {"name": 3}, None],
            skills_offered_in_prompt=["unmanaged", 7, None],
            skills_loaded_by_agent=["decimalai:install"],
            skills_delivered=["acme/refund-policy"],
        )
        out = I.translate_payload(copy.deepcopy(payload), index=index)
        assert out == payload

    def test_the_other_fields_are_untouched(self, index):
        versions = [{"name": "acme-refund-policy", "hash": "sha256:" + "c" * 64}]
        payload = _payload(skills_delivered_versions=versions, routing_id="rt_abc",
                           llm_calls=[{"model_name": "m"}], agent_name="acme-refund-policy")
        out = I.translate_payload(payload, index=index)
        for key in ("skills_delivered_versions", "routing_id", "llm_calls", "agent_name", "id"):
            assert out[key] == payload[key], key

    def test_the_payload_given_is_not_changed(self, index):
        payload = _payload(active_skills=[{"name": "acme-refund-policy", "hash": None}],
                           skills_loaded_by_agent=["acme-refund-policy"])
        before = copy.deepcopy(payload)
        out = I.translate_payload(payload, index=index)
        assert payload == before
        assert out is not payload

    def test_an_empty_index_returns_the_payload_itself(self):
        payload = _payload(skills_loaded_by_agent=["acme-refund-policy"])
        assert I.translate_payload(payload, index=I.ManagedIndex()) is payload

    def test_not_a_dict(self, index):
        assert I.translate_payload(["acme-refund-policy"], index=index) == ["acme-refund-policy"]

    def test_the_default_index_is_the_working_directorys_project(self, project, monkeypatch):
        monkeypatch.chdir(project / ".agents")
        out = I.translate_payload(_payload(skills_loaded_by_agent=["acme-refund-policy"]))
        assert out["skills_loaded_by_agent"] == ["acme/refund-policy"]


class TestActiveSkills:
    def test_a_string_entry_becomes_a_dict_with_ids(self, index):
        out = I.translate_payload(_payload(active_skills=["acme-refund-policy", "unmanaged"]),
                                  index=index)
        assert out["active_skills"] == [
            _locked("acme/refund-policy", ACME_ID, ACME_VID, ACME_BODY), "unmanaged"]

    def test_a_dict_entry_is_enriched_and_keeps_its_own_keys(self, index):
        reported = "sha256:" + _sha(REVIEW_BODY)
        out = I.translate_payload(_payload(active_skills=[
            {"name": "decimalai-skills:code-review", "hash": reported, "source": "disk"},
        ]), index=index)
        assert out["active_skills"] == [_locked("code-review", REVIEW_ID, REVIEW_VID,
                                                REVIEW_BODY, hash=reported, source="disk")]

    @pytest.mark.parametrize("key", ["body_sha256", "hash", "content_hash"])
    def test_the_hash_is_read_where_the_platform_reads_it(self, index, key):
        out = I.translate_payload(_payload(active_skills=[
            {"name": "acme-refund-policy", key: _sha(ACME_BODY).upper()},
        ]), index=index)
        assert out["active_skills"][0]["version_id"] == ACME_VID


class TestAmbiguityGuard:
    """An entry that already says which skill it is, or carries another body's hash, is
    left exactly as it was: that is a router delivery of another skill sharing the name."""

    @pytest.mark.parametrize("entry", [
        {"name": "acme-refund-policy", "skill_id": _id(50)},
        {"name": "acme-refund-policy", "skill_id": ACME_ID},
        {"name": "acme-refund-policy", "version_id": _id(5050)},
        {"name": "acme-refund-policy", "hash": "sha256:" + "e" * 64},
        {"name": "acme-refund-policy", "content_hash": "e" * 64},
        {"name": "acme-refund-policy", "hash": "not-a-sha256"},
        # The platform reads body_sha256 before hash: the matching hash is not looked at.
        {"name": "acme-refund-policy", "body_sha256": "e" * 64,
         "hash": "sha256:" + _sha(ACME_BODY)},
    ])
    def test_left_alone(self, index, entry):
        out = I.translate_payload(_payload(active_skills=[dict(entry)]), index=index)
        assert out["active_skills"] == [entry]

    # An adapter writes an active_skills entry only for a name it also lists as loaded, and
    # the platform keys every report by its name. So a name the guard leaves alone must stay
    # itself in the name lists too: translated there, it would credit the locked skill with
    # a run of another body, and the platform would record two activations.

    @pytest.mark.parametrize("guarded", [
        # the org's own skill, delivered by the router under the same name
        {"name": "acme-refund-policy", "hash": "e" * 64},
        # the caller said which skill it is
        {"name": "acme-refund-policy", "skill_id": _id(50)},
        {"name": "acme-refund-policy", "skill_id": ACME_ID, "version_id": ACME_VID},
    ])
    def test_a_name_left_alone_stays_itself_in_every_list(self, index, guarded):
        payload = _payload(
            active_skills=[dict(guarded)],
            skills_loaded_by_agent=["acme-refund-policy"],
            skills_delivered=["acme-refund-policy"],
            skills_offered_in_prompt=["acme-refund-policy", "code-review"],
        )
        out = I.translate_payload(copy.deepcopy(payload), index=index)
        assert out["active_skills"] == [guarded]
        assert out["skills_loaded_by_agent"] == ["acme-refund-policy"]
        assert out["skills_delivered"] == ["acme-refund-policy"]
        assert out["skills_offered_in_prompt"] == ["acme-refund-policy", "code-review"]
        active = {e["name"] for e in out["active_skills"]}
        assert active <= set(out["skills_loaded_by_agent"])

    def test_after_an_upstream_rename_a_name_left_alone_stays_itself(self, tmp_path):
        # The lock keeps the folder name; the registry name is now another one entirely.
        _install(tmp_path, {"acme-refund-policy": ("acme/returns", ACME_ID, ACME_VID,
                                                   ACME_BODY)})
        index = I.ManagedIndex.load(P.Project.at(tmp_path))
        out = I.translate_payload(_payload(
            active_skills=[{"name": "acme-refund-policy", "content_hash": "e" * 64}],
            skills_loaded_by_agent=["acme-refund-policy"],
        ), index=index)
        assert out["skills_loaded_by_agent"] == ["acme-refund-policy"]

    def test_every_entry_of_a_name_left_alone_stays_itself(self, index):
        # Two reports of one name with different bodies: which one the folder is cannot
        # be told, so neither is credited to it.
        entries = [{"name": "acme-refund-policy"}, {"name": "acme-refund-policy", "hash": "e" * 64},
                   "acme-refund-policy"]
        out = I.translate_payload(_payload(active_skills=copy.deepcopy(entries),
                                           skills_loaded_by_agent=["acme-refund-policy"]),
                                  index=index)
        assert out["active_skills"] == entries
        assert out["skills_loaded_by_agent"] == ["acme-refund-policy"]

    def test_only_that_exact_name_is_held_back(self, index):
        # Claude Code's plugin name can only be this project's folder: still translated.
        out = I.translate_payload(_payload(
            active_skills=[{"name": "acme-refund-policy", "hash": "e" * 64},
                           {"name": "code-review"}],
            skills_loaded_by_agent=["acme-refund-policy", "decimalai-skills:acme-refund-policy",
                                    "code-review"],
        ), index=index)
        assert out["skills_loaded_by_agent"] == ["acme-refund-policy", "acme/refund-policy",
                                                 "code-review"]
        assert out["active_skills"][1]["version_id"] == REVIEW_VID


class TestLocallyEditedFolder:
    """An edited managed folder is credited to its skill, never to the stamped version."""

    def _edit(self, project: Path, name: str, new_body: str) -> None:
        path = project / ".agents" / "skills" / name / "SKILL.md"
        text = path.read_text(encoding="utf-8")
        head, _sep, _body = text.partition("\n---\n")
        path.write_text(f"{head}\n---\n{new_body}", encoding="utf-8")

    def test_the_hash_of_the_body_read_is_sent_and_the_entry_is_marked(self, project, index):
        edited = ACME_BODY + "\nAlso refund the shipping.\n"
        self._edit(project, "acme-refund-policy", edited)
        out = I.translate_payload(_payload(active_skills=["acme-refund-policy"]), index=index)
        assert out["active_skills"] == [{
            "name": "acme/refund-policy", "skill_id": ACME_ID, "body_sha256": _sha(edited),
            "modified": True,
        }]

    def test_an_entry_that_reports_the_edited_body_is_enriched_as_edited(self, project, index):
        edited = ACME_BODY + "\nAlso refund the shipping.\n"
        self._edit(project, "acme-refund-policy", edited)
        out = I.translate_payload(_payload(active_skills=[
            {"name": "acme-refund-policy", "hash": f"sha256:{_sha(edited)}"},
        ]), index=index)
        entry = out["active_skills"][0]
        assert entry["modified"] is True and "version_id" not in entry
        assert entry["body_sha256"] == _sha(edited) and entry["skill_id"] == ACME_ID

    def test_an_entry_that_reports_the_stamped_body_keeps_its_version(self, project, index):
        # It read the locked bytes somewhere else, and its hash proves which.
        self._edit(project, "acme-refund-policy", "# Rewritten\n\nSomething else entirely.\n")
        out = I.translate_payload(_payload(active_skills=[
            {"name": "acme-refund-policy", "hash": f"sha256:{_sha(ACME_BODY)}"},
        ]), index=index)
        assert out["active_skills"][0]["version_id"] == ACME_VID
        assert "modified" not in out["active_skills"][0]

    def test_an_edit_after_the_index_was_cached_is_seen(self, project):
        index = I.current_index(project)
        assert "modified" not in I.translate_payload(
            _payload(active_skills=["code-review"]), index=index)["active_skills"][0]
        path = project / ".agents" / "skills" / "code-review" / "SKILL.md"
        self._edit(project, "code-review", REVIEW_BODY + "\nAnd run the tests.\n")
        _bump_mtime(path)
        assert I.current_index(project) is index  # the lock did not change
        entry = I.translate_payload(_payload(active_skills=["code-review"]),
                                    index=index)["active_skills"][0]
        assert entry["modified"] is True and "version_id" not in entry

    def test_a_crlf_checkout_is_not_an_edit(self, project, index):
        path = project / ".agents" / "skills" / "acme-refund-policy" / "SKILL.md"
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        out = I.translate_payload(_payload(active_skills=["acme-refund-policy"]), index=index)
        assert out["active_skills"][0]["version_id"] == ACME_VID

    def test_a_missing_folder_translates_nothing(self, project, index):
        # No loader read it from here, so the name is somebody else's report.
        (project / ".agents" / "skills" / "code-review" / "SKILL.md").unlink()
        payload = _payload(active_skills=["code-review"], skills_loaded_by_agent=["code-review"])
        assert I.translate_payload(copy.deepcopy(payload), index=index) == payload

    def test_a_skill_md_that_is_not_a_regular_file_translates_nothing(self, project, index):
        path = project / ".agents" / "skills" / "code-review" / "SKILL.md"
        path.unlink()
        path.mkdir()
        payload = _payload(active_skills=["code-review"])
        assert I.translate_payload(copy.deepcopy(payload), index=index) == payload
