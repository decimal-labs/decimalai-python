"""The platform's render vectors, vendored, and held to three things.

``vectors/platform_render_v1.json`` is a copy of the platform's own render vectors, equal
to it in every field but its ``description`` (which here names no platform path): database rows (a skill, its versions and attachments) and a caller in; the
fields and the attachment set the platform gathers from them, and the exact bundle it
serves, out. It is generated on the platform side; this repository only copies it.

* The copy equals the platform's file whenever the platform checkout is beside this one
  (a private sibling most clones lack: then the test says so and skips).
* This environment's ``skillfmt``, the renderer the probe serves the graded project with,
  reproduces every bundle from its fields: so the folders the probe renders from the
  oracle's rows are the ones the platform would serve for those fields.
* The probe serves each vector verbatim, and the real CLI installs it (``skills add``,
  then ``install --frozen`` from the manifest and the lock alone): every folder holds
  exactly the platform's bytes and modes, the lock records the platform's rendering, and
  ``skills verify`` passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from skillevaluation import skillfmt

from decimalai import skillpkg

from . import producer

REPO = Path(__file__).resolve().parents[2]
PLATFORM_VECTORS = (REPO.parent / "platform" / "backend" / "tests" / "fixtures"
                    / "skill_render_vectors.json")

VECTORS = producer.load_vectors()
CASES = {case["id"]: case for case in VECTORS["cases"]}


def test_the_vendored_vectors_equal_the_platforms():
    if not PLATFORM_VECTORS.is_file():
        pytest.skip(f"{PLATFORM_VECTORS} is not on disk: the platform repo is a private "
                    "sibling checkout; the copy is compared wherever both are present")
    vendored = json.loads(producer.VECTORS.read_text(encoding="utf-8"))
    platform = json.loads(PLATFORM_VECTORS.read_text(encoding="utf-8"))
    vendored.pop("description", None)
    platform.pop("description", None)
    assert vendored == platform, (
        "the vendored render vectors differ from the platform's: copy "
        f"{PLATFORM_VECTORS} to {producer.VECTORS} (keeping this copy's description) "
        "and run this suite again")


def test_the_vectors_are_the_version_this_suite_reads():
    assert (VECTORS["vectors_version"], VECTORS["format"]) == (1, 1)
    assert len(CASES) == len(VECTORS["cases"]) >= 7, "case ids repeat, or cases went missing"
    assert {c["variant"] for c in CASES.values()} == {"authored", "foreign"}
    assert None in {c["caller_org"] for c in CASES.values()}, "no anonymous caller left"


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_skillfmt_here_renders_each_vector_as_the_platform_did(case_id):
    case = CASES[case_id]
    fields = skillfmt.RenderFields.from_dict(case["fields"])
    bundle = skillfmt.render_bundle(fields, case["attachments"], install_name=case["install_name"],
                                    variant=case["variant"], format=VECTORS["format"])
    assert bundle.to_dict() == case["bundle"], f"{case_id}: skillfmt renders other bytes"
    assert skillfmt.render_hash(fields) == case["render_hash"]
    assert skillfmt.install_name_for(fields.registry_name, fields.skill_id) == case["install_name"]


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    return producer.produce_vectors(tmp_path_factory.mktemp("platform-vectors"))


def test_every_vector_is_installed_by_its_callers_project(installed):
    want = {caller: sorted(c["install_name"] for c in cases)
            for caller, cases in producer.vector_groups(VECTORS).items()}
    got = {caller: sorted(json.loads((made.project / ".decimal" / "skills.lock").read_text())
                          ["skills"]) for caller, made in installed.items()}
    assert got == want
    for made in installed.values():
        assert made.after_frozen == made.after_add, "install --frozen wrote other bytes than add"


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_the_cli_writes_the_platforms_bytes(installed, case_id):
    case = CASES[case_id]
    made = installed[case["caller_org"] or "anonymous"]
    bundle = case["bundle"]
    folder = f".agents/skills/{case['install_name']}/"
    written = {k[len(folder):]: v for k, v in producer.tree(made.project).items()
               if k.startswith(folder)}
    served = {f["path"]: ("file", int(f["mode"], 8), f["content"].encode("utf-8"))
              for f in bundle["files"]}
    assert written == served, f"{case_id}: the installed folder is not the platform's bundle"

    entry = json.loads((made.project / ".decimal" / "skills.lock").read_text())[
        "skills"][case["install_name"]]
    assert entry["render"] == bundle["render"]
    for key in ("skill_id", "version_id", "body_sha256", "description_sha256", "invocation",
                "unavailable"):
        assert entry[key] == bundle[key], f"{case_id}: the lock's {key} is not the platform's"
    assert entry["files"] == {f["path"]: f["sha256"] for f in bundle["files"]}
    assert entry["executable"] == sorted(f["path"] for f in bundle["files"] if f["mode"] == "0755")
    assert entry["source"]["org"] == case["caller_org"], "installed through the wrong route"


def test_verify_passes_on_every_project_of_platform_bytes(installed):
    for caller, made in sorted(installed.items()):
        result = skillpkg.verify(directory=made.project)
        assert int(result.exit_code) == 0 and not result.problems, (
            caller, [str(f) for f in result.problems])
