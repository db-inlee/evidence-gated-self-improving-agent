"""표 T4~T7, 허용 목록, 문서 쌍 목록 (Ticket 011 §1).

T4~T6은 009b 사전 등록 결과(kind result·derived)만, T7은 010 탐침(kind exploratory)만 쓴다. 표 칸의
숫자는 자리표시자로만 들어간다. writeup_render와 writeup_artifacts의 이름 허용 목록은 같다.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from evidencegate import writeup_artifacts as wa
from evidencegate import writeup_render as wr

REPO_ROOT = Path(__file__).resolve().parents[1]
VALUES = REPO_ROOT / "writeup" / "generated" / "values.json"
NEW_TABLES = ("t4", "t5", "t6", "t7")


@pytest.fixture(scope="module")
def values() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = json.loads(VALUES.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def tables(values: dict[str, Any]) -> dict[str, Any]:
    return wa.build_tables(values)


def texts(table: dict[str, Any]) -> list[str]:
    return [
        table["title"],
        table["note"],
        *table["columns"],
        *(c for r in table["rows"] for c in r),
    ]


def keys(table: dict[str, Any]) -> list[str]:
    return [k for t in texts(table) for k in wa.PLACEHOLDER.findall(t)]


# 표 ---------------------------------------------------------------------------------------------


def test_new_tables_exist_after_the_v1_tables(tables: dict[str, Any]) -> None:
    assert list(tables)[:3] == ["t1", "t2", "t3"]
    assert set(NEW_TABLES) <= set(tables)
    for table_id in NEW_TABLES:
        assert tables[table_id]["id"] == table_id


@pytest.mark.parametrize("table_id", NEW_TABLES)
def test_new_tables_resolve_and_hold_no_bare_numbers(
    tables: dict[str, Any], values: dict[str, Any], table_id: str
) -> None:
    found = keys(tables[table_id])
    assert found and [k for k in found if k not in values] == []
    for text in texts(tables[table_id]):
        bare = wa.ALLOWED_NAME.sub("", wa.PLACEHOLDER.sub("", text))
        assert not re.search(r"\d", bare), (table_id, text)


def test_t4_to_t6_use_pre_registered_values_and_t7_only_exploratory(
    tables: dict[str, Any], values: dict[str, Any]
) -> None:
    for table_id in ("t4", "t5", "t6"):
        kinds = {values[k]["kind"] for k in keys(tables[table_id])}
        assert kinds <= {"result", "derived", "design"}, (table_id, kinds)
        assert all(
            k.startswith(("t009b.", "derived.t009b.", "design.")) for k in keys(tables[table_id])
        )
    assert {values[k]["kind"] for k in keys(tables["t7"])} == {"exploratory"}


def test_t4_holds_the_rq9_counts_tests_and_verdicts(tables: dict[str, Any]) -> None:
    found = set(keys(tables["t4"]))
    rq = "t009b.rq"
    needed = {
        *(f"{rq}.rq9_1.{f}" for f in ("a", "f", "k", "n", "p", "verdict")),
        *(f"{rq}.rq9_2.{f}" for f in ("u", "useful_c4", "verdict")),
        *(f"{rq}.rq9_3.{f}" for f in ("k", "n", "interval", "verdict")),
        *(f"{rq}.rq9_5.{f}" for f in ("s1", "s2", "d", "m", "p", "verdict", "verdict_count")),
        *(f"{rq}.rq9_1_local_harm.{f}" for f in ("p", "verdict")),
        f"{rq}.rq9_2_all_useful.verdict",
        f"{rq}.rq9_3_c2.verdict",
    }
    assert needed <= found, needed - found
    rows = {tuple(r) for r in tables["t4"]["rows"]}
    descriptive = [r for r in rows if "(descriptive)" in r[0]]
    joined = " ".join(c for r in descriptive for c in r)
    for name in ("rq9_1_local_harm", "rq9_2_all_useful", "rq9_3_c2", "rq9_5.verdict_count"):
        assert name in joined, name
    for r in rows:  # 판정 행은 기술용 키를 쓰지 않는다
        if "(descriptive)" not in r[0]:
            assert not re.search(r"local_harm|all_useful|rq9_3_c2|verdict_count", " ".join(r)), r


def test_t5_is_the_cd_metric_table_for_the_five_gates(tables: dict[str, Any]) -> None:
    assert [r[0] for r in tables["t5"]["rows"]] == ["C2", "C4", "C5", "C5\u2032", "C4*"]
    found = keys(tables["t5"])
    assert all(".cd." in k for k in found if k.startswith("t009b.metrics."))
    for gate in ("C2", "C4", "C5", "C5p", "C4star"):
        assert f"t009b.metrics.{gate}.cd.severe.successes" in found, gate
        assert f"t009b.metrics.{gate}.cd.accepted.count" in found, gate


def test_t6_is_the_cluster_sensitivity_table(tables: dict[str, Any]) -> None:
    found = set(keys(tables["t6"]))
    assert "derived.t009b.cluster_groups" in found
    assert {
        f"t009b.cluster_sensitivity.rq9_1.{f}" for f in ("a", "f", "k", "n", "p", "verdict")
    } <= found
    assert "t009b.cluster_sensitivity.rq9_3.verdict" in found
    assert "t009b.cluster_sensitivity.rq9_5.verdict" in found
    assert "t009b.cluster_sensitivity.differs_from_main" in found


def test_t7_has_one_row_per_probe_variant_and_the_s0_row(
    tables: dict[str, Any], values: dict[str, Any]
) -> None:
    rows = tables["t7"]["rows"]
    names = [wa.fill(r[0], values) for r in rows]
    assert names == ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "S0"]
    found = set(keys(tables["t7"]))
    for i in range(7):
        for f in ("different_disposition", "different_tool_set", "traffic_episodes",
                  "mean_tool_calls"):  # fmt: skip
            assert f"t010.variants.{i}.{f}" in found, (i, f)
    # 판정 집계에 들지 않는 변형(P0, judged = false)은 Moved 칸에 "n/a (wiring check)"
    for i in range(7):
        judged = values[f"t010.variants.{i}.judged"]["exact"] == "true"
        moved_cell = rows[i][3]
        if judged:
            assert moved_cell == "{{" + f"t010.variants.{i}.moved" + "}}", i
        else:
            assert moved_cell == "n/a (wiring check)", i
    assert values["t010.variants.0.judged"]["exact"] == "false"
    assert "t010.decision.verdict" in found


def test_t7_only_inside_an_exploratory_section(values: dict[str, Any]) -> None:
    rules = {v.rule for v in wr.check_source("## Results\n\n{{table:t7}}\n", "md", values)}
    assert rules == {"placement"}
    assert wr.check_source("## Harness probe (Exploratory)\n\n{{table:t7}}\n", "md", values) == []
    assert wr.check_source("## Results\n\n{{table:t4}}\n\n{{table:t5}}\n\n{{table:t6}}\n", "md",
                           values) == []  # fmt: skip


# 허용 목록 --------------------------------------------------------------------------------------


def test_render_and_artifacts_share_one_name_list() -> None:
    assert wr.NAME_PATTERNS == wa.NAME_PATTERNS
    patterns = [p.pattern for p in wr.ALLOWED_PATTERNS]
    assert patterns[: len(wa.NAME_PATTERNS)] == list(wa.NAME_PATTERNS)
    assert wa.ALLOWED_NAME.pattern == "|".join(f"(?:{p})" for p in wa.NAME_PATTERNS)


NEW_NAMES = (
    "Gates C5, C5\u2032 and C4*, questions RQ9-1 to RQ9-5.",
    "Tables T4 to T7 and probe variants P0 to P6.",
    "Ticket 009, Ticket 009b and Ticket 010; ticket-009b and ticket-010 notes.",
)
FORBIDDEN = ("Ticket 011", "Ticket 012", "h(70)", "δ(420)", "a 2\u00d72 table")


@pytest.mark.parametrize("text", NEW_NAMES)
def test_new_names_are_allowed(values: dict[str, Any], text: str) -> None:
    assert wr.check_source(f"## Design\n\n{text}\n", "md", values) == []
    assert not re.search(r"\d", wa.ALLOWED_NAME.sub("", text)), text


@pytest.mark.parametrize("text", FORBIDDEN)
def test_names_with_numbers_outside_the_list_are_refused(values: dict[str, Any], text: str) -> None:
    found = wr.check_source(f"## Design\n\nSee {text}.\n", "md", values)
    assert {v.rule for v in found} == {"number"}, text
    assert re.search(r"\d", wa.ALLOWED_NAME.sub("", text)), text


# 문서 쌍 목록 -----------------------------------------------------------------------------------


def test_documents_are_the_readmes_and_the_two_study_pages() -> None:
    expected = (
        wr.Document(Path("writeup/README.template"), Path("README.md"), "en"),
        wr.Document(
            Path("writeup/study-1-mean-gates.template"), Path("docs/study-1-mean-gates.md"), "en"
        ),
        wr.Document(
            Path("writeup/study-2-cell-gate.template"), Path("docs/study-2-cell-gate.md"), "en"
        ),
        wr.Document(Path("writeup/README.ko.template"), Path("README.ko.md"), "ko"),
    )
    assert expected == wr.DOCUMENTS


def write_root(tmp_path: Path, documents: dict[str, str]) -> Path:
    root = tmp_path / "root"
    (root / "writeup" / "generated").mkdir(parents=True)
    for name in ("values.json", "values.json.sha256", "tables.json"):
        (root / "writeup" / "generated" / name).write_bytes(
            (REPO_ROOT / "writeup" / "generated" / name).read_bytes()
        )
    for template, text in documents.items():
        (root / template).parent.mkdir(parents=True, exist_ok=True)
        (root / template).write_text(text, encoding="utf-8")
    return root


CLEAN = "# Title\n\n## Results\n\nThe verdict was {{t009b.rq.rq9_1.verdict}}.\n"


def test_render_and_check_walk_every_registered_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = (
        wr.Document(Path("writeup/a.template"), Path("A.md"), "en"),
        wr.Document(Path("writeup/b.template"), Path("docs/B.md"), "en"),
    )
    monkeypatch.setattr(wr, "DOCUMENTS", docs)
    root = write_root(tmp_path, {"writeup/a.template": CLEAN, "writeup/b.template": CLEAN})
    assert wr.main(["render", "--root", str(root)]) == 0
    assert (root / "A.md").read_text() == (root / "docs/B.md").read_text()
    assert "The verdict was supported." in (root / "A.md").read_text()
    assert wr.main(["check", "--root", str(root)]) == 0
    (root / "docs/B.md").write_text("edited\n")
    assert wr.main(["check", "--root", str(root)]) == 1
    # 앞 문서의 실패도 뒤 문서가 통과해서 지워지지 않는다
    assert wr.main(["render", "--root", str(root)]) == 0
    (root / "A.md").write_text("edited\n")
    assert wr.main(["check", "--root", str(root)]) == 1
