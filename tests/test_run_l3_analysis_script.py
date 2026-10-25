"""L3 분석 실행 스크립트(scripts/run_l3_analysis.sh)의 고정 커밋 확인. 저장소 기록을 읽기만 한다.

결과 폴더(results/ticket-006/)나 이후의 변경에 따라 결과가 바뀌지 않도록 확인은 언제나 임시 출력
폴더로 한다. 복제본은 원본을 읽기만 하고 커밋하지 않는다. 아직 커밋되지 않았을 수 있는
파일(스크립트, L3 계획)은 복제본에 지금 내용으로 둔다.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_l3_analysis.sh"
PLAN = REPO_ROOT / "plans" / "ticket-006-l3-plan.json"
COPIED = (
    "scripts/run_l3_analysis.sh",
    "plans/ticket-006-l3-plan.json",
    "plans/ticket-006-l3-plan.json.sha256",
)
L3_CHECK_SETUP = "d94e935"  # L3-check 준비 커밋. 이 뒤로 분석 모듈이 바뀌었다
L3_GENERATOR = "c75fc42"  # L3 생성기 커밋. L3_CHECK_SETUP의 조상

pytestmark = pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="needs the git history")


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def check(commit: str, out_dir: Path, repo: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "run_l3_analysis.sh"),
            "--check-only",
            commit,
            "--out-dir",
            str(out_dir),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def modules_clean(repo: Path = REPO_ROOT) -> bool:
    status = git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src/evidencegate",
        "src/opsworld",
        cwd=repo,
    )
    return status == ""


def test_head_passes_when_the_analysis_modules_are_committed(tmp_path: Path) -> None:
    if not modules_clean():
        pytest.skip("analysis modules have uncommitted changes")
    result = check("HEAD", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (
        "L3 plan sha256 " in result.stdout and "Ticket 005 result sha256 51d51c9b" in result.stdout
    )


def test_uncommitted_analysis_modules_fail_in_the_working_tree(tmp_path: Path) -> None:
    if modules_clean():
        pytest.skip("analysis modules are committed")
    result = check("HEAD", tmp_path)
    assert result.returncode == 1 and "uncommitted" in result.stderr


@pytest.mark.parametrize("ref", ["0000000", "HEAD^{tree}", "no-such-ref"])
def test_something_that_is_not_a_commit_fails(ref: str, tmp_path: Path) -> None:
    result = check(ref, tmp_path)
    assert result.returncode == 1 and "not a commit" in result.stderr


def test_usage_without_a_commit() -> None:
    result = subprocess.run(["bash", str(SCRIPT)], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 64


# 아래는 저장소를 임시 폴더에 복제해 그 안에서 상황을 만든다.


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    target = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(target)], check=True
    )
    place(target)
    return target


def place(repo: Path) -> None:
    for rel in COPIED:
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO_ROOT / rel).read_bytes())


def test_clean_clone_passes(clone: Path) -> None:
    result = check("HEAD", clone.parent / "out", clone)
    assert result.returncode == 0, result.stderr


def test_an_existing_result_is_not_overwritten(clone: Path) -> None:
    out = clone.parent / "out"
    out.mkdir()
    (out / "analysis.json").write_text("{}")
    result = check("HEAD", out, clone)
    assert result.returncode == 1 and "runs once" in result.stderr


def test_a_commit_with_later_analysis_changes_fails(clone: Path) -> None:
    """L3 생성기 커밋 뒤로 l3_bench.py 등이 생겼으므로 실패한다."""
    result = check(L3_GENERATOR, clone.parent / "out", clone)
    assert result.returncode == 1 and "changed since" in result.stderr
    assert "src/evidencegate/l3_bench.py" in result.stderr


def test_a_commit_that_is_not_an_ancestor_of_head_fails(clone: Path) -> None:
    subprocess.run(["git", "checkout", "--quiet", L3_GENERATOR], cwd=clone, check=True)
    place(clone)  # checkout이 지운 파일을 다시 둔다
    result = check(L3_CHECK_SETUP, clone.parent / "out", clone)
    assert result.returncode == 1 and "not an ancestor" in result.stderr


def test_untracked_files_in_the_analysis_modules_fail(clone: Path) -> None:
    (clone / "src" / "evidencegate" / "extra.py").write_text("x = 1\n")
    result = check("HEAD", clone.parent / "out", clone)
    assert result.returncode == 1 and "uncommitted" in result.stderr and "extra.py" in result.stderr


def test_an_l3_plan_that_does_not_match_its_hash_fails(clone: Path) -> None:
    plan = clone / "plans" / "ticket-006-l3-plan.json"
    plan.write_bytes(plan.read_bytes() + b" ")
    result = check("HEAD", clone.parent / "out", clone)
    assert result.returncode == 1 and "does not match the record" in result.stderr


def test_a_missing_l3_plan_fails(clone: Path) -> None:
    (clone / "plans" / "ticket-006-l3-plan.json").unlink()
    result = check("HEAD", clone.parent / "out", clone)
    assert result.returncode == 1 and "missing" in result.stderr


def test_a_changed_ticket_005_result_fails(clone: Path) -> None:
    result_file = clone / "results" / "ticket-005" / "analysis.json"
    result_file.write_bytes(result_file.read_bytes() + b" ")
    result = check("HEAD", clone.parent / "out", clone)
    assert result.returncode == 1 and "Ticket 005 result" in result.stderr
