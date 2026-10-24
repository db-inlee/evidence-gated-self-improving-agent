"""분석 실행 스크립트(scripts/run_analysis.sh)의 고정 커밋 확인. 저장소 기록을 읽기만 한다.

실제 저장소의 결과 폴더(results/ticket-005/)나 이후의 분석 모듈 변경에 따라 결과가 바뀌지 않도록,
확인은 언제나 임시 출력 폴더로 하고, 고정 커밋의 기대값은 테스트 안에서 git diff로 정한다.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_analysis.sh"

pytestmark = pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="needs the git history")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def check(commit: str, out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--check-only", commit, "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def analysis_modules_clean() -> bool:
    return (
        git(
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src/evidencegate",
            "src/opsworld",
        )
        == ""
    )


def test_head_passes_when_the_analysis_modules_are_committed(tmp_path: Path) -> None:
    if not analysis_modules_clean():
        pytest.skip("analysis modules have uncommitted changes")
    result = check("HEAD", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "unchanged" in result.stdout and "plan sha256 99d4dd03" in result.stdout


def test_the_freeze_commit_passes_exactly_when_nothing_changed_since(tmp_path: Path) -> None:
    """기대값은 지금의 git diff로 정한다: 바뀐 게 있으면 실패, 없으면 통과."""
    if not analysis_modules_clean():
        pytest.skip("analysis modules have uncommitted changes")
    changed = git(
        "diff", "--name-only", "bdbecf0", "HEAD", "--", "src/evidencegate", "src/opsworld"
    )
    result = check("bdbecf0", tmp_path)
    if changed:
        assert result.returncode == 1 and "changed since" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


def test_a_commit_with_later_analysis_changes_fails(tmp_path: Path) -> None:
    """484a685(순서 7) 뒤에 bench_runner.py가 생겼으므로 실패한다."""
    result = check("484a685", tmp_path)
    assert result.returncode == 1
    assert "changed since" in result.stderr and "src/evidencegate/bench_runner.py" in result.stderr


@pytest.mark.parametrize("ref", ["0000000", "HEAD^{tree}", "no-such-ref"])
def test_something_that_is_not_a_commit_fails(ref: str, tmp_path: Path) -> None:
    result = check(ref, tmp_path)
    assert result.returncode == 1 and "not a commit" in result.stderr


def test_an_existing_result_is_not_overwritten(tmp_path: Path) -> None:
    if not analysis_modules_clean():
        pytest.skip("analysis modules have uncommitted changes")
    (tmp_path / "analysis.json").write_text("{}")
    result = check("HEAD", tmp_path)
    assert result.returncode == 1 and "runs once" in result.stderr


def test_usage_without_a_commit() -> None:
    result = subprocess.run(["bash", str(SCRIPT)], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 64


# 아래는 저장소를 임시 폴더에 복제해(원본은 읽기만 함) 그 안에서 상황을 만든다.
# 커밋은 만들지 않는다.


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    target = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(target)], check=True
    )
    (target / "scripts" / "run_analysis.sh").write_text(SCRIPT.read_text())
    return target


def check_in(repo: Path, commit: str) -> subprocess.CompletedProcess[str]:
    out_dir = repo.parent / "out"  # 복제본에 결과 폴더가 커밋돼 있어도 영향 없게
    return subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "run_analysis.sh"),
            "--check-only",
            commit,
            "--out-dir",
            str(out_dir),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_a_commit_that_is_not_an_ancestor_of_head_fails(clone: Path) -> None:
    subprocess.run(["git", "checkout", "--quiet", "484a685"], cwd=clone, check=True)
    result = check_in(clone, "bdbecf0")
    assert result.returncode == 1 and "not an ancestor" in result.stderr


def test_uncommitted_files_in_the_analysis_modules_fail(clone: Path) -> None:
    assert check_in(clone, "HEAD").returncode == 0
    (clone / "src" / "opsworld" / "extra.py").write_text("x = 1\n")
    result = check_in(clone, "HEAD")
    assert result.returncode == 1 and "uncommitted" in result.stderr and "extra.py" in result.stderr


def test_a_plan_that_does_not_match_its_hash_fails(clone: Path) -> None:
    plan = clone / "plans" / "ticket-005-plan.json"
    plan.write_bytes(plan.read_bytes() + b" ")
    result = check_in(clone, "HEAD")
    assert result.returncode == 1 and "does not match the record" in result.stderr
