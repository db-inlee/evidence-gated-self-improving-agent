"""사전 등록 대응표(Ticket 005 §1): 7.8절 항목과 §8 해석 33개 모두에 구현 함수와 테스트가 있다."""

import ast
import importlib
from pathlib import Path

import pytest

from evidencegate.prereg_map import ITEMS, PREREG_78_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_7_8_item_and_section_8_interpretation_is_mapped() -> None:
    ids = [item.id for item in ITEMS]
    assert len(ids) == len(set(ids))
    assert [i for i in ids if i.startswith("7.8")] == list(PREREG_78_IDS)
    assert [i for i in ids if i.startswith("§8")] == [f"§8-{n}" for n in range(1, 34)]


def test_every_item_has_a_function_and_a_test() -> None:
    for item in ITEMS:
        assert item.functions, item.id
        assert item.tests, item.id
        assert item.text, item.id


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.id)
def test_mapped_functions_exist(item: object) -> None:
    for ref in item.functions:  # type: ignore[attr-defined]
        module_name, _, qualname = ref.partition(":")
        obj: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        assert callable(obj) or isinstance(obj, type), ref


def defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.id)
def test_mapped_tests_exist(item: object) -> None:
    for ref in item.tests:  # type: ignore[attr-defined]
        file, _, name = ref.partition("::")
        assert name in defined_tests(REPO_ROOT / file), ref
