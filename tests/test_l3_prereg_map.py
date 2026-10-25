"""Ticket 006 사전 등록 대응표: §목표~§5, §7, §10 해석 모두에 구현 함수와 테스트가 있다."""

import ast
import importlib
from pathlib import Path

import pytest

from evidencegate.l3_analysis import SECTION10_VERSION
from evidencegate.l3_prereg_map import ITEMS, L3_PREREG_IDS
from evidencegate.prereg_map import PreregItem

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_preregistered_item_and_interpretation_is_mapped_in_order() -> None:
    ids = [item.id for item in ITEMS]
    assert len(ids) == len(set(ids))
    assert ids == list(L3_PREREG_IDS)


def test_the_result_records_the_same_number_of_section_10_items() -> None:
    last = max(int(i.removeprefix("§10-")) for i in L3_PREREG_IDS if i.startswith("§10-"))
    assert f"items 1-{last})" in SECTION10_VERSION


def test_every_item_has_text_a_function_and_a_test() -> None:
    for item in ITEMS:
        assert item.text and item.functions and item.tests, item.id


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.id)
def test_mapped_functions_exist(item: PreregItem) -> None:
    for ref in item.functions:
        module_name, _, qualname = ref.partition(":")
        obj: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        assert callable(obj), ref


def defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i.id)
def test_mapped_tests_exist(item: PreregItem) -> None:
    for ref in item.tests:
        file, _, name = ref.partition("::")
        assert name in defined_tests(REPO_ROOT / file), ref
