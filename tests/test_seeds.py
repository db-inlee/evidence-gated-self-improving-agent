"""LLM seed 규칙: derive_llm_seed(family, incident_seed, repeat)."""

import os
import subprocess
import sys

from evidencegate.seeds import derive_llm_seed
from opsworld.state import Family


def test_same_input_gives_same_seed() -> None:
    assert derive_llm_seed(Family.TRAFFIC_SPIKE, 900_001, 2) == derive_llm_seed(
        Family.TRAFFIC_SPIKE, 900_001, 2
    )


def test_family_given_as_plain_string_matches_enum() -> None:
    assert derive_llm_seed("bad_deployment", 5, 0) == derive_llm_seed(Family.BAD_DEPLOYMENT, 5, 0)


def test_seed_range_is_non_negative_31_bit() -> None:
    for family in Family:
        for incident_seed in range(200):
            for repeat in range(5):
                seed = derive_llm_seed(family, incident_seed, repeat)
                assert 0 <= seed < 2**31


def test_different_repeat_incident_seed_or_family_give_different_seeds() -> None:
    seeds = {
        derive_llm_seed(family, incident_seed, repeat)
        for family in Family
        for incident_seed in range(100)
        for repeat in range(5)
    }
    assert len(seeds) == len(Family) * 100 * 5


def test_seed_does_not_depend_on_process_hash_seed() -> None:
    script = (
        "from evidencegate.seeds import derive_llm_seed\n"
        "print([derive_llm_seed(f, s, r) for f in ('bad_deployment', 'traffic_spike')"
        " for s in range(10) for r in range(3)])\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        completed = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(completed.stdout)
    assert len(outputs) == 1


def test_known_value_is_stable() -> None:
    """sha256 기반 정의가 바뀌면 이전 결과와 paired 비교가 깨진다. 값을 고정해 둔다."""
    import hashlib

    digest = hashlib.sha256(b"bad_deployment:0:0").digest()
    expected = int.from_bytes(digest[:8], "big") % 2**31
    assert derive_llm_seed(Family.BAD_DEPLOYMENT, 0, 0) == expected
