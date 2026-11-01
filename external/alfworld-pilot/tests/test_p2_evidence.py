"""P2 보조 지표 계산 (Ticket 008 §2). E-a 판정 문장 패턴을 여기서 고정한다."""

from alfworld_pilot.evidence import (
    StepRecord,
    contents_seen,
    e_a_holds,
    e_b_rate,
    e_c_holds,
    successful_take,
)


def test_contents_seen_reads_on_the_receptacle_sentences() -> None:
    obs = "You arrive at shelf 1. On the shelf 1, you see a pen 1, a book 2, and a cd 3."
    assert contents_seen(obs) == {("pen 1", "shelf 1"), ("book 2", "shelf 1"), ("cd 3", "shelf 1")}


def test_contents_seen_reads_in_it_after_open_or_is_open() -> None:
    opened = "You open the drawer 1. The drawer 1 is open. In it, you see a key 2."
    assert contents_seen(opened) == {("key 2", "drawer 1")}
    arrived = "You arrive at fridge 1. The fridge 1 is open. In it, you see a apple 1, and a egg 2."
    assert contents_seen(arrived) == {("apple 1", "fridge 1"), ("egg 2", "fridge 1")}


def test_contents_seen_ignores_nothing_closed_and_room_listing() -> None:
    assert contents_seen("You arrive at shelf 1. On the shelf 1, you see nothing.") == set()
    assert contents_seen("You arrive at drawer 1. The drawer 1 is closed.") == set()
    room = (
        "You are in the middle of a room. Looking quickly around you, you see a cabinet 1, "
        "and a shelf 2.\n\nYour task is to: put a pen in shelf."
    )
    assert contents_seen(room) == set()
    # "In it"은 같은 관측 안의 열린 수용체에만 붙는다
    assert contents_seen("In it, you see a key 2.") == set()


def test_successful_take_needs_the_pick_up_sentence() -> None:
    assert successful_take(
        "take pen 1 from shelf 2", "You pick up the pen 1 from the shelf 2."
    ) == (
        "pen 1",
        "shelf 2",
    )
    assert successful_take("take pen 1 from shelf 2", "Nothing happens.") is None
    assert successful_take("go to shelf 2", "You arrive at shelf 2.") is None


def step(command: str, obs: str, admissible: bool = True) -> StepRecord:
    return StepRecord(command=command, observation=obs, admissible=admissible)


FIRST = "You are in the middle of a room. Looking quickly around you, you see a shelf 1, a shelf 2."


def test_e_a_holds_when_the_object_was_seen_in_that_receptacle_before_the_take() -> None:
    steps = [
        step("go to shelf 2", "You arrive at shelf 2. On the shelf 2, you see a pen 1."),
        step("take pen 1 from shelf 2", "You pick up the pen 1 from the shelf 2."),
    ]
    assert e_a_holds(FIRST, steps, won=True)


def test_e_a_fails_for_a_take_from_an_unseen_neighbour_receptacle() -> None:
    # P1에서 본 경우: shelf 1에 서서 보지 않은 shelf 2의 pen 1을 꺼낸다
    steps = [
        step("go to shelf 1", "You arrive at shelf 1. On the shelf 1, you see nothing."),
        step("take pen 1 from shelf 2", "You pick up the pen 1 from the shelf 2."),
    ]
    assert not e_a_holds(FIRST, steps, won=True)


def test_e_a_needs_a_win_and_ignores_failed_takes() -> None:
    seen = step("go to shelf 2", "You arrive at shelf 2. On the shelf 2, you see a pen 1.")
    took = step("take pen 1 from shelf 2", "You pick up the pen 1 from the shelf 2.")
    failed = step("take cd 9 from shelf 1", "Nothing happens.", admissible=False)
    assert e_a_holds(FIRST, [seen, failed, took], won=True)
    assert not e_a_holds(FIRST, [seen, took], won=False)
    assert not e_a_holds(FIRST, [], won=True)  # 성공한 take가 없으면 성립하지 않는다


def test_e_a_uses_only_observations_strictly_before_the_take() -> None:
    steps = [
        step("take pen 1 from shelf 2", "You pick up the pen 1 from the shelf 2."),
        step("go to shelf 2", "You arrive at shelf 2. On the shelf 2, you see a pen 1."),
    ]
    assert not e_a_holds(FIRST, steps, won=True)


def test_e_b_rate_counts_admissible_steps() -> None:
    steps = [step("look", "x"), step("fly", "Nothing happens.", admissible=False)]
    assert e_b_rate(steps) == (1, 2)
    assert e_b_rate([]) == (0, 0)


def test_e_c_compares_steps_with_twice_the_expert_length() -> None:
    assert e_c_holds(won=True, steps=10, expert_steps=5)
    assert not e_c_holds(won=True, steps=11, expert_steps=5)
    assert not e_c_holds(won=False, steps=3, expert_steps=5)
    assert e_c_holds(won=True, steps=3, expert_steps=None) is None  # 전문가 실패 게임은 정의 안 됨
