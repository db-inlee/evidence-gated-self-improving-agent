"""Ticket 009b 사전 등록의 절 규칙(티켓 원칙 5, 완료 기준).

009 r1 본문을 어떻게 옮겼는지 기계적으로 확인한다.

절은 `## `·`### ` 머리글로 나눈다(머리글 앞 머리말은 이름 ""). 009 r1의 절마다 규칙이 하나다:

- SAME: 머리글과 본문이 바이트까지 같다.
- REPLACED: 머리글이 같고, 본문은 009 본문에 아래 (옛 문장, 새 문장) 쌍을 적용한 것과
  같다. 옛 문장은 009 본문에 정확히 한 번 있어야 한다. 쌍에 없는 차이는 허용하지 않는다.
- MOVED: 009 본문이 009b의 다른 절(새 머리글) 본문과 바이트까지 같다. 009의 머리글은 009b에도 남고
  그 본문은 새로 쓴다.
- PREFIX: 머리글이 같고, 009b 본문이 009 본문(끝 빈 줄 제외)으로 시작한다(뒤에 더함).
- FREE: 새로 쓴 절(머리말, 2.1, 11, 12).

009b에만 있는 절은 NEW_HEADINGS로 둔다.
"""

H1 = "## 1. 실행 조건과 분할"
H2 = "## 2. 데이터 분리"
H21 = "### 2.1 후보 생성 절차"
H3 = "## 3. Ticket 005 §8 해석의 처리"
H4 = "## 4. 게이트, 지표, 결과 라벨"
H41 = "### 4.1 게이트 (proposer 후보, Dev-B N = 420, CD와 ES 각각)"
H42 = "### 4.2 h(n_c) (보정 기록만 사용, `evidencegate.t009.rq9.cell_threshold_h`)"
H43 = "### 4.3 지표와 결과 라벨 (Oracle, proposer 후보)"
H5 = "## 5. δ(N)"
H6 = "## 6. 연구 질문과 판정 규칙 (proposer 후보, CD)"
H61 = "### 6.1 RQ9-1 (주 질문, 선택적 거르기) — `evidencegate.t009.rq9.rq9_1`"
H62 = "### 6.2 RQ9-2 (비용) — `evidencegate.t009.rq9.rq9_2`"
H63 = "### 6.3 RQ9-3 (v1 발견 재현) — `evidencegate.t009.rq9.rq9_3`"
H64 = "### 6.4 RQ9-4 (표본 크기, 기술용, 판정 없음)"
H65 = (
    "### 6.5 RQ9-5 (보조 질문, 같은 채택 수의 엄격한 평균 게이트와 비교)"
    " — `evidencegate.t009.rq9.rq9_5`"
)
H66 = "### 6.6 군집 민감도 (기술용 민감도 판정, 주 판정은 6.1~6.5 그대로)"
H7 = "## 7. 보고 항목"
H8 = "## 8. 고정 뒤 규칙"
H9 = "## 9. 예상 검정력 (계획용 모의, v1 공개 결과만 사용)"
H91 = "### 9.1 009 계획 모의 (009 사전 등록 9절 본문 그대로)"
H92 = "### 9.2 009b 진행 기준 N_go 모의"
H10 = "## 10. 정의를 정하며 본 것과 정한 경위"
H11 = "## 11. 구현 위치"
H12 = "## 12. 변경 기록"

SAME = (H4, H41, H43, H6, H61, H62, H63, H64, H65)
FREE = ("", H21, H11, H12)
MOVED = {H9: H91}
PREFIX = (H10,)
NEW_HEADINGS = (H66, H91, H92)

H_TABLE = """
| n_c | CD h | CD 달성 거짓 기각률 | ES h | ES 달성 거짓 기각률 |
| ---: | --- | --- | --- | --- |
| 70 | 4/35 | 7/200 | 1/35 | 43/1000 |
| 4 | 1/2 | 1/500 | 0 | 49/2000 |
| 8 | 1/4 | 17/500 | 0 | 79/2000 |
| 16 | 3/16 | 59/2000 | 1/16 | 9/1000 |
| 34 | 5/34 | 57/2000 | 1/34 | 9/250 |
"""

DELTA_TABLE = """
| N | CD δ | ES δ |
| ---: | --- | --- |
| 420 | 1/35 | 1/140 |
| 24 | 1/12 | 0 |
| 48 | 1/16 | 1/48 |
| 96 | 5/96 | 1/96 |
| 204 | 2/51 | 1/102 |
"""

THRESHOLDS = (
    "`results/ticket-009/thresholds.json`"
    "(sha256 `d030b1734811d8a0acc8fdfd198d3d2f0b8ce87c1e81faf322e1f6b1805c3520`)"
)

REPLACED: dict[str, tuple[tuple[str, str], ...]] = {
    H1: (
        (
            "  - S0: Dev-A 60(반복 0), Dev-B 420(반복 0), Oracle 300 \u00d7 반복 0~4.",
            "  - S0: Dev-A 60(반복 0, 009에서 실행한 기록을 그대로 쓴다), Dev-B 420(반복 0), "
            "Oracle 300 \u00d7 반복 0~4.",
        ),
        (
            "  - proposer 후보: Dev-B 420 + Oracle 300(반복 0).",
            "  - proposer 후보(동결한 풀 전체, N_go ≤ N ≤ 40, 2.1·8절): "
            "Dev-B 420 + Oracle 300(반복 0).",
        ),
    ),
    H2: (
        (
            "- C5는 oracle 정보를 쓰지 않는다. h는 보정 S0 반복에서만 계산하고, "
            "τ_cell(Oracle S0)은 게이트에 쓰지 않는다.\n",
            "- C5는 oracle 정보를 쓰지 않는다. h는 보정 S0 반복에서만 계산하고, "
            "τ_cell(Oracle S0)은 게이트에 쓰지 않는다.\n"
            "- 009 proposer 기록(`runs/proposer-009/`, 후보 80개)은 proposer 입력에 넣지 않는다. "
            "중복 판정 정의의 근거(10절)와 사후 중복 표시에만 쓴다. "
            "009 후보는 009b 풀에 넣지 않는다.\n",
        ),
    ),
    H3: (
        (
            "| 2 | 이어 씀 | δ(N) 알고리즘(아래 5절). 값은 009 보정으로 다시 계산 |",
            "| 2 | 이어 씀 | δ(N) 알고리즘(아래 5절). 값은 009에서 계산한 "
            "`results/ticket-009/thresholds.json`의 값(다시 계산하지 않음) |",
        ),
    ),
    H42: (
        (
            "- 이 문서는 계산 방법만 고정한다. h 값은 보정 추가 실행 뒤, 후보 생성 전에 계산해 "
            "별도 커밋으로 남긴다.\n",
            "- h 값은 009에서 보정 추가 실행 뒤 이 방법으로 계산해 기록한 "
            + THRESHOLDS
            + "의 값을 쓴다. 다시 계산하지 않는다. "
            "그 계산은 009 최초 사전 등록(태그 `t009-prereg`, "
            "sha256 `efb79e1b2a7756298d4ac8bcc641eec33d820234701e01ef492a54b2021c2333`) 아래에서 "
            "했다. 실행 전 점검(preflight)에서 이 파일의 sha256을 확인한다.\n" + H_TABLE,
        ),
    ),
    H5: (
        (
            "- N ∈ {420, 24, 48, 96, 204}마다 계산한다. 값은 보정 추가 실행 뒤 h와 같은 커밋으로 "
            "기록한다. ES의 δ는 계산한 값을 그대로 쓴다.\n",
            "- N ∈ {420, 24, 48, 96, 204}마다 계산한다. 값은 4.2와 같은 파일("
            + THRESHOLDS
            + ")의 값을 쓴다. 다시 계산하지 않는다. ES의 δ는 계산한 값을 그대로 쓴다.\n"
            + DELTA_TABLE,
        ),
    ),
    H7: (
        (
            "- 보정 기록은 시점이 둘이다. 900,000~900,019는 v1 기간, 900,020~900,069는 009 기간에 "
            "실행했다. 실행 조건은 같다(1절).\n",
            "- 보정 기록은 시점이 둘이다. 900,000~900,019는 v1 기간, 900,020~900,069는 009 기간에 "
            "실행했다. 실행 조건은 같다(1절).\n"
            "- 009 종료 내력: 009는 풀 단계에서 미달(풀 14 / 목표 40)로 닫았고 Dev-B·Oracle은 "
            "실행하지 않았다(태그 `ticket-009-closed`).\n"
            "- 후보 생성 기록: 호출 수, 멈춘 이유(풀 40, 포화, 120회), 풀 크기 N과 진행 기준(8절), "
            "중복 표시(통제·v1·009 후보), 비용, 응답 모델 이름과 009 스냅숏(`gpt-5.4-2026-03-05`) "
            "비교\n"
            "- 군집 민감도(6.6): 묶음별 풀 구성, "
            "대표 풀의 RQ9-1 민감도 판정과 RQ9-3·RQ9-5 기술용 값, "
            "주 판정과 다르면 한계\n",
        ),
    ),
    H8: (
        (
            "- 후보 풀은 유효성 검사(Ticket 004 §5)와 중복 제거(Jaccard 0.6) "
            "외의 이유로 빼지 않는다. "
            "v1 후보·통제 후보와 중복이면 남기되 표시한다. 동결 뒤에는 더하거나 빼지 않는다.",
            "- 후보 풀은 유효성 검사(Ticket 004 §5)와 중복 제거(2.1의 W, t = 0.55) "
            "외의 이유로 빼지 "
            "않는다. 통제 후보·v1 후보·009 후보와 중복이면 남기되 표시한다. 동결 뒤에는 더하거나 "
            "빼지 않는다.",
        ),
        (
            "- 유효 proposer 후보가 40개에 못 미치면 "
            '"미달"로 기록하고, 진행 여부는 사용자가 정한다. '
            "진행하면 판정 규칙은 그대로 쓴다.",
            "- 진행 기준: 호출이 멈췄을 때(풀 40, 포화, 120회 중 어느 이유든) "
            "풀이 N_go = 25 이상이면 "
            '그 풀로 본 실행을 한다. 미만이면 "미달"로 기록하고 멈춘다. 비용 상한·잔액 부족·형식 '
            "오류율·인증·서버 오류로 멈춘 경우는 이 기준이 아니다. 원인을 바로잡고 같은 설정으로 "
            "이어 부르거나, 진행 여부를 사용자가 정한다. 본 실행을 하면 판정 규칙은 그대로 쓴다.",
        ),
    ),
}


def split_sections(text: str) -> dict[str, str]:
    """머리글 → 본문(머리글 줄 다음부터 다음 머리글 앞까지, 줄바꿈 포함). 머리말은 ""."""
    sections: dict[str, str] = {}
    heading = ""
    body: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith(("## ", "### ")):
            if heading in sections:
                raise ValueError(f"duplicate heading {heading!r}")
            sections[heading] = "".join(body)
            heading, body = line.rstrip("\n"), []
        else:
            body.append(line)
    if heading in sections:
        raise ValueError(f"duplicate heading {heading!r}")
    sections[heading] = "".join(body)
    return sections


def ruled_headings() -> list[str]:
    """규칙이 있는 009 r1 머리글(머리말 "" 제외)."""
    return [h for h in (*SAME, *FREE, *MOVED, *PREFIX, *REPLACED) if h]


def apply_replacements(body: str, pairs: tuple[tuple[str, str], ...]) -> str:
    for old, new in pairs:
        if body.count(old) != 1:
            raise ValueError(f"replacement source must appear once: {old[:60]!r}")
        body = body.replace(old, new)
    return body


def check(text_009: str, text_009b: str) -> list[str]:
    """규칙을 어긴 곳의 설명 목록. 비면 통과."""
    old, new = split_sections(text_009), split_sections(text_009b)
    problems = []
    expected = sorted([*old, *NEW_HEADINGS])
    if sorted(new) != expected:
        problems.append(f"headings differ: missing {sorted(set(expected) - set(new))}, "
                        f"extra {sorted(set(new) - set(expected))}")  # fmt: skip
    for heading in SAME:
        if new.get(heading) != old[heading]:
            problems.append(f"{heading}: not byte-identical to 009 r1")
    for heading, pairs in REPLACED.items():
        try:
            want = apply_replacements(old[heading], pairs)
        except ValueError as error:
            problems.append(f"{heading}: {error}")
            continue
        if new.get(heading) != want:
            problems.append(f"{heading}: differs beyond the listed replacements")
    for heading, target in MOVED.items():
        if new.get(target) != old[heading]:
            problems.append(f"{heading} -> {target}: moved body is not byte-identical")
    for heading in PREFIX:
        if not new.get(heading, "").startswith(old[heading].rstrip("\n")):
            problems.append(f"{heading}: does not start with the 009 r1 body")
    return problems
