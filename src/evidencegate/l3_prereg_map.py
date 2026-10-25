"""Ticket 006 사전 등록 대응표: §목표~§5, §7, §10 해석마다 구현 함수와 테스트.

`tests/test_l3_prereg_map.py`가 모든 항목에 함수와 테스트가 있고, 그 함수와 테스트가 실제로 있는지
확인한다. 빈칸이 있으면 실패한다(L3 분석 코드 고정의 조건, §10 해석 3). 함수는 "모듈:이름",
테스트는 "파일::테스트 이름"으로 쓴다. 코드가 없는 항목(범위 밖, 실행 시점)은 그 약속을 지키는
코드와 테스트에 대응시킨다.
"""

from evidencegate.prereg_map import PreregItem

EV = "evidencegate"
OW = "opsworld"
T_GEN = "tests/test_l3_generator.py"
T_L2 = "tests/test_l2_generator.py"
T_BENCH = "tests/test_l3_bench.py"
T_AN = "tests/test_l3_analysis.py"
T_CELL = "tests/test_cell_structures.py"
T_SCRIPT = "tests/test_run_l3_analysis_script.py"
T_DATA = "tests/test_eval_data.py"
MD = f"{T_AN}::test_markdown_has_korean_verdicts_and_required_notes"
GEN = f"{OW}.generator:generate_incident"
RUN = f"{EV}.l3_analysis:run_l3_analysis"


def _i(id: str, text: str, functions: tuple[str, ...], tests: tuple[str, ...]) -> PreregItem:
    return PreregItem(id, text, functions, tests)


ITEMS: tuple[PreregItem, ...] = (
    # ---- 목표 ----
    _i(
        "목표-1",
        "기존 세 계열 안에서 L2 방해 요인을 조합한 L3 장애, L1/L2 생성 결과 불변(골든 해시)",
        (GEN, f"{OW}.generator:_generate_l3"),
        (
            f"{T_GEN}::test_l3_combines_both_l2_subtypes_and_records_them_as_hidden_values",
            f"{T_L2}::test_l1_incidents_are_unchanged_by_ticket_003",
            f"{T_L2}::test_l2_incidents_are_unchanged_by_ticket_006",
        ),
    ),
    _i(
        "목표-2",
        "후보 19개와 S0가 L3 Oracle에서 실행된다(계획)",
        (f"{EV}.l3_bench:build_l3_plan",),
        (f"{T_BENCH}::test_l3_plan_has_the_ticket_blocks_and_7200_episodes",),
    ),
    _i(
        "목표-3",
        "Ticket 005 게이트 결정을 그대로 두고 L3 oracle Δ로 채택 오류 지표를 다시 계산해 비교",
        (RUN,),
        (
            f"{T_AN}::test_a_ticket_005_result_that_does_not_reproduce_stops_the_analysis",
            f"{T_AN}::test_l3_metrics_use_ticket_005_decisions_and_l3_estimates",
            f"{T_AN}::test_comparison_table_has_l12_l3_and_difference",
        ),
    ),
    _i(
        "목표-4",
        "새 장애 계열·도구·술어·조치 없음, 외부 벤치마크 범위 밖",
        (GEN,),
        (
            f"{T_GEN}::test_predicate_set_is_exactly_the_ticket_table",
            f"{T_GEN}::test_only_the_correct_action_can_have_its_required_evidence",
        ),
    ),
    _i(
        "목표-5",
        "이동 크기의 사전 기대: dep만 새 술어 조합, bad는 관측만 새로움, traffic은 대조 칸",
        (f"{EV}.l3_analysis:render_markdown_l3", f"{OW}.generator:_generate_l3"),
        (MD, f"{T_GEN}::test_traffic_spike_l3_has_the_l2_extra_deployment_structure"),
    ),
    # ---- 연구 질문 ----
    _i(
        "RQ-L3-0 기준값",
        "기준값: Ticket 005 Oracle S0(300개, 반복 5회 다수결)",
        (f"{EV}.l3_analysis:manipulation_samples",),
        (f"{T_AN}::test_rq_l3_0_point_estimates_by_hand",),
    ),
    _i(
        "RQ-L3-0 CD",
        "Δ₀ = CD(L3) - CD(L1/L2), 칸 층화 독립 bootstrap 2,000회 seed 0 95% 양측, 양 끝 규칙으로 "
        "Δ₀ < -ε",
        (f"{EV}.l3_analysis:manipulation_check",),
        (
            f"{T_AN}::test_manipulation_confirmed_when_l3_is_clearly_harder",
            f"{T_AN}::test_manipulation_undecidable_when_an_interval_crosses_the_boundary",
            f"{T_AN}::test_manipulation_bootstrap_is_one_seed_0_stream_l12_cells_first",
        ),
    ),
    _i(
        "RQ-L3-0 CD-ES",
        "g = CD율 - ES율(S0 다수결), g(L3) - g(L1/L2) > ε를 같은 방법으로",
        (f"{EV}.l3_analysis:manipulation_check",),
        (f"{T_AN}::test_manipulation_gap_uses_cd_minus_es",),
    ),
    _i(
        "RQ-L3-0 종합",
        "하나라도 확정 참이면 조작 확인, 둘 다 확정 거짓이면 조작 미달, 그 밖은 판정 불가",
        (f"{EV}.l3_analysis:manipulation_check",),
        (
            f"{T_AN}::test_manipulation_confirmed_when_l3_is_clearly_harder",
            f"{T_AN}::test_manipulation_not_confirmed_when_both_are_surely_false",
            f"{T_AN}::test_manipulation_undecidable_when_an_interval_crosses_the_boundary",
        ),
    ),
    _i(
        "RQ-L3-0 family",
        "전체와 family별로 보고(traffic은 대조 칸)",
        (f"{EV}.l3_analysis:manipulation_samples",),
        (f"{T_AN}::test_rq_l3_0_reports_all_and_each_family", MD),
    ),
    _i(
        "RQ-L3-0 미달",
        "미달이어도 아래 RQ는 그대로 보고",
        (RUN,),
        (f"{T_AN}::test_rq_section_of_the_result",),
    ),
    _i(
        "RQ-L3-1",
        "C1·C2·C4마다 Harmful(L3) > Harmful(L1/L2), 주 metric CD, ES는 보고, 종합과 판정 불가",
        (f"{EV}.l3_analysis:rq_l3_1",),
        (f"{T_AN}::test_rq_l3_1_categories", f"{T_AN}::test_rq_section_of_the_result"),
    ),
    _i(
        "RQ-L3-2",
        "RQ2 첫째, RQ3 첫째·추세(Budget Curve 판정 x L3), RQ2 둘째(민감도 C1 판정 x L3)의 판정 "
        "범주가 같은지(보존 / 바뀜), 종합 없음",
        (f"{EV}.l3_analysis:rq_l3_2", f"{EV}.analysis:_rq"),
        (
            f"{T_AN}::test_rq_l3_2_preserved_or_changed",
            f"{T_AN}::test_rq_section_of_the_result",
            f"{T_AN}::test_budget_curve_and_sensitivity_use_ticket_005_decisions_with_l3_oracle",
        ),
    ),
    _i(
        "RQ-L3-3",
        "후보마다 L1/L2 하위집합 라벨과 L3 라벨이 같은 비율, 바뀐 후보 목록, 칸 정의 해석 주의",
        (f"{EV}.l3_analysis:rq_l3_3",),
        (f"{T_AN}::test_rq_l3_3_counts_same_labels_and_lists_changes", MD),
    ),
    _i(
        "RQ 범위",
        "주 결론은 proposer 12개, 통제 후보는 보조, 모든 비교는 기술적(검정 없음)",
        (RUN, f"{EV}.l3_analysis:render_markdown_l3"),
        (f"{T_AN}::test_l1_l2_metrics_are_the_ticket_005_metrics", MD),
    ),
    # ---- 고정된 설계 결정 ----
    _i(
        "고정-1",
        "Ticket 001~005 고정 결정과 실행 조건(엔진, config 7c66ee5bf13c, tool_spec_id, S0)",
        (f"{EV}.eval_data:check_run_conditions", f"{EV}.l3_bench:l3_preflight"),
        (
            f"{T_AN}::test_l3_records_from_another_run_condition_stop_the_analysis",
            f"{T_BENCH}::test_l3_preflight_fails_when_any_check_fails",
        ),
    ),
    _i(
        "고정-2",
        "후보 풀 7c48a6d 그대로(L3를 보고 후보를 만들거나 고르지 않음)",
        (f"{EV}.l3_bench:build_l3_plan", f"{EV}.preflight:check_frozen_pool"),
        (
            f"{T_BENCH}::test_l3_plan_refuses_a_skill_that_does_not_match_the_pool",
            f"{T_BENCH}::test_l3_preflight_passes_and_includes_the_l3_preregistration",
        ),
    ),
    _i(
        "고정-3",
        "게이트 결정은 Ticket 005 결과 그대로, L3 Dev-A·Dev-B 없음, L3는 사후 채점 전용",
        (RUN, f"{EV}.l3_bench:l3_split_cases"),
        (
            f"{T_AN}::test_a_ticket_005_result_that_does_not_reproduce_stops_the_analysis",
            f"{T_BENCH}::test_l3_splits_match_the_ticket",
        ),
    ),
    _i(
        "고정-4",
        "판정 정의는 Ticket 005 §8 그대로, 칸 구조만 L3에 맞게",
        (
            f"{EV}.oracle:local_harm",
            f"{EV}.oracle:tau_cells",
            f"{EV}.gates:bootstrap_quantiles",
            f"{EV}.l3_analysis:l3_estimate",
        ),
        (
            f"{T_CELL}::test_local_harm_and_subsets_use_the_estimate_cells",
            f"{T_CELL}::test_bootstrap_quantiles_with_three_cells_match_a_manual_draw",
            f"{T_AN}::test_l3_oracle_delta_interval_and_cells_by_hand",
        ),
    ),
    _i(
        "고정-5",
        "L1/L2 불변: L1 골든 해시(기존), L2 골든 해시(L3 구현 전 기록)",
        (GEN,),
        (
            f"{T_L2}::test_l1_incidents_are_unchanged_by_ticket_003",
            f"{T_L2}::test_l2_incidents_are_unchanged_by_ticket_006",
        ),
    ),
    _i(
        "고정-6",
        "Ticket 005 분석과 분리: 명시적 칸·split 목록, L3_CELLS·L3_SPLITS, Level.L3가 Ticket 005 "
        "분석 동작을 바꾸지 않음",
        (f"{EV}.l3_analysis:l3_split", f"{EV}.l3_analysis:ticket005_bytes"),
        (
            f"{T_DATA}::test_cells_and_splits_come_from_explicit_lists",
            f"{T_BENCH}::test_ticket_005_splits_stay_l1_l2",
            f"{T_AN}::test_a_ticket_005_result_that_does_not_reproduce_stops_the_analysis",
        ),
    ),
    # ---- §1 L3 정의 ----
    _i(
        "§1-표",
        "L3 = 그 계열의 L2 하위 유형 둘을 동시에: bad(버그 배포 + 추가 배포 + CPU 포화), dep(창 안 "
        "배포 + 창 밖 방해 배포 + CPU 포화 + 트래픽 급증 + 의존성 "
        "장애), traffic(L2 extra_deployment "
        "구조)",
        (f"{OW}.generator:_generate_l3",),
        (
            f"{T_GEN}::test_bad_deployment_l3_has_buggy_and_extra_deployments_with_saturated_cpu",
            f"{T_GEN}::test_dependency_failure_l3_has_in_window_and_outside_deployments_and_traffic_trap",
            f"{T_GEN}::test_traffic_spike_l3_has_the_l2_extra_deployment_structure",
        ),
    ),
    _i(
        "§1-T1",
        "traffic은 다른 family의 방해 요인을 넣지 않는다(dependency_ruled_out 유지)",
        (f"{OW}.generator:_generate_l3",),
        (f"{T_GEN}::test_predicate_set_is_exactly_the_ticket_table",),
    ),
    _i(
        "§1-난수",
        "난수 흐름 '<family>:<seed>:L3', 소비 순서는 L2와 같음, L1·L2 흐름 그대로",
        (f"{OW}.generator:_generate_l3",),
        (
            f"{T_GEN}::test_l3_uses_its_own_random_stream_in_the_l2_order",
            f"{T_L2}::test_l2_incidents_are_unchanged_by_ticket_006",
        ),
    ),
    _i(
        "§1-id",
        "incident_id는 기존 규칙(<family>:<seed>:L3 해시)",
        (f"{OW}.generator:_incident_id",),
        (f"{T_GEN}::test_l3_differs_from_l1_and_l2_of_the_same_seed_and_has_its_own_id",),
    ),
    _i(
        "§1-숨은 값",
        "새 숨은 값 l3_subtypes, 도구 출력에 넣지 않음",
        (f"{OW}.generator:_generate_l3",),
        (
            f"{T_GEN}::test_l3_combines_both_l2_subtypes_and_records_them_as_hidden_values",
            f"{T_GEN}::test_tool_outputs_and_alert_do_not_reveal_level_or_subtypes",
        ),
    ),
    _i(
        "§1-dep 배포",
        "dep L3의 배포는 창 안 배포와 창 밖 방해 배포의 합집합",
        (f"{OW}.generator:_generate_l3",),
        (
            f"{T_GEN}::test_dependency_failure_l3_has_in_window_and_outside_deployments_and_traffic_trap",
        ),
    ),
    _i(
        "§1-범위 밖",
        "축 A 없음, 조작 확인 결과를 본 뒤 L3를 바꾸지 않음(§7)",
        (f"{OW}.generator:_generate_l3",),
        (f"{T_GEN}::test_l3_uses_its_own_random_stream_in_the_l2_order",),
    ),
    # ---- §2 생성기 불변식 ----
    _i(
        "§2-1",
        "술어 집합이 family마다 표와 정확히 같다",
        (GEN,),
        (f"{T_GEN}::test_predicate_set_is_exactly_the_ticket_table",),
    ),
    _i(
        "§2-2",
        "정답 조치만 근거를 갖출 수 있다(함정 술어 문장 포함)",
        (GEN,),
        (
            f"{T_GEN}::test_only_the_correct_action_can_have_its_required_evidence",
            f"{T_GEN}::test_the_traps_named_in_the_ticket_block_the_wrong_actions",
            f"{T_GEN}::test_reference_trajectory_is_evidence_safe",
        ),
    ),
    _i(
        "§2-3",
        "bad: 가장 최근 배포가 버그 배포이고 다른 배포와 8분 이상",
        (GEN,),
        (f"{T_GEN}::test_bad_deployment_most_recent_is_buggy_and_separated",),
    ),
    _i(
        "§2-4",
        "경계 여유 표(창 끝 3분, ±30분 2분, bad 추가 배포 5분, 트래픽 0.25/0.3, CPU 3%p)",
        (GEN,),
        (
            f"{T_GEN}::test_in_window_deployments_keep_the_margin_to_the_window_end",
            f"{T_GEN}::test_deployments_keep_the_margin_to_the_recent_deployment_boundary",
            f"{T_GEN}::test_bad_deployment_extra_deployment_is_well_outside_the_before_window",
            f"{T_GEN}::test_traffic_ratio_keeps_the_margin_to_both_thresholds",
            f"{T_GEN}::test_cpu_keeps_the_margin_to_the_saturation_threshold",
            f"{T_GEN}::test_margin_constants_match_the_ticket_table",
        ),
    ),
    _i(
        "§2-5",
        "결정성: 같은 (family, L3, seed)는 같은 incident",
        (GEN,),
        (
            f"{T_GEN}::test_l3_is_deterministic",
            f"{T_GEN}::test_l3_is_deterministic_across_processes",
        ),
    ),
    _i(
        "§2-6",
        "L1·L2 골든 해시 그대로",
        (GEN,),
        (
            f"{T_L2}::test_l1_incidents_are_unchanged_by_ticket_003",
            f"{T_L2}::test_l2_incidents_are_unchanged_by_ticket_006",
        ),
    ),
    # ---- §3 분할과 실행 규모 ----
    _i(
        "§3-split",
        "L3-Oracle 300(family마다 100, 400,000~400,099), L3-check 30(410,000~410,009), 다른 범위와 "
        "겹치지 않음",
        (f"{EV}.l3_bench:l3_split_cases",),
        (
            f"{T_BENCH}::test_l3_splits_match_the_ticket",
            f"{T_BENCH}::test_l3_split_seeds_do_not_overlap_other_splits",
        ),
    ),
    _i(
        "§3-check",
        "L3-check는 Oracle 밖이고 분석에 쓰지 않는다",
        (f"{EV}.l3_analysis:load_l3_records", f"{EV}.l3_bench:l3_check_block"),
        (
            f"{T_BENCH}::test_l3_check_block_is_s0_repeat_0_over_the_30_check_cases",
            f"{T_AN}::test_l3_split_is_300_cases_in_three_family_cells",
        ),
    ),
    _i(
        "§3-규모",
        "후보 19 x 300(반복 0) + S0 x 300(반복 0~4) = 7,200",
        (f"{EV}.l3_bench:build_l3_plan",),
        (
            f"{T_BENCH}::test_l3_plan_has_the_ticket_blocks_and_7200_episodes",
            f"{T_BENCH}::test_committed_l3_plan_matches_its_hash_and_a_rebuild",
        ),
    ),
    _i(
        "§3-실행기",
        "실행기·블록 순서·중단 규칙·실행 전 점검은 Ticket 005 §2 그대로",
        (f"{EV}.l3_bench:main", f"{EV}.l3_bench:l3_preflight", f"{EV}.bench_runner:run_blocks"),
        (
            f"{T_BENCH}::test_run_executes_the_plan_blocks_and_logs_to_ticket_006",
            f"{T_BENCH}::test_l3_plan_order_is_the_canonical_order_shuffled_by_random_0",
            f"{T_BENCH}::test_main_run_preflight_adds_the_l3_plan_check",
        ),
    ),
    # ---- §4 분석 ----
    _i(
        "§4-Δ",
        "L3 oracle Δ: 후보 반복 0 대 S0 반복 0~4 다수결, 3칸 "
        "층화 bootstrap 2,000회 seed 0 95% 양측",
        (f"{EV}.l3_analysis:l3_paired", f"{EV}.l3_analysis:l3_estimate"),
        (f"{T_AN}::test_l3_oracle_delta_interval_and_cells_by_hand",),
    ),
    _i(
        "§4-τ",
        "τ_cell: L3 S0 반복 5회 leave-one-out(칸마다), ε = 5%p, L1/L2와 L3 τ를 나란히",
        (f"{EV}.oracle:tau_cells",),
        (
            f"{T_AN}::test_tau_tables_side_by_side",
            f"{T_CELL}::test_tau_cells_accept_cells_and_a_cell_key",
        ),
    ),
    _i(
        "§4-지표",
        "지표 6개: Ticket 005 게이트 결정 x L3 oracle, "
        "불확정 규칙·Clopper-Pearson·집합 분리 그대로",
        (RUN, f"{EV}.eval_metrics:gate_metrics"),
        (
            f"{T_AN}::test_l3_metrics_use_ticket_005_decisions_and_l3_estimates",
            f"{T_AN}::test_l1_l2_metrics_are_the_ticket_005_metrics",
        ),
    ),
    _i(
        "§4-비교 표",
        "게이트 x metric x 지표마다 [L1/L2, L3, 차이]",
        (RUN,),
        (f"{T_AN}::test_comparison_table_has_l12_l3_and_difference",),
    ),
    _i(
        "§4-Local Harm",
        "주 지표는 L3 3칸, 보조 지표는 L1/L2를 family 단위(100개)로 다시 묶은 Local Harm, 3칸 대 "
        "3칸, 칸 수 해석 주의",
        (f"{EV}.l3_analysis:family_estimate", f"{EV}.oracle:local_harm"),
        (f"{T_AN}::test_family_level_local_harm_by_hand", MD),
    ),
    _i(
        "§4-F",
        "Oracle 적합성(F)은 L3에 적용하지 않는다",
        (RUN,),
        (f"{T_AN}::test_fitness_is_not_applied_to_l3", MD),
    ),
    _i(
        "§4-범주",
        "판정 범주: 지지 / 부분 지지 / 지지 안 됨 / 판정 불가(RQ-L3-2는 보존 / 바뀜), "
        "Ticket 005 §8-25~28·33",
        (f"{EV}.l3_analysis:rq_l3_1", f"{EV}.l3_analysis:rq_l3_2"),
        (f"{T_AN}::test_rq_l3_1_categories", f"{T_AN}::test_rq_l3_2_preserved_or_changed"),
    ),
    # ---- §5 범위 밖 ----
    _i(
        "§5",
        "새 계열·도구·술어·조치, 축 A·E, L3에서 후보 생성·게이트 재판정, 외부 벤치마크, v2 없음",
        (f"{EV}.l3_bench:l3_split_cases", RUN),
        (
            f"{T_BENCH}::test_l3_splits_match_the_ticket",
            f"{T_AN}::test_a_ticket_005_result_that_does_not_reproduce_stops_the_analysis",
        ),
    ),
    # ---- §7 결과를 보기 전에 정해 둔 것 ----
    _i(
        "§7-시점",
        "실행 시점: Ticket 005 직후(계획은 Ticket 005 계획과 따로)",
        (f"{EV}.l3_bench:build_l3_plan",),
        (f"{T_BENCH}::test_ticket_005_plan_is_not_the_l3_plan",),
    ),
    _i(
        "§7-미달",
        "조작 확인이 미달·판정 불가여도 RQ-L3를 그대로 보고하고 L3를 바꾸거나 다시 실행하지 않는다",
        (RUN, f"{EV}.l3_analysis:main"),
        (
            f"{T_AN}::test_rq_section_of_the_result",
            f"{T_AN}::test_main_writes_json_hash_and_markdown_once",
        ),
    ),
    # ---- §10 해석 ----
    _i(
        "§10-1",
        "RQ-L3-0 독립 bootstrap은 각 표본의 원래 설계 칸으로 층화(전체 6칸·3칸, family 2칸·1칸)",
        (f"{EV}.l3_analysis:manipulation_samples", f"{EV}.l3_analysis:manipulation_check"),
        (
            f"{T_AN}::test_rq_l3_0_reports_all_and_each_family",
            f"{T_AN}::test_manipulation_bootstrap_is_one_seed_0_stream_l12_cells_first",
        ),
    ),
    _i(
        "§10-2",
        "§2-1 술어 표의 '성립' 열이 술어 집합 전체(traffic L3의 "
        "dependency_failure_confirmed 불성립)",
        (GEN,),
        (f"{T_GEN}::test_predicate_set_is_exactly_the_ticket_table",),
    ),
    _i(
        "§10-3",
        "L3 분석 코드 고정 커밋, 고정 커밋을 직접 받아 확인하는 스크립트로 한 번만 분석",
        (f"{EV}.l3_analysis:main",),
        (
            f"{T_SCRIPT}::test_a_commit_with_later_analysis_changes_fails",
            f"{T_SCRIPT}::test_an_existing_result_is_not_overwritten",
            f"{T_AN}::test_main_writes_json_hash_and_markdown_once",
        ),
    ),
    _i(
        "§10-4",
        "(제안) RQ-L3-0 재표본: 칸 크기 [L1/L2 칸..., L3 칸...]로 seed 0 흐름 하나, CD와 g가 같은 "
        "재표본, 구간은 2.5·97.5 백분위수(유리수)",
        (f"{EV}.l3_analysis:manipulation_check",),
        (f"{T_AN}::test_manipulation_bootstrap_is_one_seed_0_stream_l12_cells_first",),
    ),
    _i(
        "§10-5",
        "(제안) L3 oracle 구간의 재표본 색인은 3칸 크기로 seed 0에서 새로 시작해 모든 후보가 공유",
        (f"{EV}.l3_analysis:l3_estimate", f"{EV}.gates:bootstrap_quantiles"),
        (f"{T_AN}::test_l3_oracle_delta_interval_and_cells_by_hand",),
    ),
    _i(
        "§10-6",
        "(제안) 보조 Local Harm: family Δ = L1/L2 oracle 짝 표의 family 100개 평균, τ_family = S0 "
        "L1/L2 Oracle 반복 5회를 family로 묶은 leave-one-out, 점추정",
        (f"{EV}.l3_analysis:family_estimate", f"{EV}.oracle:tau_cells"),
        (f"{T_AN}::test_family_level_local_harm_by_hand",),
    ),
    _i(
        "§10-7",
        "(제안) Ticket 005 결과를 같은 레코드로 다시 계산해 "
        "바이트까지 같아야 분석, Budget Curve·민감도 "
        "재표본 판정은 같은 코드로 다시 계산",
        (RUN, f"{EV}.l3_analysis:main"),
        (
            f"{T_AN}::test_a_ticket_005_result_that_does_not_reproduce_stops_the_analysis",
            f"{T_AN}::test_main_checks_the_registered_ticket_005_hash_by_default",
            f"{T_AN}::test_budget_curve_and_sensitivity_use_ticket_005_decisions_with_l3_oracle",
        ),
    ),
    _i(
        "§10-8",
        "(제안) 비교 표는 C1~C4(C3 보조) x CD·ES x 지표 6개, "
        "proposer와 통제 따로, 차이 = L3 - L1/L2",
        (RUN,),
        (f"{T_AN}::test_comparison_table_has_l12_l3_and_difference",),
    ),
    _i(
        "§10-9",
        "(제안) RQ-L3-1의 ES 판정도 같은 방식으로 내고 보고만 한다(주 metric CD)",
        (f"{EV}.l3_analysis:rq_l3_1",),
        (f"{T_AN}::test_rq_section_of_the_result",),
    ),
    _i(
        "§10-10",
        "(제안) RQ-L3-2의 보존은 판정 범주 문자열이 같은 것(판정 불가끼리도 보존)",
        (f"{EV}.l3_analysis:rq_l3_2",),
        (f"{T_AN}::test_rq_l3_2_preserved_or_changed",),
    ),
    _i(
        "§10-11",
        "(제안) RQ-L3-3은 라벨 다섯 개(uncertain 포함)를 그대로 비교하고 한쪽 이상 uncertain 수를 "
        "함께 보고, proposer 주·통제 보조, metric별",
        (f"{EV}.l3_analysis:rq_l3_3",),
        (f"{T_AN}::test_rq_l3_3_counts_same_labels_and_lists_changes",),
    ),
    _i(
        "§10-12",
        "(제안) L3 계획 블록 순서: 후보(풀 순서) 반복 0, S0 반복 "
        "0~4를 Random(0)으로 섞음(파일럿 없음). "
        "L3-check 점검은 계획 점검 없음, 본 실행 점검은 L3 계획 점검 포함",
        (f"{EV}.l3_bench:build_l3_plan", f"{EV}.l3_bench:l3_preflight"),
        (
            f"{T_BENCH}::test_l3_plan_order_is_the_canonical_order_shuffled_by_random_0",
            f"{T_BENCH}::test_main_run_preflight_adds_the_l3_plan_check",
        ),
    ),
)

L3_PREREG_IDS: tuple[str, ...] = (
    *(f"목표-{n}" for n in range(1, 6)),
    "RQ-L3-0 기준값",
    "RQ-L3-0 CD",
    "RQ-L3-0 CD-ES",
    "RQ-L3-0 종합",
    "RQ-L3-0 family",
    "RQ-L3-0 미달",
    "RQ-L3-1",
    "RQ-L3-2",
    "RQ-L3-3",
    "RQ 범위",
    *(f"고정-{n}" for n in range(1, 7)),
    "§1-표",
    "§1-T1",
    "§1-난수",
    "§1-id",
    "§1-숨은 값",
    "§1-dep 배포",
    "§1-범위 밖",
    *(f"§2-{n}" for n in range(1, 7)),
    "§3-split",
    "§3-check",
    "§3-규모",
    "§3-실행기",
    "§4-Δ",
    "§4-τ",
    "§4-지표",
    "§4-비교 표",
    "§4-Local Harm",
    "§4-F",
    "§4-범주",
    "§5",
    "§7-시점",
    "§7-미달",
    *(f"§10-{n}" for n in range(1, 13)),
)
