"""사전 등록 대응표 (Ticket 005 §1): 7.8절 항목과 §8 해석마다 구현 함수와 테스트.

`tests/test_prereg_map.py`가 모든 항목에 함수와 테스트가 있고, 그 함수와 테스트가 실제로 있는지
확인한다. 빈칸이 있으면 실패한다(분석 코드 고정의 조건). 함수는 "모듈:이름", 테스트는
"파일::테스트 이름"으로 쓴다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreregItem:
    id: str
    text: str
    functions: tuple[str, ...]
    tests: tuple[str, ...]


EV = "evidencegate"
T_DATA = "tests/test_eval_data.py"
T_EXACT = "tests/test_exact.py"
T_GATES = "tests/test_gates.py"
T_ORACLE = "tests/test_oracle.py"
T_METRICS = "tests/test_eval_metrics.py"
T_RQ = "tests/test_rq.py"
T_BUDGET = "tests/test_budget_curve.py"
T_ANALYSIS = "tests/test_analysis.py"
MD_TEST = f"{T_ANALYSIS}::test_markdown_has_korean_verdicts_and_required_notes"


def _i(id: str, text: str, functions: tuple[str, ...], tests: tuple[str, ...]) -> PreregItem:
    return PreregItem(id, text, functions, tests)


ITEMS: tuple[PreregItem, ...] = (
    # ---- 7.8 A. 실행 조건과 기준선 ----
    _i(
        "7.8 A-1",
        "실행 조건(정책, config, 장비, tool_spec, S0) 고정",
        (f"{EV}.eval_data:check_run_conditions", f"{EV}.analysis:run_analysis"),
        (
            f"{T_DATA}::test_expected_run_condition_is_enforced",
            f"{T_ANALYSIS}::test_mixed_run_conditions_stop_the_analysis",
        ),
    ),
    _i(
        "7.8 A-2",
        "분할 Dev-A 60 / Dev-B 210 / Oracle 300, 칸 = family x level 6개",
        (f"{EV}.eval_data:split_cases", f"{EV}.eval_data:cell_of"),
        (
            f"{T_DATA}::test_split_cases_have_the_preregistered_sizes_and_order",
            f"{T_DATA}::test_cells_are_family_by_level_in_sorted_order",
        ),
    ),
    _i(
        "7.8 A-3",
        "후보는 모든 split 반복 1회, S0는 Dev-A·Dev-B 반복 1회와 Oracle 반복 5회",
        (f"{EV}.eval_data:paired_gate", f"{EV}.eval_data:paired_oracle"),
        (
            f"{T_DATA}::test_paired_gate_uses_repeat_0_for_both",
            f"{T_DATA}::test_paired_oracle_needs_all_five_s0_repeats",
        ),
    ),
    _i(
        "7.8 A-4",
        "게이트가 보는 S0: Dev-A·Dev-B 반복 0 한 번",
        (f"{EV}.eval_data:paired_gate",),
        (f"{T_DATA}::test_paired_gate_uses_repeat_0_for_both",),
    ),
    _i(
        "7.8 A-5",
        "Oracle 채점이 보는 S0: 반복 5회 다수결",
        (f"{EV}.eval_data:majority", f"{EV}.eval_data:paired_oracle"),
        (
            f"{T_DATA}::test_majority_of_five",
            f"{T_DATA}::test_paired_oracle_uses_candidate_repeat_0_and_s0_majority",
        ),
    ),
    _i(
        "7.8 A-6",
        "노이즈 기준: 보정 목록 S0 반복 5회, metric별",
        (f"{EV}.analysis:run_analysis",),
        (f"{T_ANALYSIS}::test_delta_is_exact_from_the_noise_records",),
    ),
    _i(
        "7.8 A-7",
        "δ(N) = null_diff_quantile(노이즈, metric, N, q 0.95, 2,000회, seed 0), ES δ = 0",
        (f"{EV}.exact:null_diff_quantile_exact",),
        (
            f"{T_EXACT}::test_exact_delta_matches_null_diff_quantile",
            f"{T_EXACT}::test_exact_delta_of_identical_repeats_is_zero",
        ),
    ),
    _i(
        "7.8 A-8",
        "한계: δ(N)은 칸 층화 없이, C4 bootstrap은 칸 층화로 재표본",
        (f"{EV}.analysis:render_markdown",),
        (MD_TEST,),
    ),
    # ---- 7.8 B. 게이트 ----
    _i(
        "7.8 B-1",
        "케이스 Δ = 후보 - S0(같은 케이스, 둘 다 반복 0), 평균 Δ",
        (f"{EV}.eval_data:PairedTable", f"{EV}.exact:mean_fraction"),
        (
            f"{T_DATA}::test_paired_gate_uses_repeat_0_for_both",
            f"{T_EXACT}::test_mean_fraction_is_exact",
        ),
    ),
    _i(
        "7.8 B-2",
        "C1: Dev-A 후보 평균 > S0 평균, 같으면 기각",
        (f"{EV}.gates:gate_c1",),
        (f"{T_GATES}::test_c1_accepts_a_higher_mean", f"{T_GATES}::test_c1_rejects_an_equal_mean"),
    ),
    _i(
        "7.8 B-3",
        "C2: Dev-B 평균 Δ >= 0",
        (f"{EV}.gates:gate_c2",),
        (f"{T_GATES}::test_c2_accepts_zero_mean", f"{T_GATES}::test_c2_rejects_any_regression"),
    ),
    _i(
        "7.8 B-4",
        "C3(CD 보조): 평균 Δ > δ(N)",
        (f"{EV}.gates:gate_c3",),
        (
            f"{T_GATES}::test_c3_rejects_mean_equal_to_delta",
            f"{T_GATES}::test_c3_accepts_one_case_above_delta",
        ),
    ),
    _i(
        "7.8 B-5",
        "C4: 칸 층화 paired bootstrap 2,000회 5번째 백분위수 > 0 그리고 평균 Δ > δ(N)",
        (f"{EV}.gates:gate_c4", f"{EV}.gates:bootstrap_lower"),
        (
            f"{T_GATES}::test_c4_needs_the_mean_above_delta_too",
            f"{T_GATES}::test_c4_needs_the_lower_bound_above_zero_too",
            f"{T_GATES}::test_bootstrap_lower_matches_the_reference_calculation",
        ),
    ),
    _i(
        "7.8 B-6",
        "주 비교는 C1·C2·C4, C3는 보조, ES의 C3도 보고",
        (f"{EV}.analysis:run_analysis", f"{EV}.analysis:render_markdown"),
        (f"{T_GATES}::test_c3_with_zero_delta_rejects_zero_mean_unlike_c2", MD_TEST),
    ),
    _i(
        "7.8 B-7",
        "판정은 N = 210(Dev-B 전체)",
        (f"{EV}.analysis:run_analysis",),
        (f"{T_ANALYSIS}::test_candidate_gates_match_decide_gates",),
    ),
    # ---- 7.8 C. Budget Curve와 RQ2 민감도 ----
    _i(
        "7.8 C-1",
        "Budget Curve: N = 24·48·96·204, 칸 층화 비복원 1,000회, seed 0, "
        "C2·C4(+C3) 재판정, 지표 재표본 평균",
        (
            f"{EV}.budget_curve:resample_positions",
            f"{EV}.budget_curve:budget_decisions",
            f"{EV}.budget_curve:budget_points",
        ),
        (
            f"{T_BUDGET}::test_resample_positions_are_stratified_without_replacement",
            f"{T_BUDGET}::test_budget_decisions_match_the_gate_functions_on_explicit_subtables",
        ),
    ),
    _i(
        "7.8 C-2",
        "RQ2 민감도: Dev-A 36/18/6, family 안 L1:L2 1:1, 복원 1,000회, seed 0, C1 재계산",
        (f"{EV}.budget_curve:sensitivity_positions", f"{EV}.budget_curve:sensitivity_decisions"),
        (
            f"{T_BUDGET}::test_sensitivity_resample_cell_composition",
            f"{T_BUDGET}::test_sensitivity_decisions_recompute_c1_on_the_resample",
        ),
    ),
    _i(
        "7.8 C-3",
        "한계: proposer 입력의 치우침은 재현하지 않음",
        (f"{EV}.analysis:render_markdown",),
        (MD_TEST,),
    ),
    # ---- 7.8 D. Oracle 채점과 지표 ----
    _i(
        "7.8 D-1",
        "Oracle Δ = 후보(반복 1회) - S0 5회 다수결, 전체와 칸별",
        (f"{EV}.oracle:estimate",),
        (
            f"{T_ORACLE}::test_estimate_overall_and_cell_deltas",
            f"{T_ANALYSIS}::test_candidate_oracle_delta_matches_the_paired_table",
        ),
    ),
    _i(
        "7.8 D-2",
        "Oracle Δ 구간: paired bootstrap 95% 양측, 2,000회, 칸 층화, seed 0",
        (f"{EV}.oracle:estimate", f"{EV}.gates:bootstrap_quantiles"),
        (
            f"{T_ORACLE}::test_estimate_interval_is_the_2_5_and_97_5_percentiles",
            f"{T_ORACLE}::test_bootstrap_quantiles_match_the_reference",
        ),
    ),
    _i(
        "7.8 D-3",
        "실질 효과 임계값 ε = 5%p",
        (f"{EV}.oracle:useful", f"{EV}.oracle:harmful"),
        (
            f"{T_ORACLE}::test_epsilon_is_five_points",
            f"{T_ORACLE}::test_oracle_delta_exactly_epsilon_is_not_useful_but_uncertain",
        ),
    ),
    _i(
        "7.8 D-4",
        "τ_cell = max(ε, 칸 잡음 q95), leave-one-out 4회 평균",
        (f"{EV}.oracle:tau_cells",),
        (f"{T_ORACLE}::test_tau_by_hand", f"{T_ORACLE}::test_tau_is_epsilon_without_noise"),
    ),
    _i(
        "7.8 D-5",
        "불확정 규칙, Unsupported는 예외",
        (f"{EV}.oracle:useful", f"{EV}.oracle:harmful", f"{EV}.oracle:unsupported"),
        (
            f"{T_ORACLE}::test_useful_uses_the_plus_epsilon_boundary",
            f"{T_ORACLE}::test_unsupported_is_lower_bound_at_most_zero_without_uncertainty",
        ),
    ),
    _i(
        "7.8 D-6",
        "지표 6개(Harmful, Unsupported, Useful, Expected Oracle Gain, False Rejection, Local Harm)",
        (f"{EV}.eval_metrics:gate_metrics",),
        (
            f"{T_METRICS}::test_harmful_acceptance_rate_excludes_uncertain",
            f"{T_METRICS}::test_unsupported_counts_every_accepted_candidate",
            f"{T_METRICS}::test_useful_precision_excludes_the_uncertain_candidate",
            f"{T_METRICS}::test_expected_oracle_gain_is_the_mean_delta_of_accepted",
            f"{T_METRICS}::test_false_rejection_rate_uses_certain_useful_candidates",
            f"{T_METRICS}::test_local_harm_acceptance_uses_cells_without_exclusion",
        ),
    ),
    _i(
        "7.8 D-7",
        "비율은 분자/분모와 Clopper-Pearson 95%, 게이트 간 비교는 기술적",
        (f"{EV}.exact:clopper_pearson", f"{EV}.eval_metrics:Rate"),
        (
            f"{T_EXACT}::test_clopper_pearson_known_values",
            f"{T_METRICS}::test_rate_value_and_interval",
        ),
    ),
    _i(
        "7.8 D-8",
        "보조 지표: expected_effect 방향과 oracle 방향의 일치율, CD·ES 각각",
        (f"{EV}.eval_metrics:effect_agreement",),
        (f"{T_METRICS}::test_effect_agreement_excludes_uncertain_and_cards_without_effect",),
    ),
    # ---- 7.8 E. 하위집합 ----
    _i(
        "7.8 E-1",
        "positive / negative / mixed / neutral(neutral 안의 큰 칸)",
        (f"{EV}.oracle:subset_label",),
        (
            f"{T_ORACLE}::test_subset_positive_and_mixed_with_mixed_tau_cells",
            f"{T_ORACLE}::test_subset_negative",
            f"{T_ORACLE}::test_subset_neutral_reports_big_cells_by_tau",
        ),
    ),
    _i(
        "7.8 E-2",
        "불확정 규칙 적용, 통제 후보 사전 라벨 유지, 다르면 '예상과 다름'",
        (f"{EV}.oracle:check_control_label",),
        (
            f"{T_ORACLE}::test_control_with_a_different_label_is_unexpected",
            f"{T_ORACLE}::test_uncertain_control_is_reported_as_uncertain_not_unexpected",
        ),
    ),
    _i(
        "7.8 E-3",
        "결과표 행: Positive/Negative/Mixed/Neutral controls, Proposer 생성과 하위집합별",
        (f"{EV}.analysis:_results_table",),
        (f"{T_ANALYSIS}::test_results_table_rows_follow_7_8_e",),
    ),
    # ---- 7.8 F. Oracle 적합성 ----
    _i(
        "7.8 F-1",
        "부호 기준: 통제 후보의 같은 family에 > τ 칸과 < -τ 칸",
        (f"{EV}.oracle:fitness",),
        (
            f"{T_ORACLE}::test_fitness_sign_criterion_within_a_family",
            f"{T_ORACLE}::test_fitness_sign_needs_the_same_family",
        ),
    ),
    _i(
        "7.8 F-2",
        "크기 기준: 같은 family 두 칸의 Δ_cell 차이 >= 30%p",
        (f"{EV}.oracle:fitness",),
        (f"{T_ORACLE}::test_fitness_size_criterion_is_at_least_30_points",),
    ),
    _i(
        "7.8 F-3",
        "CD·ES 각각, 둘 다 못 맞추면 '미달'",
        (f"{EV}.oracle:fitness", f"{EV}.analysis:render_markdown"),
        (
            f"{T_ORACLE}::test_fitness_fails_when_neither_criterion_is_met",
            f"{T_ANALYSIS}::test_fitness_and_tau_sections",
        ),
    ),
    # ---- 7.8 G. 가설과 보고 원칙 ----
    _i(
        "7.8 G-1",
        "RQ1: CD 채택 중 ES 해로움 비율 > ES 채택 중 ES 해로움 비율(C2·C4 각각)",
        (f"{EV}.rq:rq1",),
        (
            f"{T_RQ}::test_rq1_supported_when_both_gates_hold",
            f"{T_RQ}::test_rq1_from_gate_metrics_when_c4_accepts_nothing",
        ),
    ),
    _i(
        "7.8 G-2",
        "RQ2: C1의 Harmful·Local Harm이 C2·C4보다 높고, 민감도 분석에서 차이가 커짐",
        (f"{EV}.rq:rq2_first", f"{EV}.rq:rq2_second", f"{EV}.budget_curve:sensitivity_rates"),
        (
            f"{T_RQ}::test_rq2_first_supported",
            f"{T_RQ}::test_rq2_second_grows_and_does_not_grow",
            f"{T_BUDGET}::test_sensitivity_rates_average_c1_harmful_and_local_harm",
        ),
    ),
    _i(
        "7.8 G-3",
        "RQ3: C4가 C2보다 Useful 높고 Harmful 낮음, N이 작을수록 차이가 큼",
        (f"{EV}.rq:rq3_first", f"{EV}.rq:rq3_trend"),
        (f"{T_RQ}::test_rq3_first_supported_partial_not", f"{T_RQ}::test_rq3_trend_categories"),
    ),
    _i(
        "7.8 G-4",
        "C2는 Δ = 0도 채택해 neutral이 많으면 정밀도가 떨어진다(해석 명시)",
        (f"{EV}.analysis:render_markdown",),
        (MD_TEST,),
    ),
    _i(
        "7.8 G-5",
        "의도된 조건을 결과 해석에 명시",
        (f"{EV}.analysis:render_markdown",),
        (MD_TEST,),
    ),
    _i(
        "7.8 G-6",
        "주요 결론은 proposer 후보로만, 통제 후보와 C3는 보조",
        (f"{EV}.analysis:run_analysis", f"{EV}.analysis:render_markdown"),
        (f"{T_ANALYSIS}::test_metrics_are_reported_per_group_and_never_pooled", MD_TEST),
    ),
    _i(
        "7.8 G-7",
        "후보 주제 분포는 기술용, 판정과 라벨에 쓰지 않음",
        (f"{EV}.analysis:render_markdown",),
        (MD_TEST,),
    ),
    _i(
        "7.8 G-8",
        "결과가 가설과 다르면 그대로, 사전 기준 미달은 '미달'",
        (f"{EV}.analysis:render_markdown", f"{EV}.oracle:fitness"),
        (MD_TEST, f"{T_ORACLE}::test_fitness_fails_when_neither_criterion_is_met"),
    ),
    # ---- 티켓 §8 해석 1~33 ----
    _i(
        "§8-1",
        "seed와 난수: 호출마다 seed 0, 공통 난수, PCG64",
        (f"{EV}.exact:stratified_bootstrap_indices", f"{EV}.exact:stratified_subsample_indices"),
        (
            f"{T_EXACT}::test_bootstrap_indices_restart_from_the_seed_every_call",
            f"{T_EXACT}::test_subsample_is_deterministic_and_seeded",
        ),
    ),
    _i(
        "§8-2",
        "δ(N) 추출: null_diff_quantile 그대로(모든 N 복원 추출)",
        (f"{EV}.exact:null_diff_quantile_exact",),
        (f"{T_EXACT}::test_exact_delta_matches_null_diff_quantile",),
    ),
    _i(
        "§8-3",
        "백분위수 선형 보간",
        (f"{EV}.exact:percentile_exact",),
        (
            f"{T_EXACT}::test_percentile_exact_interpolates_linearly",
            f"{T_EXACT}::test_percentile_exact_matches_the_float_percentile",
        ),
    ),
    _i(
        "§8-4",
        "판정 비교는 유리수",
        (f"{EV}.exact:mean_fraction", f"{EV}.gates:gate_c3"),
        (
            f"{T_GATES}::test_c3_rejects_a_float_delta",
            f"{T_GATES}::test_c3_rejects_mean_equal_to_delta",
        ),
    ),
    _i(
        "§8-5",
        "후보 반복 번호는 모든 split에서 0",
        (f"{EV}.eval_data:paired_oracle", f"{EV}.eval_data:paired_gate"),
        (f"{T_DATA}::test_paired_oracle_uses_candidate_repeat_0_and_s0_majority",),
    ),
    _i(
        "§8-6",
        "처분 없이 끝난 에피소드는 기록 그대로, 개수 보고",
        (f"{EV}.eval_data:outcomes", f"{EV}.analysis:run_analysis"),
        (
            f"{T_DATA}::test_episode_without_disposition_counts_as_recorded",
            f"{T_ANALYSIS}::test_format_errors_and_no_disposition_are_counted",
        ),
    ),
    _i(
        "§8-7",
        "C4 층화: 칸 6개, 칸 크기만큼 복원 추출(Budget Curve는 재표본 칸 크기)",
        (f"{EV}.gates:bootstrap_quantiles",),
        (
            f"{T_GATES}::test_bootstrap_is_stratified_by_cell",
            f"{T_GATES}::test_c4_works_on_a_smaller_stratified_subsample",
        ),
    ),
    _i(
        "§8-8",
        "ES의 C3는 계산해 보고, 주 비교에는 넣지 않음",
        (f"{EV}.gates:gate_c3",),
        (f"{T_GATES}::test_c3_with_zero_delta_rejects_zero_mean_unlike_c2",),
    ),
    _i(
        "§8-9",
        "Budget Curve seed: 재표본 공유, C4는 seed 0으로 새로 시작",
        (f"{EV}.budget_curve:budget_decisions",),
        (
            f"{T_BUDGET}::test_all_candidates_share_the_same_resamples",
            f"{T_BUDGET}::test_c4_with_precomputed_blocks_equals_c4_without",
        ),
    ),
    _i(
        "§8-10",
        "Budget Curve 분모 0 재표본은 빼고 수를 보고, 2.5·97.5 백분위수",
        (f"{EV}.budget_curve:summarize_values",),
        (f"{T_BUDGET}::test_summarize_values_drops_undefined_resamples_and_counts_them",),
    ),
    _i(
        "§8-11",
        "Budget Curve의 oracle 값은 고정",
        (f"{EV}.budget_curve:budget_points",),
        (f"{T_BUDGET}::test_budget_points_count_resamples_where_a_gate_accepts_nothing",),
    ),
    _i(
        "§8-12",
        "RQ2 민감도 재표본: 칸마다 복원 18·18·9·9·3·3, 공유, C1만",
        (f"{EV}.budget_curve:sensitivity_positions",),
        (
            f"{T_BUDGET}::test_sensitivity_resample_cell_composition",
            f"{T_BUDGET}::test_sensitivity_resamples_are_shared_and_seeded",
        ),
    ),
    _i(
        "§8-13",
        "불확정: 조건을 하한·상한에 적용해 다르면 불확정",
        (f"{EV}.oracle:useful", f"{EV}.oracle:classify_effect"),
        (
            f"{T_ORACLE}::test_useful_uses_the_plus_epsilon_boundary",
            f"{T_ORACLE}::test_classify_effect_three_way",
        ),
    ),
    _i(
        "§8-14",
        "칸 단위 조건은 점추정, 불확정 제외 없음",
        (f"{EV}.oracle:local_harm", f"{EV}.oracle:subset_label"),
        (
            f"{T_ORACLE}::test_local_harm_with_mixed_tau_cells",
            f"{T_ORACLE}::test_local_harm_boundary_is_strict",
        ),
    ),
    _i(
        "§8-15",
        "지표별 경계(Harmful -ε, Useful +ε, False Rejection +ε, 나머지 제외 없음)",
        (f"{EV}.eval_metrics:gate_metrics",),
        (
            f"{T_METRICS}::test_useful_precision_excludes_the_uncertain_candidate",
            f"{T_METRICS}::test_harmful_rate_excludes_candidates_uncertain_at_minus_epsilon",
        ),
    ),
    _i(
        "§8-16",
        "Expected Oracle Gain은 채택 수와 평균만",
        (f"{EV}.eval_metrics:gate_metrics",),
        (f"{T_METRICS}::test_expected_oracle_gain_is_the_mean_delta_of_accepted",),
    ),
    _i(
        "§8-17",
        "하위집합 불확정(세 갈래 분류에 양 끝 규칙), 통제 후보는 '불확정'",
        (f"{EV}.oracle:subset_label", f"{EV}.oracle:check_control_label"),
        (
            f"{T_ORACLE}::test_subset_uncertain_when_the_interval_crosses_epsilon",
            f"{T_ORACLE}::test_uncertain_control_is_reported_as_uncertain_not_unexpected",
        ),
    ),
    _i(
        "§8-18",
        "보조 지표: 불확정 제외, none/up/down 대응",
        (f"{EV}.eval_metrics:effect_agreement",),
        (f"{T_METRICS}::test_effect_agreement_excludes_uncertain_and_cards_without_effect",),
    ),
    _i(
        "§8-19",
        "후보 집합: proposer 12개와 통제 7개 따로",
        (f"{EV}.analysis:run_analysis",),
        (f"{T_ANALYSIS}::test_metrics_are_reported_per_group_and_never_pooled",),
    ),
    _i(
        "§8-20",
        "결과표 칸: 게이트 x metric마다 '채택 수 / 후보 수'",
        (f"{EV}.analysis:_results_table",),
        (f"{T_ANALYSIS}::test_results_table_rows_follow_7_8_e",),
    ),
    _i(
        "§8-21",
        "Clopper-Pearson 정확한 양측 95%, 분모 0이면 판정 불가",
        (f"{EV}.exact:clopper_pearson",),
        (
            f"{T_EXACT}::test_clopper_pearson_tails_are_two_and_a_half_percent",
            f"{T_EXACT}::test_clopper_pearson_with_zero_denominator_is_undecidable",
        ),
    ),
    _i(
        "§8-22",
        "τ_cell은 metric별·칸별",
        (f"{EV}.oracle:tau_cells",),
        (
            f"{T_ORACLE}::test_tau_is_per_metric_and_never_below_epsilon",
            f"{T_ANALYSIS}::test_fitness_and_tau_sections",
        ),
    ),
    _i(
        "§8-23",
        "F: metric마다 부호 또는 크기 기준, 둘 다 없으면 미달",
        (f"{EV}.oracle:fitness",),
        (
            f"{T_ORACLE}::test_fitness_fails_when_neither_criterion_is_met",
            f"{T_ORACLE}::test_fitness_sign_uses_each_cells_own_tau",
        ),
    ),
    _i(
        "§8-24",
        "RQ2·RQ3은 CD로 판정, ES는 보고만",
        (f"{EV}.analysis:_rq",),
        (f"{T_ANALYSIS}::test_rq_verdicts_are_present_for_cd_and_es",),
    ),
    _i(
        "§8-25",
        "RQ1 게이트별 비교와 종합",
        (f"{EV}.rq:rq1",),
        (
            f"{T_RQ}::test_rq1_partial_when_one_gate_holds",
            f"{T_RQ}::test_rq1_zero_denominator_rises_to_undecidable",
        ),
    ),
    _i(
        "§8-26",
        "RQ2 첫째 문장의 비교 4개와 둘째 문장",
        (f"{EV}.rq:rq2_first", f"{EV}.rq:rq2_second"),
        (
            f"{T_RQ}::test_rq2_first_partial",
            f"{T_RQ}::test_rq2_second_partly_grows_harmful_but_not_local_harm",
        ),
    ),
    _i(
        "§8-27",
        "RQ3 첫째 문장(N = 210)과 추세",
        (f"{EV}.rq:rq3_first", f"{EV}.rq:rq3_trend"),
        (
            f"{T_RQ}::test_rq3_first_supported_partial_not",
            f"{T_RQ}::test_rq3_trend_reports_monotonicity_separately",
        ),
    ),
    _i(
        "§8-28",
        "동률은 지지 안 됨, 분모 0은 그 비교 판정 불가",
        (f"{EV}.rq:rq1", f"{EV}.rq:rq2_first"),
        (
            f"{T_RQ}::test_rq1_tie_is_not_supported",
            f"{T_RQ}::test_rq2_first_tie_everywhere_is_not_supported",
        ),
    ),
    _i(
        "§8-29",
        "유리수 δ(N): 같은 알고리즘·같은 추출 순서",
        (f"{EV}.exact:null_diff_quantile_exact",),
        (
            f"{T_EXACT}::test_exact_delta_matches_draw_by_draw",
            f"{T_EXACT}::test_exact_delta_has_the_expected_denominator",
        ),
    ),
    _i(
        "§8-30",
        "RQ2 둘째 문장의 종합(커진다 / 일부 커진다 / 커지지 않는다)",
        (f"{EV}.rq:rq2_second",),
        (
            f"{T_RQ}::test_rq2_second_partly_grows_harmful_but_not_local_harm",
            f"{T_RQ}::test_rq2_second_undecidable_when_a_rate_is_undefined",
        ),
    ),
    _i(
        "§8-31",
        "RQ3 첫째 문장의 종합",
        (f"{EV}.rq:rq3_first",),
        (
            f"{T_RQ}::test_rq3_first_supported_partial_not",
            f"{T_RQ}::test_rq3_first_undecidable_when_c4_accepts_nothing",
        ),
    ),
    _i(
        "§8-32",
        "RQ3 추세의 종합(d_U, d_H), 비증가(같은 값 허용)는 따로",
        (f"{EV}.rq:rq3_trend",),
        (
            f"{T_RQ}::test_rq3_trend_categories",
            f"{T_RQ}::test_rq3_trend_reports_monotonicity_separately",
        ),
    ),
    _i(
        "§8-33",
        "한 문장 안에 판정 불가가 있으면 그 문장은 판정 불가, 판정 가능한 비교는 각각 보고. "
        "RQ2 둘째의 C2·C4 성질은 해석 주의",
        (f"{EV}.rq:rq2_first", f"{EV}.analysis:render_markdown"),
        (f"{T_RQ}::test_rq2_first_some_comparisons_undecidable", MD_TEST),
    ),
)

PREREG_78_IDS: tuple[str, ...] = tuple(
    f"7.8 {section}-{n}"
    for section, count in (("A", 8), ("B", 7), ("C", 3), ("D", 8), ("E", 3), ("F", 3), ("G", 8))
    for n in range(1, count + 1)
)
