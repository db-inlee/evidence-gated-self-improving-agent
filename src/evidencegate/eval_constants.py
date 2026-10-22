"""Ticket 005 분석 상수 (설계 문서 7.8절, 티켓 §8 해석). 임계값과 seed는 여기에만 둔다."""

from fractions import Fraction

SEED = 0  # 7.8: 모든 bootstrap·재표본의 seed(§8-1: 호출마다 새로 시작)
N_BOOT = 2000  # 7.8 A·B·D: δ(N), C4, Oracle 구간의 bootstrap 횟수
NOISE_Q = Fraction(19, 20)  # 7.8 A: δ(N)의 q = 0.95
CI_ALPHA = Fraction(1, 20)  # 7.8 D: Clopper-Pearson 95% 양측(§8-21)

GATE_REPEAT = 0  # 7.8 B: 게이트는 후보·S0 모두 반복 0(§8-5: 후보는 모든 split에서 반복 0)
S0_ORACLE_REPEATS = (0, 1, 2, 3, 4)  # 7.8 A: Oracle S0 반복 5회 다수결
C4_Q = Fraction(1, 20)  # 7.8 B: C4는 bootstrap 평균 Δ의 5번째 백분위수(단측 95% 하한)

EPSILON = Fraction(1, 20)  # 7.8 D: 실질 효과 임계값 ε = 5%p
ORACLE_CI_QS = (Fraction(1, 40), Fraction(39, 40))  # 7.8 D: Oracle Δ 95% 양측(2.5·97.5 백분위수)
TAU_Q = Fraction(19, 20)  # 7.8 D: 칸 잡음은 leave-one-out 다섯 값의 95번째 백분위수
FITNESS_SIZE = Fraction(3, 10)  # 7.8 F: 크기 기준, 같은 family 두 칸의 Δ_cell 차이 >= 30%p
BUDGET_NS = (24, 48, 96, 204)  # 7.8 C: Budget Curve의 N(칸당 4, 8, 16, 34)
N_RESAMPLE = 1000  # 7.8 C: Budget Curve와 RQ2 민감도의 재표본 수
# 7.8 C: RQ2 민감도 재표본의 칸 구성(CELLS 순서: bad L1·L2, dep L1·L2, traffic L1·L2)
SENSITIVITY_TAKE = (18, 18, 3, 3, 9, 9)
# §8-10: 재표본 분포의 2.5·97.5 백분위수(기술용)
BUDGET_SPREAD_QS = (Fraction(1, 40), Fraction(39, 40))
