"""Ticket 005 분석 상수 (설계 문서 7.8절, 티켓 §8 해석). 임계값과 seed는 여기에만 둔다."""

from fractions import Fraction

SEED = 0  # 7.8: 모든 bootstrap·재표본의 seed(§8-1: 호출마다 새로 시작)
N_BOOT = 2000  # 7.8 A·B·D: δ(N), C4, Oracle 구간의 bootstrap 횟수
NOISE_Q = Fraction(19, 20)  # 7.8 A: δ(N)의 q = 0.95
CI_ALPHA = Fraction(1, 20)  # 7.8 D: Clopper-Pearson 95% 양측(§8-21)

GATE_REPEAT = 0  # 7.8 B: 게이트는 후보·S0 모두 반복 0(§8-5: 후보는 모든 split에서 반복 0)
S0_ORACLE_REPEATS = (0, 1, 2, 3, 4)  # 7.8 A: Oracle S0 반복 5회 다수결
