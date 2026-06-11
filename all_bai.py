"""
╔══════════════════════════════════════════════════════════════════════╗
║  BÀI TẬP CUỐI KỲ – CÁC MÔ HÌNH RA QUYẾT ĐỊNH                       ║
║  Tổng hợp 12 bài – chạy tuần tự từ Bài 1 đến Bài 12                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Thư viện cần thiết:                                                 ║
║    numpy, pandas, scipy, matplotlib, seaborn,                        ║
║    pulp, cvxpy, pyomo, pymoo, gymnasium                              ║
║  Cài đặt:                                                            ║
║    pip install numpy pandas scipy matplotlib seaborn                 ║
║    pip install pulp cvxpy pyomo pymoo gymnasium                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────────
# IMPORTS CHUNG
# ──────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from io import StringIO

from scipy.optimize import linprog
from scipy.optimize import minimize as scipy_minimize

import pulp
from pulp import (
    LpProblem, LpMaximize, LpMinimize, LpVariable, LpStatus,
    lpSum, value, PULP_CBC_CMD
)

import cvxpy as cp

import pyomo.environ as pyo

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as moo_minimize
from pymoo.termination import get_termination

import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Chạy không cần GUI

print("✔  Đã nạp toàn bộ thư viện.")

# ══════════════════════════════════════════════════════════════════════
# BÀI 1 – Ước lượng TFP & Phân rã tăng trưởng GDP
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# BÀI TẬP CUỐI KÌ – CÁC MÔ HÌNH RA QUYẾT ĐỊNH
# Câu 1.4.1 → 1.4.4: Ước lượng TFP & Phân rã tăng trưởng GDP
# ============================================================

print("\n" + "█"*60)
print("  BÀI 1 – ƯỚC LƯỢNG TFP & PHÂN RÃ TĂNG TRƯỞNG GDP")
print("█"*60)

# ── Dữ liệu ─────────────────────────────────────────────────────────
data_1 = {
    'year':   [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
    'Y':      [5779.0,6014.0,6148.0,6342.0,6628.0,6881.0,7174.0,7540.0,7997.0,8393.0,8555.0,9095.0,9547.0,9932.0,10561.0],
    'K':      [11882.0,13154.0,14340.0,15590.0,16726.0,17944.0,19303.0,20745.0,22280.0,23701.0,24880.0,26115.0,27000.0,27500.0,28200.0],
    'L':      [50.39,51.39,52.35,53.25,53.75,53.98,54.35,54.82,55.00,55.04,54.84,52.97,53.66,54.48,55.12],
    'D':      [8.9,9.8,10.5,11.2,12.0,12.8,13.5,14.2,15.0,16.0,17.1,18.3,19.5,20.3,21.2],
    'AI_idx': [10.0,11.0,12.5,14.0,16.0,18.5,22.0,28.0,38.0,52.0,64.0,75.0,82.0,86.0,92.0],
    'H':      [20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,28.5,29.0,29.5,30.0,30.0,30.5],
}
df1 = pd.DataFrame(data_1)

alpha_1, beta_1, gamma_1, delta_1, theta_1 = 0.33, 0.42, 0.10, 0.08, 0.07

# ── Câu 1.4.1: Ước lượng TFP ────────────────────────────────────────
df1['ln_A'] = (np.log(df1['Y'])
               - alpha_1*np.log(df1['K'])
               - beta_1*np.log(df1['L'])
               - gamma_1*np.log(df1['D'])
               - delta_1*np.log(df1['AI_idx'])
               - theta_1*np.log(df1['H']))
df1['A'] = np.exp(df1['ln_A'])

print("\n[1.4.1] TFP theo năm:")
print(df1[['year','Y','K','L','A']].to_string(index=False))

# ── Câu 1.4.2: Phân rã tăng trưởng ─────────────────────────────────
df1['g_Y']  = df1['Y'].pct_change() * 100
df1['g_K']  = alpha_1 * df1['K'].pct_change() * 100
df1['g_L']  = beta_1  * df1['L'].pct_change() * 100
df1['g_D']  = gamma_1 * df1['D'].pct_change() * 100
df1['g_AI'] = delta_1 * df1['AI_idx'].pct_change() * 100
df1['g_H']  = theta_1 * df1['H'].pct_change() * 100
df1['g_TFP']= df1['g_Y'] - df1['g_K'] - df1['g_L'] - df1['g_D'] - df1['g_AI'] - df1['g_H']

print("\n[1.4.2] Phân rã tăng trưởng GDP (%):")
cols_show = ['year','g_Y','g_K','g_L','g_D','g_AI','g_H','g_TFP']
print(df1[cols_show].dropna().round(2).to_string(index=False))

# ── Câu 1.4.3: Dự báo 2025-2030 ────────────────────────────────────
growth_rates = {'K':0.055,'L':0.008,'D':0.06,'AI_idx':0.12,'H':0.02}
last = df1.iloc[-1].copy()
forecast_rows = []
for yr in range(2025, 2031):
    row = {'year': yr}
    for col, gr in growth_rates.items():
        last[col] = last[col] * (1 + gr)
        row[col] = last[col]
    ln_A_trend = np.polyfit(df1['year'], df1['ln_A'], 1)
    row['A'] = np.exp(np.polyval(ln_A_trend, yr))
    row['Y'] = (row['A'] * row['K']**alpha_1 * row['L']**beta_1
                * row['D']**gamma_1 * row['AI_idx']**delta_1 * row['H']**theta_1)
    forecast_rows.append(row)
df1_fc = pd.DataFrame(forecast_rows)

print("\n[1.4.3] Dự báo GDP 2025-2030:")
print(df1_fc[['year','K','L','D','AI_idx','H','A','Y']].round(1).to_string(index=False))

# ── Câu 1.4.4: Kịch bản chính sách ─────────────────────────────────
scenarios_1 = {
    'Baseline':        {'K':0.055,'L':0.008,'D':0.06,'AI_idx':0.12,'H':0.02},
    'Đẩy mạnh số hóa': {'K':0.055,'L':0.008,'D':0.10,'AI_idx':0.15,'H':0.03},
    'Tập trung K':     {'K':0.08, 'L':0.008,'D':0.06,'AI_idx':0.12,'H':0.02},
}
print("\n[1.4.4] GDP 2030 theo kịch bản:")
for sc_name, gr in scenarios_1.items():
    row_sc = df1.iloc[-1].copy()
    for _ in range(6):
        for col, r in gr.items():
            row_sc[col] *= (1 + r)
    Y_sc = (33.70 * row_sc['K']**alpha_1 * row_sc['L']**beta_1
            * row_sc['D']**gamma_1 * row_sc['AI_idx']**delta_1 * row_sc['H']**theta_1)
    print(f"  {sc_name:<22}: GDP 2030 ≈ {Y_sc:,.0f} nghìn tỷ VND")

# ── Biểu đồ ─────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(df1['year'], df1['A'], 'b-o', markersize=4)
axes[0].set_title('TFP theo năm'); axes[0].set_ylabel('A'); axes[0].grid(True, alpha=0.3)
df1_plot = df1.dropna(subset=['g_TFP'])
axes[1].bar(df1_plot['year'], df1_plot['g_TFP'], color='steelblue')
axes[1].set_title('Đóng góp TFP vào tăng trưởng (%)'); axes[1].grid(axis='y', alpha=0.3)
plt.suptitle('Bài 1: TFP Việt Nam', fontsize=14)
plt.tight_layout(); plt.savefig('bai1_tfp.png', dpi=120)
print("\n[Bài 1] Đã lưu biểu đồ: bai1_tfp.png")


# ══════════════════════════════════════════════════════════════════════
# BÀI 2 – Tối ưu phân bổ ngân sách AI quốc gia
# ══════════════════════════════════════════════════════════════════════
import os
OUTPUT_DIR = '.'
os.makedirs(OUTPUT_DIR, exist_ok=True) if OUTPUT_DIR not in (".", "") else None
# ============================================================
# BÀI TOÁN 2.4 - TỐI ƯU HÓA PHÂN BỔ NGÂN SÁCH AI QUỐC GIA
# Môn: Các Mô Hình Ra Quyết Định
# ============================================================

print("\n" + "█"*60)
print("  BÀI 2 – TỐI ƯU PHÂN BỔ NGÂN SÁCH AI QUỐC GIA")
print("█"*60)

# ── Dữ liệu bài 2 ───────────────────────────────────────────────────
sectors_2 = ['Hạ tầng số', 'AI R&D', 'Nhân lực số', 'An ninh mạng',
             'Chính phủ số', 'KT số địa phương']
n2 = len(sectors_2)
B_total_2 = 50000  # tỷ VND

# Hệ số hiệu quả (GDP tăng / tỷ VND đầu tư)
alpha_2 = np.array([0.85, 1.20, 0.95, 0.60, 0.75, 0.80])
# Hệ số tạo việc làm (nghìn việc / tỷ VND)
beta_2  = np.array([0.12, 0.08, 0.18, 0.05, 0.10, 0.15])
# Hệ số an ninh / bảo mật
gamma_2 = np.array([0.30, 0.50, 0.20, 0.90, 0.40, 0.25])

# Tỷ trọng mục tiêu
w_GDP_2, w_job_2, w_sec_2 = 0.50, 0.30, 0.20

# ── Câu 2.4.1: LP tối đa hóa lợi ích tổng hợp ──────────────────────
print("\n[2.4.1] LP tối đa hóa lợi ích tổng hợp")
c2 = -(w_GDP_2*alpha_2 + w_job_2*beta_2 + w_sec_2*gamma_2)
A_ub_2 = np.ones((1, n2))
b_ub_2 = np.array([B_total_2])
bounds_2 = [(2000, 15000)] * n2

res_2 = linprog(c2, A_ub=A_ub_2, b_ub=b_ub_2, bounds=bounds_2, method='highs')
x_opt_2 = res_2.x
Z_opt_2 = -res_2.fun

print(f"  Tổng lợi ích Z* = {Z_opt_2:.4f}")
print(f"  {'Ngành':<22} {'Phân bổ (tỷ)':>14} {'Lợi ích':>10}")
for i, s in enumerate(sectors_2):
    b_i = w_GDP_2*alpha_2[i] + w_job_2*beta_2[i] + w_sec_2*gamma_2[i]
    print(f"  {s:<22} {x_opt_2[i]:>14,.0f} {b_i*x_opt_2[i]:>10.2f}")
print(f"  {'Tổng':<22} {x_opt_2.sum():>14,.0f}")

# ── Câu 2.4.2: Phân tích độ nhạy (thay đổi ngân sách) ───────────────
print("\n[2.4.2] Phân tích độ nhạy theo ngân sách")
budgets_2 = np.linspace(30000, 80000, 11)
Z_vals_2  = []
for b_t in budgets_2:
    r = linprog(c2, A_ub=np.ones((1,n2)), b_ub=[b_t],
                bounds=[(2000,min(15000, b_t/n2+5000))]*n2, method='highs')
    Z_vals_2.append(-r.fun if r.success else np.nan)
print("  Ngân sách (tỷ)  →  Z*")
for bv, zv in zip(budgets_2, Z_vals_2):
    print(f"    {bv:>10,.0f}       {zv:.4f}")

# ── Câu 2.4.3: Ràng buộc cân bằng vùng ─────────────────────────────
print("\n[2.4.3] Thêm ràng buộc cân bằng vùng (min 15% cho KT số địa phương)")
A_eq_2 = np.zeros((1, n2)); A_eq_2[0, 5] = 1
b_eq_2 = np.array([B_total_2 * 0.15])
res_2b = linprog(c2, A_ub=A_ub_2, b_ub=b_ub_2,
                 A_eq=A_eq_2, b_eq=b_eq_2,
                 bounds=bounds_2, method='highs')
if res_2b.success:
    print(f"  Z* (có ràng buộc vùng) = {-res_2b.fun:.4f}")
    print(f"  KT số địa phương = {res_2b.x[5]:,.0f} tỷ (={res_2b.x[5]/B_total_2*100:.1f}%)")

# ── Câu 2.4.4: Kịch bản tăng tốc AI ────────────────────────────────
print("\n[2.4.4] Kịch bản tăng tốc AI (alpha_AI tăng 50%)")
alpha_2b = alpha_2.copy(); alpha_2b[1] *= 1.50
c2b = -(w_GDP_2*alpha_2b + w_job_2*beta_2 + w_sec_2*gamma_2)
res_2c = linprog(c2b, A_ub=A_ub_2, b_ub=b_ub_2, bounds=bounds_2, method='highs')
if res_2c.success:
    print(f"  Z* (kịch bản tăng AI) = {-res_2c.fun:.4f}")
    print(f"  Phân bổ AI R&D: {res_2c.x[1]:,.0f} tỷ (vs {x_opt_2[1]:,.0f} tỷ trước)")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
axes2[0].bar(sectors_2, x_opt_2/1000, color='steelblue')
axes2[0].set_title('Phân bổ ngân sách tối ưu (nghìn tỷ)')
axes2[0].tick_params(axis='x', rotation=30); axes2[0].grid(axis='y', alpha=0.3)
axes2[1].plot(budgets_2/1000, Z_vals_2, 'b-o', markersize=5)
axes2[1].set_title('Phân tích độ nhạy Z* vs Ngân sách')
axes2[1].set_xlabel('Ngân sách (nghìn tỷ VND)'); axes2[1].grid(True, alpha=0.3)
plt.suptitle('Bài 2: Phân bổ ngân sách AI', fontsize=14)
plt.tight_layout(); plt.savefig('bai2_budget.png', dpi=120)
print("\n[Bài 2] Đã lưu biểu đồ: bai2_budget.png")


# ══════════════════════════════════════════════════════════════════════
# BÀI 3 – TOPSIS + Entropy: Ưu tiên đầu tư ngành kinh tế VN 2024
# ══════════════════════════════════════════════════════════════════════
# ================================================================
#  BÀI TOÁN 3.4 – Phân tích ưu tiên đầu tư ngành kinh tế VN 2024
# ================================================================

print("\n" + "█"*60)
print("  BÀI 3 – ƯU TIÊN ĐẦU TƯ NGÀNH KINH TẾ VN 2024 (TOPSIS/Entropy)")
print("█"*60)

RAW_3 = """sector,gdp_contrib,growth_rate,employ_mill,fdi_share,export_share,digit_ready,env_score
Nông-lâm-thủy sản,11.96,3.28,14.30,2.1,12.5,35,55
Công nghiệp chế biến,24.02,8.50,11.50,42.5,68.3,62,40
Xây dựng,5.87,7.12,4.80,3.2,1.2,45,35
Thương mại bán lẻ,10.45,9.23,7.80,5.8,8.5,72,60
Tài chính ngân hàng,4.82,10.50,0.55,8.5,2.3,85,65
Logistics vận tải,4.15,7.80,1.95,4.2,5.8,58,42
CNTT-Truyền thông,2.75,13.20,0.62,12.3,15.2,92,70
Giáo dục đào tạo,3.12,6.50,2.15,1.8,0.8,68,75
Du lịch dịch vụ,4.20,15.30,4.25,6.5,9.2,75,62
Y tế chăm sóc,2.45,8.90,0.98,2.1,1.5,55,80"""

df3 = pd.read_csv(StringIO(RAW_3))
criteria_3 = ['gdp_contrib','growth_rate','employ_mill','fdi_share',
              'export_share','digit_ready','env_score']
is_ben_3 = [True, True, True, True, True, True, True]

X3 = df3[criteria_3].values.astype(float)
w_exp_3 = np.array([0.15, 0.20, 0.15, 0.15, 0.10, 0.15, 0.10])

def topsis_3(X, w, is_ben):
    R = X / np.sqrt((X**2).sum(0))
    V = R * w
    A_s = np.where(is_ben, V.max(0), V.min(0))
    A_n = np.where(is_ben, V.min(0), V.max(0))
    S_s = np.sqrt(((V - A_s)**2).sum(1))
    S_n = np.sqrt(((V - A_n)**2).sum(1))
    return S_n / (S_s + S_n)

def entropy_w_3(X):
    P = X / X.sum(0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), 0)
    d = 1 - E
    return d / d.sum()

C_exp_3 = topsis_3(X3, w_exp_3, is_ben_3)
w_ent_3  = entropy_w_3(X3)
C_ent_3  = topsis_3(X3, w_ent_3, is_ben_3)

print("\n[3.4.1] Xếp hạng TOPSIS (trọng số chuyên gia vs Entropy):")
print(f"  {'Ngành':<25} {'C*_expert':>10} {'Rank_E':>6} {'C*_entropy':>10} {'Rank_En':>7}")
rank_exp_3 = np.argsort(-C_exp_3)
rank_ent_3 = np.argsort(-C_ent_3)
for i in range(len(df3)):
    r_e  = list(rank_exp_3).index(i) + 1
    r_en = list(rank_ent_3).index(i) + 1
    print(f"  {df3['sector'][i]:<25} {C_exp_3[i]:>10.4f} {r_e:>6} {C_ent_3[i]:>10.4f} {r_en:>7}")

print(f"\n  Trọng số Entropy: {dict(zip([c[:8] for c in criteria_3], np.round(w_ent_3,4)))}")

# Phân tích độ nhạy: thay đổi trọng số growth_rate
print("\n[3.4.2] Phân tích độ nhạy khi tăng trọng số tăng trưởng:")
for w_gr in [0.10, 0.20, 0.30, 0.40]:
    w_sens = w_exp_3.copy()
    w_sens[1] = w_gr
    w_sens = w_sens / w_sens.sum()
    C_sens = topsis_3(X3, w_sens, is_ben_3)
    top3 = [df3['sector'][r] for r in np.argsort(-C_sens)[:3]]
    print(f"  w_growth={w_gr:.2f} → Top 3: {top3}")

fig3, ax3 = plt.subplots(figsize=(10, 5))
idx_sort = np.argsort(-C_exp_3)
ax3.barh([df3['sector'][i] for i in idx_sort], C_exp_3[idx_sort], color='steelblue')
ax3.set_xlabel('TOPSIS Score'); ax3.set_title('Bài 3: Ưu tiên đầu tư ngành KT VN 2024')
ax3.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig('bai3_topsis.png', dpi=120)
print("\n[Bài 3] Đã lưu biểu đồ: bai3_topsis.png")


# ══════════════════════════════════════════════════════════════════════
# BÀI 4 – Tối ưu phân bổ ngân sách số hóa Việt Nam
# ══════════════════════════════════════════════════════════════════════
# ============================================================
#  BÀI 4.4 – TỐI ƯU PHÂN BỔ NGÂN SÁCH SỐ HOÁ VIỆT NAM
# ============================================================

print("\n" + "█"*60)
print("  BÀI 4 – TỐI ƯU PHÂN BỔ NGÂN SÁCH SỐ HOÁ VIỆT NAM")
print("█"*60)

import pulp as _pulp4

sectors_4 = ['Hạ tầng số','AI R&D','Nhân lực số','An ninh mạng',
             'Chính phủ số','KT số địa phương']
n4 = len(sectors_4)
B4 = 50000

# Tham số
alpha_4 = np.array([0.85, 1.20, 0.95, 0.60, 0.75, 0.80])
beta_4  = np.array([0.12, 0.08, 0.18, 0.05, 0.10, 0.15])
gamma_4 = np.array([0.30, 0.50, 0.20, 0.90, 0.40, 0.25])
w4 = np.array([0.50, 0.30, 0.20])

# ── Câu 4.4.1: LP ────────────────────────────────────────────────────
print("\n[4.4.1] LP tối đa hóa lợi ích tổng hợp (PuLP)")
m4_1 = _pulp4.LpProblem("Budget_Digital_4", _pulp4.LpMaximize)
x4 = _pulp4.LpVariable.dicts("x", range(n4), lowBound=2000, upBound=15000)
obj_coeff_4 = w4[0]*alpha_4 + w4[1]*beta_4 + w4[2]*gamma_4
m4_1 += _pulp4.lpSum(obj_coeff_4[i] * x4[i] for i in range(n4))
m4_1 += _pulp4.lpSum(x4[i] for i in range(n4)) <= B4
m4_1.solve(_pulp4.PULP_CBC_CMD(msg=False))
print(f"  Z* = {_pulp4.value(m4_1.objective):.4f}")
for i, s in enumerate(sectors_4):
    print(f"  {s:<22}: {_pulp4.value(x4[i]):>10,.0f} tỷ")

# ── Câu 4.4.2: CVXPY – Hàm mục tiêu phi tuyến ───────────────────────
print("\n[4.4.2] CVXPY – mục tiêu phi tuyến (log utility)")
x4_cp = cp.Variable(n4, nonneg=True)
log_obj = cp.sum(cp.multiply(obj_coeff_4, cp.log(x4_cp + 1)))
prob4_2 = cp.Problem(cp.Maximize(log_obj),
                     [cp.sum(x4_cp) <= B4,
                      x4_cp >= 2000, x4_cp <= 15000])
prob4_2.solve(solver=cp.CLARABEL, verbose=False)
if prob4_2.status in ['optimal', 'optimal_inaccurate']:
    print(f"  Z* (log) = {prob4_2.value:.4f}")
    for i, s in enumerate(sectors_4):
        print(f"  {s:<22}: {x4_cp.value[i]:>10,.0f} tỷ")
else:
    print(f"  Solver status: {prob4_2.status}")

# ── Câu 4.4.3: Ràng buộc công bằng vùng ────────────────────────────
print("\n[4.4.3] Thêm ràng buộc KT số địa phương >= 20% ngân sách")
m4_3 = _pulp4.LpProblem("Budget_Digital_4b", _pulp4.LpMaximize)
x4b = _pulp4.LpVariable.dicts("x", range(n4), lowBound=2000, upBound=15000)
m4_3 += _pulp4.lpSum(obj_coeff_4[i] * x4b[i] for i in range(n4))
m4_3 += _pulp4.lpSum(x4b[i] for i in range(n4)) <= B4
m4_3 += x4b[5] >= 0.20 * B4
m4_3.solve(_pulp4.PULP_CBC_CMD(msg=False))
print(f"  Z* (có ràng buộc vùng) = {_pulp4.value(m4_3.objective):.4f}")
print(f"  KT số địa phương = {_pulp4.value(x4b[5]):,.0f} tỷ")

# ── Câu 4.4.4: Phân tích shadow price ───────────────────────────────
print("\n[4.4.4] Shadow price ngân sách")
for b_test in [45000, 50000, 55000, 60000]:
    r_sp = linprog(-(w4[0]*alpha_4 + w4[1]*beta_4 + w4[2]*gamma_4),
                   A_ub=np.ones((1,n4)), b_ub=[b_test],
                   bounds=[(2000,15000)]*n4, method='highs')
    print(f"  B={b_test:>6,} tỷ → Z* = {-r_sp.fun:.4f}")

print("\n[Bài 4] Hoàn thành.")


# ══════════════════════════════════════════════════════════════════════
# BÀI 5 – Lựa chọn dự án đầu tư công VN (ILP)
# ══════════════════════════════════════════════════════════════════════
"""
╔══════════════════════════════════════════════════════════════╗
║  BÀI TOÁN LỰA CHỌN DỰ ÁN ĐẦU TƯ CÔNG VIỆT NAM             ║
║  5.4.1 – Bài toán gốc (80k / 40k tỷ VND)                    ║
║  5.4.2 – Nới ngân sách GĐ1 lên 100.000 tỷ                   ║
║  5.4.3 – Bắt buộc P1 VÀ P2 (redundancy Quốc hội)            ║
║  5.4.4 – Mở rộng rủi ro: Tối đa E[Z] = Σ pᵢ·Bᵢ·yᵢ         ║
╚══════════════════════════════════════════════════════════════╝
"""

print("\n" + "█"*64)
print("  BÀI 5 – LỰA CHỌN DỰ ÁN ĐẦU TƯ CÔNG VIỆT NAM (ILP)")
print("█"*64)

P5 = list(range(1, 16))

C5 = {
    1: 12000, 2: 11500, 3: 18000, 4:  4500,  5: 3200,
    6:  5800, 7:  6500, 8: 15000, 9:  2500, 10: 7200,
   11:  4800,12:  8500,13: 20000,14:  3800, 15: 1500
}
C5_2 = {
    1:  8500, 2:  7500, 3: 12000, 4:  3500,  5: 2500,
    6:  4000, 7:  4500, 8:  9000, 9:  1800, 10: 5000,
   11:  3500,12:  5500,13: 13000,14:  2800, 15: 1200
}
B5 = {
    1: 21500, 2: 20800, 3: 32500, 4:  9200,  5: 6800,
    6: 11400, 7: 12200, 8: 28500, 9:  5800, 10:13800,
   11:  8500,12: 16200,13: 35000,14:  7500, 15: 3800
}

_INFRA5 = {3, 8, 14}; _EGOV5 = {1, 2, 12}; _AI5 = {7, 10, 13}
p_risk5 = {
    i: 0.85 if i in _INFRA5 else
       0.75 if i in _EGOV5  else
       0.65 if i in _AI5    else 0.80
    for i in P5
}

def build_model5(budget1=80_000, budget2=40_000,
                 force_p1_and_p2=False, use_expected=False):
    m = LpProblem("VN_Project_Selection", LpMaximize)
    y = LpVariable.dicts("y", P5, cat="Binary")
    if use_expected:
        m += lpSum(p_risk5[i] * B5[i] * y[i] for i in P5), "Max_E_Z"
    else:
        m += lpSum(B5[i] * y[i] for i in P5), "Max_Z"
    m += lpSum(C5[i]  * y[i] for i in P5) <= budget1, "Budget_GD1"
    m += lpSum(C5_2[i] * y[i] for i in P5) <= budget2, "Budget_GD2"
    if force_p1_and_p2:
        m += y[1] == 1, "Force_P1"
        m += y[2] == 1, "Force_P2"
    else:
        m += y[1] + y[2] <= 1, "MutualExcl_P1_P2"
    m += y[8]  <= y[12], "Depend_P8_req_P12"
    m += y[13] <= y[12], "Depend_P13_req_P12"
    m += y[4] + y[5] >= 1, "Region_P4orP5"
    m += y[14] >= 1,        "Mandatory_P14"
    m += lpSum(y[i] for i in P5) >= 7,  "Min_7_Projects"
    m += lpSum(y[i] for i in P5) <= 11, "Max_11_Projects"
    return m, y

SEP5  = "═" * 64
SEP5b = "─" * 64

def solve_and_report5(m, y, title, use_expected=False):
    print(f"\n{SEP5}\n  {title}\n{SEP5}")
    m.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[m.status]
    print(f"  Trạng thái: {status}")
    if status != "Optimal":
        print("  ⚠  KHÔNG TÌM ĐƯỢC LỜI GIẢI TỐI ƯU!")
        if status == "Infeasible":
            print(f"     Chi phí P1+P2: GĐ1={C5[1]+C5[2]:,} | GĐ2={C5_2[1]+C5_2[2]:,} tỷ")
        return None
    selected = [i for i in P5 if y[i].value() > 0.5]
    tc1 = sum(C5[i]  for i in selected)
    tc2 = sum(C5_2[i] for i in selected)
    tb  = sum(B5[i]  for i in selected)
    z   = value(m.objective)
    print(f"  Dự án: {[f'P{i}' for i in selected]}  (n={len(selected)})")
    print(f"  GĐ1={tc1:,.0f}  GĐ2={tc2:,.0f}  B_total={tb:,.0f}  Z*={z:,.0f}")
    return z, selected

m5_1, y5_1 = build_model5(80_000, 40_000)
r5_1 = solve_and_report5(m5_1, y5_1, "5.4.1 – Bài toán gốc [GĐ1:80k | GĐ2:40k]")

m5_2, y5_2 = build_model5(100_000, 40_000)
r5_2 = solve_and_report5(m5_2, y5_2, "5.4.2 – Nới ngân sách [GĐ1:100k | GĐ2:40k]")

if r5_1 and r5_2:
    added5 = sorted(set(r5_2[1]) - set(r5_1[1]))
    dz5 = r5_2[0] - r5_1[0]
    print(f"\n  ▶ Thêm: {[f'P{i}' for i in added5]} | ΔZ={dz5:+,.0f} tỷ")

m5_3, y5_3 = build_model5(80_000, 40_000, force_p1_and_p2=True)
r5_3 = solve_and_report5(m5_3, y5_3, "5.4.3 – Bắt buộc P1+P2 [GĐ1:80k | GĐ2:40k]")

m5_4, y5_4 = build_model5(80_000, 40_000, use_expected=True)
r5_4 = solve_and_report5(m5_4, y5_4, "5.4.4 – E[Z] rủi ro [GĐ1:80k | GĐ2:40k]", use_expected=True)

print(f"\n{SEP5}\n  TỔNG KẾT BÀI 5\n{SEP5}")
for lbl, res5 in [("5.4.1 Gốc", r5_1),("5.4.2 Nới NS",r5_2),
                  ("5.4.3 P1+P2",r5_3),("5.4.4 E[Z]",r5_4)]:
    if res5:
        print(f"  {lbl:<18}: Z*={res5[0]:>14,.0f}  n={len(res5[1])}")
    else:
        print(f"  {lbl:<18}: INFEASIBLE")


# ══════════════════════════════════════════════════════════════════════
# BÀI 6 – TOPSIS + Entropy + Sensitivity + AHP
# ══════════════════════════════════════════════════════════════════════
"""
=============================================================================
  Câu 6.4: TOPSIS + Entropy Weights + Sensitivity Analysis + AHP
=============================================================================
"""

print("\n" + "█"*60)
print("  BÀI 6 – TOPSIS + ENTROPY + AHP (PHÂN TÍCH ĐA TIÊU CHÍ)")
print("█"*60)

# ── Dữ liệu vùng kinh tế ────────────────────────────────────────────
RAW_6 = """region_id,region_name,grdp_per_capita,fdi_bn_usd,digital_index,ai_readiness,trained_labor_pct,rd_intensity,internet_pct,gini
1,NMM,57.0,3.5,38,22,21.5,0.18,72,0.405
2,RRD,152.3,20.0,78,68,36.8,0.85,92,0.358
3,NSCC,87.5,8.2,55,40,27.5,0.32,84,0.372
4,CH,68.9,0.8,32,18,18.2,0.15,68,0.412
5,SE,158.9,18.5,82,75,42.5,0.78,94,0.385
6,MD,80.5,2.1,48,30,16.8,0.22,78,0.392"""
df6 = pd.read_csv(StringIO(RAW_6))

crit_6 = ['grdp_per_capita','fdi_bn_usd','digital_index','ai_readiness',
          'trained_labor_pct','rd_intensity','internet_pct','gini']
ib_6   = [True, True, True, True, True, True, True, False]
w6_exp = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

X6 = df6[crit_6].values.astype(float)

def topsis_6(X, w, is_ben):
    R = X / np.sqrt((X**2).sum(0))
    V = R * w
    A_s = np.where(is_ben, V.max(0), V.min(0))
    A_n = np.where(is_ben, V.min(0), V.max(0))
    S_s = np.sqrt(((V - A_s)**2).sum(1))
    S_n = np.sqrt(((V - A_n)**2).sum(1))
    return S_n / (S_s + S_n)

def ew_6(X):
    P = X / X.sum(0); k = 1/np.log(len(X))
    E = -k*np.nansum(P*np.log(P+1e-12),0); d=1-E
    return d/d.sum()

w6_ent = ew_6(X6)
C6_exp = topsis_6(X6, w6_exp, ib_6)
C6_ent = topsis_6(X6, w6_ent, ib_6)

print("\n[6.4.1] TOPSIS – Xếp hạng vùng kinh tế:")
print(f"  {'Vùng':<8} {'C*_expert':>10} {'Rank_E':>7} {'C*_entropy':>11} {'Rank_En':>8}")
for i, rn in enumerate(df6['region_name']):
    re  = sorted(range(len(C6_exp)), key=lambda j: -C6_exp[j]).index(i)+1
    ren = sorted(range(len(C6_ent)), key=lambda j: -C6_ent[j]).index(i)+1
    print(f"  {rn:<8} {C6_exp[i]:>10.4f} {re:>7} {C6_ent[i]:>11.4f} {ren:>8}")

# AHP – ma trận so sánh đơn giản (chỉ cho 3 tiêu chí đại diện)
print("\n[6.4.2] AHP – kiểm tra tính nhất quán (CR):")
AHP_mat = np.array([
    [1,    3,    5   ],
    [1/3,  1,    3   ],
    [1/5,  1/3,  1   ],
])
eigvals = np.linalg.eigvals(AHP_mat)
lambda_max = np.max(eigvals.real)
n_ahp = AHP_mat.shape[0]
CI = (lambda_max - n_ahp) / (n_ahp - 1)
RI = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12}.get(n_ahp, 1.12)
CR = CI / RI
print(f"  λ_max={lambda_max:.4f}  CI={CI:.4f}  CR={CR:.4f}  {'✔ Nhất quán' if CR < 0.10 else '✗ Không nhất quán'}")

# Sensitivity: thay đổi w_ai_readiness
print("\n[6.4.3] Phân tích độ nhạy (ai_readiness weight 0.10→0.40):")
for w_ai in [0.10, 0.20, 0.30, 0.40]:
    w_s = w6_exp.copy(); w_s[3] = w_ai; w_s = w_s/w_s.sum()
    C_s = topsis_6(X6, w_s, ib_6)
    top2 = [df6['region_name'][r] for r in np.argsort(-C_s)[:2]]
    print(f"  w_AI={w_ai:.2f} → Top 2: {top2}")

print("\n[Bài 6] Hoàn thành.")


# ══════════════════════════════════════════════════════════════════════
# BÀI 7 – NSGA-II + TOPSIS: Tối ưu đa mục tiêu đầu tư số hóa VN
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# BÀI 7.4 - TỐI ƯU HÓA ĐẦU TƯ SỐ HÓA VIỆT NAM (NSGA-II + TOPSIS)
# ============================================================

print("\n" + "█"*60)
print("  BÀI 7 – NSGA-II + TOPSIS (TỐI ƯU ĐA MỤC TIÊU)")
print("█"*60)

sectors_7 = ['Hạ tầng số','AI R&D','Nhân lực số','An ninh mạng',
             'Chính phủ số','KT số địa phương']
n7 = len(sectors_7)
B7 = 50000

alpha_7 = np.array([0.85, 1.20, 0.95, 0.60, 0.75, 0.80])
beta_7  = np.array([0.12, 0.08, 0.18, 0.05, 0.10, 0.15])
gamma_7 = np.array([0.30, 0.50, 0.20, 0.90, 0.40, 0.25])

class DigitInvestProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=n7, n_obj=3, n_ieq_constr=1,
                         xl=np.full(n7, 2000.0),
                         xu=np.full(n7, 15000.0))
    def _evaluate(self, x, out, *args, **kwargs):
        f1 = -np.dot(alpha_7, x)          # GDP (max → min)
        f2 = -np.dot(beta_7,  x)          # Việc làm
        f3 = -np.dot(gamma_7, x)          # An ninh
        g1 = x.sum() - B7                 # tổng <= B7
        out["F"] = [f1, f2, f3]
        out["G"] = [g1]

print("\n[7.4.1] Chạy NSGA-II (200 thế hệ × 100 cá thể)...")
prob7 = DigitInvestProblem()
algo7 = NSGA2(pop_size=100)
res7  = moo_minimize(prob7, algo7,
                     get_termination("n_gen", 200),
                     seed=42, verbose=False)
F7 = res7.F
print(f"  Số nghiệm Pareto: {len(F7)}")
print(f"  f1 (−GDP) range:  [{F7[:,0].min():.0f}, {F7[:,0].max():.0f}]")
print(f"  f2 (−Job) range:  [{F7[:,1].min():.0f}, {F7[:,1].max():.0f}]")
print(f"  f3 (−Sec) range:  [{F7[:,2].min():.0f}, {F7[:,2].max():.0f}]")

# TOPSIS chọn nghiệm tốt nhất
print("\n[7.4.2] TOPSIS chọn nghiệm tốt nhất từ mặt Pareto:")
F7_pos = -F7  # chuyển về tối đa
w7_topsis = np.array([0.50, 0.30, 0.20])
ib7 = [True, True, True]
C7 = topsis_3(F7_pos, w7_topsis, ib7)
best_idx7 = np.argmax(C7)
x_best7 = res7.X[best_idx7]
print(f"  Nghiệm tốt nhất (TOPSIS C*={C7[best_idx7]:.4f}):")
for i, s in enumerate(sectors_7):
    print(f"    {s:<22}: {x_best7[i]:>10,.0f} tỷ")
print(f"  GDP score: {np.dot(alpha_7, x_best7):,.0f}")
print(f"  Job score: {np.dot(beta_7,  x_best7):,.2f}")
print(f"  Sec score: {np.dot(gamma_7, x_best7):,.2f}")

fig7, ax7 = plt.subplots(figsize=(8, 5))
sc = ax7.scatter(-F7[:,0], -F7[:,1], c=-F7[:,2], cmap='viridis', s=20, alpha=0.7)
plt.colorbar(sc, ax=ax7, label='An ninh score')
ax7.scatter(-F7[best_idx7,0], -F7[best_idx7,1], c='red', s=100, zorder=5, label='TOPSIS best')
ax7.set_xlabel('GDP score'); ax7.set_ylabel('Job score')
ax7.set_title('Bài 7: Mặt Pareto (NSGA-II)')
ax7.legend(); ax7.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('bai7_pareto.png', dpi=120)
print("\n[Bài 7] Đã lưu: bai7_pareto.png")


# ══════════════════════════════════════════════════════════════════════
# BÀI 8 – Tối ưu động phân bổ liên thời gian 2026-2035
# ══════════════════════════════════════════════════════════════════════
import numpy as np
from scipy.optimize import minimize as scipy_minimize

print("\n" + "█"*60)
print("  BÀI 8 – TỐI ƯU ĐỘNG 2026-2035")
print("█"*60)

alpha_8, beta_cd_8 = 0.33, 0.42
gamma_d_8, delta_ai_8, theta_h_8 = 0.10, 0.08, 0.07
delta_K_8, delta_D_8, delta_AI_8 = 0.05, 0.12, 0.15
theta_H_eff_8, mu_8 = 0.8, 0.02
phi1_8, phi2_8, phi3_8 = 0.003, 0.002, 0.004
rho_disc_8, gamma_cr_8 = 0.97, 1.5
T8 = 10

K0_8, L0_8 = 27500.0, 53.9
D0_8, AI0_8, H0_8 = 20.3, 86.0, 30.0
Y0_8 = 12847.6
A0_8 = Y0_8 / (K0_8**alpha_8 * L0_8**beta_cd_8 * D0_8**gamma_d_8
                * AI0_8**delta_ai_8 * H0_8**theta_h_8)
L8 = np.array([L0_8 * 1.009**t for t in range(T8+1)])

def compute_trajectory8(u, shock_year=None, shock_pct=0.0):
    IK = u[0::4]; ID = u[1::4]; IAI = u[2::4]; IH = u[3::4]
    K = np.zeros(T8+1); D = np.zeros(T8+1)
    AI = np.zeros(T8+1); H = np.zeros(T8+1)
    A = np.zeros(T8+1); Y = np.zeros(T8+1); C = np.zeros(T8)
    K[0]=K0_8; D[0]=D0_8; AI[0]=AI0_8; H[0]=H0_8; A[0]=A0_8
    for t in range(T8):
        if shock_year is not None and t == shock_year:
            A[t] *= (1 - shock_pct)
        Y[t] = A[t]*K[t]**alpha_8*L8[t]**beta_cd_8*D[t]**gamma_d_8*AI[t]**delta_ai_8*H[t]**theta_h_8
        C[t] = Y[t] - IK[t] - ID[t] - IAI[t] - IH[t]
        if C[t] <= 0: return None
        K[t+1] = (1-delta_K_8)*K[t] + IK[t]
        D[t+1] = (1-delta_D_8)*D[t] + ID[t]
        AI[t+1] = (1-delta_AI_8)*AI[t] + IAI[t]
        H[t+1] = H[t] + theta_H_eff_8*IH[t] - mu_8*H[t]
        A[t+1] = A[t]*(1 + phi1_8*(D[t]/100) + phi2_8*(AI[t]/100) + phi3_8*(H[t]/100))
    Y[T8] = A[T8]*K[T8]**alpha_8*L8[T8]**beta_cd_8*D[T8]**gamma_d_8*AI[T8]**delta_ai_8*H[T8]**theta_h_8
    return K, D, AI, H, Y, C, A

def welfare8(u, shock_year=None, shock_pct=0.0):
    result = compute_trajectory8(u, shock_year, shock_pct)
    if result is None: return 1e15
    K,D,AI,H,Y,C,A = result
    if np.any(C <= 0): return 1e15
    W = sum(rho_disc_8**t * (C[t]**(1-gamma_cr_8)-1)/(1-gamma_cr_8) for t in range(T8))
    return -W

Y_est8 = 14000; invest_frac8 = 0.15
total_invest8 = Y_est8 * invest_frac8
u0_8 = np.zeros(T8*4)
for t in range(T8):
    u0_8[t*4+0] = total_invest8*0.40; u0_8[t*4+1] = total_invest8*0.25
    u0_8[t*4+2] = total_invest8*0.20; u0_8[t*4+3] = total_invest8*0.15
bounds8 = [(0, None)]*(T8*4)
constr8 = [{'type':'ineq','fun': lambda u: min(compute_trajectory8(u)[5])-1
            if compute_trajectory8(u) else -1e10}]

print("\n[8.3.1] Tối ưu hóa quỹ đạo cơ sở...")
res8 = scipy_minimize(welfare8, u0_8, method='SLSQP', bounds=bounds8,
                      constraints=constr8, options={'maxiter':500,'ftol':1e-7,'disp':False})
traj8 = compute_trajectory8(res8.x)
K8,D8,AI8,H8,Y8,C8,A8 = traj8
print(f"  W* = {-res8.fun:.4f}")
print(f"  GDP 2026→2035: {Y8[0]:,.0f} → {Y8[T8]:,.0f} nghìn tỷ VND")

print("\n[8.3.3] Sốc TFP năm 2028 (giảm 8%):")
SHOCK_T8, SHOCK_PCT8 = 2, 0.08
_,_,_,_,Y8_sh,C8_sh,_ = compute_trajectory8(res8.x, SHOCK_T8, SHOCK_PCT8)
W_base8 = -welfare8(res8.x)
W_shock8 = -welfare8(res8.x, SHOCK_T8, SHOCK_PCT8)
print(f"  W_base={W_base8:.4f}  W_shock={W_shock8:.4f}  "
      f"Mất={W_base8-W_shock8:.4f} ({(W_base8-W_shock8)/W_base8*100:.4f}%)")

res8_sh = scipy_minimize(lambda u: welfare8(u, SHOCK_T8, SHOCK_PCT8),
                         res8.x, method='SLSQP', bounds=bounds8,
                         constraints=constr8,
                         options={'maxiter':500,'ftol':1e-7,'disp':False})
W_reopt8 = -res8_sh.fun
print(f"  W_reopt={W_reopt8:.4f}  Phục hồi={W_reopt8-W_shock8:.4f}")

print("\n[8.3.4] So sánh chiến lược đầu tư:")
u_even8 = u0_8.copy()
u_front8 = np.zeros(T8*4)
for t in range(T8):
    f8 = 1.5 if t < 3 else 0.7
    u_front8[t*4:t*4+4] = [total_invest8*r*f8 for r in [0.40,0.25,0.20,0.15]]
for name8, u8 in [('Tối ưu', res8.x),('Đầu tư đều', u_even8),('Front-load', u_front8)]:
    W_c = -welfare8(u8)
    t8_end = compute_trajectory8(u8)
    Y_end = t8_end[4][T8] if t8_end else 0
    print(f"  {name8:<15}: W={W_c:.4f}  GDP_2035={Y_end:,.0f}")


# ══════════════════════════════════════════════════════════════════════
# BÀI 9 – Tác động AI tới thị trường lao động VN
# ══════════════════════════════════════════════════════════════════════
# ============================================================
# BAI 9: Tac dong AI toi thi truong lao dong VN
# ============================================================

print("\n" + "█"*60)
print("  BÀI 9 – TÁC ĐỘNG AI TỚI THỊ TRƯỜNG LAO ĐỘNG VN")
print("█"*60)

N9 = 8
sectors_9 = ['Nông-LT','CN chế biến','Xây dựng','Bán buôn-bán lẻ',
             'Tài chính-NH','Logistics','CNTT-TT','Giáo dục-ĐT']

L9 = np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])
risk9 = np.array([18,42,25,38,52,35,28,22])/100

a1_9 = np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
b1_9 = np.array([45,28,35,32,22,30,20,55])
c1_9 = np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
d1_9 = np.array([50,32,42,38,26,36,24,62])

coeff_AI9 = a1_9 - c1_9 * risk9
coeff_H9  = b1_9.copy()

c_obj9 = np.concatenate([-coeff_AI9, -coeff_H9])
A1_9 = np.concatenate([np.ones(N9), np.ones(N9)]).reshape(1,-1)
A1b_9 = np.concatenate([-np.ones(N9), np.zeros(N9)]).reshape(1,-1)
A2_9 = np.zeros((N9, 2*N9))
for i in range(N9): A2_9[i,i]=-coeff_AI9[i]; A2_9[i,N9+i]=-coeff_H9[i]
A3_9 = np.zeros((N9, 2*N9))
for i in range(N9): A3_9[i,i]=c1_9[i]*risk9[i]; A3_9[i,N9+i]=-d1_9[i]

A_ub9 = np.vstack([A1_9, A1b_9, A2_9, A3_9])
b_ub9 = np.concatenate([[30000], [-9000], np.zeros(N9), np.zeros(N9)])
res9  = linprog(c_obj9, A_ub=A_ub9, b_ub=b_ub9,
                bounds=[(0,None)]*(2*N9), method='highs')

x_AI9, x_H9 = res9.x[:N9], res9.x[N9:]
NetJob9 = coeff_AI9*x_AI9 + coeff_H9*x_H9
NewJob9 = a1_9*x_AI9
Displaced9 = c1_9*risk9*x_AI9

print(f"\n[9.4.1] LP tối đa NetJob: Tổng = {-res9.fun:,.0f} việc làm")
print(f"  {'Ngành':<16} {'x_AI':>8} {'x_H':>8} {'NetJob':>8}")
for i in range(N9):
    print(f"  {sectors_9[i]:<16} {x_AI9[i]:>8.0f} {x_H9[i]:>8.0f} {NetJob9[i]:>8.0f}")

print(f"\n[9.4.2] Ngưỡng đào tạo tối thiểu – CN chế biến (i=1):")
i9 = 1
retrain_ratio9 = c1_9[i9]*risk9[i9]/d1_9[i9]
print(f"  net_AI={a1_9[i9]-c1_9[i9]*risk9[i9]:.1f}  retrain_ratio={retrain_ratio9:.4f}")
print(f"  Mỗi 1 tỷ đầu tư AI cần {retrain_ratio9:.4f} tỷ đào tạo để đảm bảo capacity")

print(f"\n[9.4.3] Nhóm dễ tổn thương (Nông-LT, Xây dựng, Bán buôn):")
for i in [0,2,3]:
    pct = NetJob9[i]/(L9[i]*1e6)*100
    print(f"  {sectors_9[i]:<16}: Displaced={Displaced9[i]:>8.0f}  "
          f"NetJob={NetJob9[i]:>8.0f}  NetJob/LĐ={pct:.2f}%")

print(f"\n[9.4.4] Thêm ràng buộc Displaced <= 5% L:")
A4_9 = np.zeros((N9, 2*N9))
for i in range(N9): A4_9[i,i] = c1_9[i]*risk9[i]
b4_9 = 0.05*L9*1e6
res9b = linprog(c_obj9, A_ub=np.vstack([A_ub9,A4_9]),
                b_ub=np.concatenate([b_ub9,b4_9]),
                bounds=[(0,None)]*(2*N9), method='highs')
if res9b.success:
    diff9 = -res9.fun - (-res9b.fun)
    print(f"  NetJob(không RB) = {-res9.fun:,.0f}")
    print(f"  NetJob(có RB 5%) = {-res9b.fun:,.0f}")
    print(f"  Giảm: {diff9:,.0f} ({diff9/(-res9.fun)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# BÀI 10 – Quy hoạch ngẫu nhiên 2 giai đoạn (Pyomo)
# ══════════════════════════════════════════════════════════════════════
import pyomo.environ as pyo

print("\n" + "█"*60)
print("  BÀI 10 – QUY HOẠCH NGẪU NHIÊN 2 GIAI ĐOẠN (PYOMO)")
print("█"*60)

J10 = ['I','D','AI','H']
S10 = ['s1','s2','s3','s4']
p_s10 = {'s1':0.30,'s2':0.45,'s3':0.20,'s4':0.05}

beta_base10 = {'I':1.00,'D':1.10,'AI':1.25,'H':0.95}
beta_s10 = {
    ('s1','I'):1.25,('s1','D'):1.35,('s1','AI'):1.55,('s1','H'):1.05,
    ('s2','I'):1.00,('s2','D'):1.10,('s2','AI'):1.25,('s2','H'):0.95,
    ('s3','I'):0.75,('s3','D'):0.85,('s3','AI'):0.90,('s3','H'):1.00,
    ('s4','I'):0.40,('s4','D'):0.50,('s4','AI'):0.55,('s4','H'):1.10,
}

def build_model10(include_second=True, fixed_scenario=None, fixed_x=None):
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=J10)
    m.S = pyo.Set(initialize=S10 if fixed_scenario is None else [fixed_scenario])
    m.beta = pyo.Param(m.J, initialize=beta_base10)
    bs_f = {(s,j): beta_s10[s,j]
            for s in (S10 if fixed_scenario is None else [fixed_scenario]) for j in J10}
    m.beta_s = pyo.Param(m.S, m.J, initialize=bs_f)
    m.p = pyo.Param(m.S,
                    initialize=p_s10 if fixed_scenario is None else {fixed_scenario:1.0})
    if fixed_x is None:
        m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    else:
        m.x = pyo.Param(m.J, initialize=fixed_x)
    if include_second:
        m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
        def b2_rule(m,s): return sum(m.y[s,j] for j in m.J) <= 15000
        m.budget2 = pyo.Constraint(m.S, rule=b2_rule)
        def ai_cap10(m,s):
            xH = m.x['H'] if fixed_x is None else fixed_x['H']
            return m.y[s,'AI'] <= 0.5*xH
        m.ai_cap = pyo.Constraint(m.S, rule=ai_cap10)
    if fixed_x is None:
        m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65000)
    def obj_rule(m):
        first = sum(m.beta[j]*m.x[j] for j in m.J)
        if include_second:
            second = sum(m.p[s]*sum(m.beta_s[s,j]*m.y[s,j] for j in m.J) for s in m.S)
            return first + second
        return first
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    return m

def solve10(m):
    for solver_name in ['appsi_highs','glpk','cbc']:
        solver = pyo.SolverFactory(solver_name)
        if solver.available():
            solver.solve(m, tee=False)
            return
    raise RuntimeError("Không tìm thấy solver LP (highs/glpk/cbc)")

def extract10(m, has_y=True):
    x_v = {j: pyo.value(m.x[j]) for j in J10}
    y_v = {}
    if has_y:
        for s in list(m.S):
            y_v[s] = {j: pyo.value(m.y[s,j]) for j in J10}
    return x_v, y_v, pyo.value(m.obj)

# Câu 10.5.1
print("\n[10.5.1] Mô hình SP:")
m10 = build_model10()
solve10(m10)
x10_sp, y10_sp, Z10_SP = extract10(m10)
print(f"  Z*_SP = {Z10_SP:,.2f}")
print(f"  First-stage: " + ", ".join(f"{j}={x10_sp[j]:.0f}" for j in J10))

# Câu 10.5.2 – EV/WS
print("\n[10.5.2] Giải từng kịch bản riêng:")
det10 = {}
for s in S10:
    md = build_model10(fixed_scenario=s)
    solve10(md); x_d,y_d,Z_d = extract10(md)
    det10[s] = {'x':x_d,'y':y_d,'Z':Z_d}
    print(f"  {s}: Z*={Z_d:,.2f}")

Z10_WS = sum(p_s10[s]*det10[s]['Z'] for s in S10)

# EV
beta_avg10 = {j: sum(p_s10[s]*beta_s10[s,j] for s in S10) for j in J10}
m_ev10 = pyo.ConcreteModel()
m_ev10.J = pyo.Set(initialize=J10)
m_ev10.x = pyo.Var(m_ev10.J, within=pyo.NonNegativeReals)
m_ev10.b1 = pyo.Constraint(expr=sum(m_ev10.x[j] for j in m_ev10.J) <= 65000)
m_ev10.obj = pyo.Objective(expr=sum(beta_avg10[j]*m_ev10.x[j] for j in m_ev10.J),
                           sense=pyo.maximize)
solve10(m_ev10)
x10_ev = {j: pyo.value(m_ev10.x[j]) for j in J10}

Z10_EV = sum(beta_base10[j]*x10_ev[j] for j in J10)
for s in S10:
    mt = build_model10(fixed_x=x10_ev, fixed_scenario=s)
    solve10(mt); _, yt, _ = extract10(mt)
    Z10_EV += p_s10[s]*sum(beta_s10[s,j]*yt[s][j] for j in J10)

# Câu 10.5.3
print(f"\n[10.5.3] VSS và EVPI:")
VSS10 = Z10_SP - Z10_EV; EVPI10 = Z10_WS - Z10_SP
print(f"  Z_SP={Z10_SP:,.2f}  Z_EV={Z10_EV:,.2f}  Z_WS={Z10_WS:,.2f}")
print(f"  VSS={VSS10:,.2f} ({VSS10/Z10_SP*100:.2f}%)")
print(f"  EVPI={EVPI10:,.2f} ({EVPI10/Z10_SP*100:.2f}%)")

# Câu 10.5.4 – Robust (minimax regret)
print("\n[10.5.4] Robust optimization (minimax regret):")
m_rob10 = pyo.ConcreteModel()
m_rob10.J = pyo.Set(initialize=J10)
m_rob10.S = pyo.Set(initialize=S10)
m_rob10.x = pyo.Var(m_rob10.J, within=pyo.NonNegativeReals)
m_rob10.y = pyo.Var(m_rob10.S, m_rob10.J, within=pyo.NonNegativeReals)
m_rob10.w = pyo.Var(within=pyo.Reals)
m_rob10.beta = pyo.Param(m_rob10.J, initialize=beta_base10)
m_rob10.beta_s = pyo.Param(m_rob10.S, m_rob10.J,
    initialize={(s,j):beta_s10[s,j] for s in S10 for j in J10})
m_rob10.p = pyo.Param(m_rob10.S, initialize=p_s10)
m_rob10.Zstar = pyo.Param(m_rob10.S, initialize={s:det10[s]['Z'] for s in S10})
m_rob10.b1 = pyo.Constraint(expr=sum(m_rob10.x[j] for j in m_rob10.J) <= 65000)
def b2r(m,s): return sum(m.y[s,j] for j in m.J) <= 15000
m_rob10.b2 = pyo.Constraint(m_rob10.S, rule=b2r)
def aic_r(m,s): return m.y[s,'AI'] <= 0.5*m.x['H']
m_rob10.aic = pyo.Constraint(m_rob10.S, rule=aic_r)
def reg_r(m,s):
    z_h = sum(m.beta[j]*m.x[j] for j in m.J)+sum(m.beta_s[s,j]*m.y[s,j] for j in m.J)
    return m.Zstar[s]-z_h <= m.w
m_rob10.regret = pyo.Constraint(m_rob10.S, rule=reg_r)
m_rob10.obj = pyo.Objective(expr=m_rob10.w, sense=pyo.minimize)
solve10(m_rob10)
print(f"  Minimax regret = {pyo.value(m_rob10.w):.0f}")


# ══════════════════════════════════════════════════════════════════════
# BÀI 11 – Q-learning cho chính sách kinh tế thích nghi
# ══════════════════════════════════════════════════════════════════════
import gymnasium as gym
from gymnasium import spaces

print("\n" + "█"*60)
print("  BÀI 11 – Q-LEARNING CHÍNH SÁCH KINH TẾ THÍCH NGHI")
print("█"*60)

class VietnamEconomyEnv11(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([3,3,3,3])
        self.T = 10
        self.allocation = {
            0: np.array([0.70,0.10,0.10,0.10]),
            1: np.array([0.40,0.25,0.15,0.20]),
            2: np.array([0.25,0.45,0.15,0.15]),
            3: np.array([0.20,0.20,0.45,0.15]),
            4: np.array([0.30,0.20,0.10,0.40]),
        }
        self.action_names = ['Truyền thống','Cân bằng','Số hóa nhanh','AI dẫn dắt','Bao trùm']
        self.w = np.array([0.40,0.25,0.20,0.15])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'state' in options:
            self.state = np.array(options['state'])
        else:
            self.state = self.np_random.integers(0,3,size=4)
        self.t=0; self.K=27500.0; self.D=20.3; self.AI=86.0; self.H=30.0; self.Y_prev=12847.6
        return self.state.copy(), {}

    def step(self, action):
        a = self.allocation[action]; budget=2100.0
        dK=a[0]*budget; dD=a[1]*budget*0.01; dAI=a[2]*budget*0.05; dH=a[3]*budget*0.01
        self.K=(1-0.05)*self.K+dK; self.D=(1-0.12)*self.D+dD
        self.AI=(1-0.15)*self.AI+dAI; self.H=self.H+0.8*dH-0.02*self.H
        A=33.70*(1+0.003*(self.D/100)+0.002*(self.AI/100)+0.004*(self.H/100))**self.t
        L=53.9*1.009**self.t
        Y=A*self.K**0.33*L**0.42*self.D**0.10*self.AI**0.08*self.H**0.07
        delta_gdp=(Y-self.Y_prev)/self.Y_prev
        delta_u=max(0,-delta_gdp*0.5)
        cyber_risk=(self.AI/(self.H+1))*0.01
        emission=(self.K+self.AI)*0.0001
        reward=(self.w[0]*delta_gdp*100 - self.w[1]*delta_u*100
                - self.w[2]*cyber_risk - self.w[3]*emission)
        self.Y_prev=Y; self.t+=1
        gdp_l=0 if delta_gdp<0.03 else (1 if delta_gdp<0.06 else 2)
        d_l=0 if self.D<25 else (1 if self.D<35 else 2)
        ai_l=0 if self.AI<100 else (1 if self.AI<200 else 2)
        h_l=0 if self.H<35 else (1 if self.H<50 else 2)
        self.state=np.array([gdp_l,d_l,ai_l,h_l])
        return self.state.copy(), reward, self.t>=self.T, False, {}

print("\n[11.3.1+2] Huấn luyện Q-learning (10.000 episodes)...")
Q11 = np.zeros((3,3,3,3,5))
env11 = VietnamEconomyEnv11()
for ep in range(10000):
    s11,_ = env11.reset()
    eps11 = max(0.05, 1.0-ep/5000)
    while True:
        a11 = env11.action_space.sample() if np.random.rand()<eps11 else int(np.argmax(Q11[tuple(s11)]))
        s11_next,r11,done11,_,_ = env11.step(a11)
        Q11[tuple(s11)+(a11,)] += 0.1*(r11+0.95*np.max(Q11[tuple(s11_next)])*(1-done11)
                                        - Q11[tuple(s11)+(a11,)])
        s11 = s11_next
        if done11: break

print("\n[11.3.3] Chính sách π*(s):")
test_states11 = [
    ([1,1,0,1], "VN 2026 thực tế"),
    ([0,0,0,2], "Kịch bản tệ"),
    ([2,2,2,2], "Kịch bản tốt"),
]
for st11, desc11 in test_states11:
    a_best = int(np.argmax(Q11[tuple(st11)]))
    print(f"  {desc11}: π* = {env11.action_names[a_best]}")

print("\n[11.3.4] So sánh chính sách:")
def eval_policy11(fn, n=300):
    rs = []
    for _ in range(n):
        s,_ = env11.reset(); tot=0
        while True:
            a = fn(s); s,r,d,_,_ = env11.step(a); tot+=r
            if d: break
        rs.append(tot)
    return np.mean(rs), np.std(rs)

for name11, fn11 in [('π* Q-learning', lambda s: int(np.argmax(Q11[tuple(s)]))),
                     ('Luôn Cân bằng', lambda s: 1),
                     ('Luôn AI dẫn dắt', lambda s: 3),
                     ('Random', lambda s: np.random.randint(5))]:
    m11,std11 = eval_policy11(fn11)
    print(f"  {name11:<20}: mean={m11:8.2f}  std={std11:.2f}")


# ══════════════════════════════════════════════════════════════════════
# BÀI 12 – AIDEOM-VN: Nguyên mẫu tích hợp 6 module
# ══════════════════════════════════════════════════════════════════════
"""AIDEOM-VN: Nguyên mẫu tích hợp 6 module."""

print("\n" + "█"*60)
print("  BÀI 12 – AIDEOM-VN: TÍCH HỢP 6 MODULE")
print("█"*60)

# ── M1: Dự báo Cobb-Douglas 2026-2030 ───────────────────────────────
print("\n[M1] Dự báo Cobb-Douglas 2026-2030:")
a12,b12,g12,d12,th12 = 0.33,0.42,0.10,0.08,0.07
K0_12,L0_12,D0_12,AI0_12,H0_12,A0_12 = 27500,53.9,20.3,86,30,33.70
T12=4; budget12=3000

def forecast12(K0,D0,AI0,H0,A0,L0,alloc,T=T12):
    K,D,AI,H,A=K0,D0,AI0,H0,A0
    traj=[A*K**a12*L0**b12*D**g12*AI**d12*H**th12]
    for t in range(T):
        K=(1-0.05)*K+alloc['K']*budget12
        D=(1-0.12)*D+alloc['D']*budget12*0.01
        AI=(1-0.15)*AI+alloc['AI']*budget12*0.05
        H=H+0.8*alloc['H']*budget12*0.01-0.02*H
        A=A*(1+0.003*(D/100)+0.002*(AI/100)+0.004*(H/100))
        L=L0*1.009**(t+1)
        traj.append(A*K**a12*L**b12*D**g12*AI**d12*H**th12)
    return traj

scenarios12 = {
    'S1_Truyền thống': {'K':0.70,'D':0.10,'AI':0.10,'H':0.10},
    'S2_Số hóa nhanh': {'K':0.25,'D':0.45,'AI':0.15,'H':0.15},
    'S3_AI dẫn dắt':   {'K':0.20,'D':0.20,'AI':0.45,'H':0.15},
    'S4_Bao trùm':     {'K':0.30,'D':0.20,'AI':0.10,'H':0.40},
    'S5_Tối ưu':       {'K':0.25,'D':0.25,'AI':0.30,'H':0.20},
}
gdp_fc12 = {n: forecast12(K0_12,D0_12,AI0_12,H0_12,A0_12,L0_12,al)
            for n,al in scenarios12.items()}
years12 = list(range(2026, 2026+T12+1))
header12 = f"  {'Kịch bản':<22}" + "".join(f"{y:>8}" for y in years12)
print(header12); print("  "+"-"*60)
for name12, traj12 in gdp_fc12.items():
    print(f"  {name12:<22}" + "".join(f"{v:>8.0f}" for v in traj12))

# ── M2: TOPSIS + Entropy ────────────────────────────────────────────
print("\n[M2] TOPSIS + Entropy weight – phân loại vùng kinh tế:")
RAW_12 = """region_id,region_name_en,region_name_short,grdp_per_capita_million_VND,fdi_registered_billion_USD,digital_index_0_100,ai_readiness_0_100,trained_labor_pct,rd_intensity_pct,internet_penetration_pct,gini_coef
1,NMM,NMM,57.0,3.5,38,22,21.5,0.18,72,0.405
2,RRD,RRD,152.3,20.0,78,68,36.8,0.85,92,0.358
3,NSCC,NSCC,87.5,8.2,55,40,27.5,0.32,84,0.372
4,CH,CH,68.9,0.8,32,18,18.2,0.15,68,0.412
5,SE,SE,158.9,18.5,82,75,42.5,0.78,94,0.385
6,MD,MD,80.5,2.1,48,30,16.8,0.22,78,0.392"""
df12 = pd.read_csv(StringIO(RAW_12))
crit12 = ['grdp_per_capita_million_VND','fdi_registered_billion_USD',
          'digital_index_0_100','ai_readiness_0_100',
          'trained_labor_pct','rd_intensity_pct',
          'internet_penetration_pct','gini_coef']
ib12 = [True,True,True,True,True,True,True,False]
w12_exp = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
X12 = df12[crit12].values.astype(float)
w12_ent = ew_6(X12)
C12_exp = topsis_6(X12, w12_exp, ib12)
C12_ent = topsis_6(X12, w12_ent, ib12)
print(f"  {'Vùng':<6} {'C*_expert':>10} {'C*_entropy':>11}")
for i in range(len(df12)):
    print(f"  {df12['region_name_short'][i]:<6} {C12_exp[i]:>10.4f} {C12_ent[i]:>11.4f}")

# ── M3: LP phân bổ ngân sách ─────────────────────────────────────────
print("\n[M3] LP phân bổ ngân sách theo vùng:")
regions12 = ['NMM','RRD','NCC','CH','SE','MD']
items12 = ['I','D','AI','H']
beta12 = {
    ('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
    ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
    ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
    ('CH','I') :1.20,('CH','D') :0.75,('CH','AI') :0.45,('CH','H') :1.35,
    ('SE','I') :0.90,('SE','D') :1.30,('SE','AI') :1.55,('SE','H') :1.00,
    ('MD','I') :1.10,('MD','D') :0.85,('MD','AI') :0.65,('MD','H') :1.25,
}
m12_lp = pulp.LpProblem('M3_LP', pulp.LpMaximize)
x12 = pulp.LpVariable.dicts('x', (regions12, items12), lowBound=0)
m12_lp += pulp.lpSum(beta12[(r,j)]*x12[r][j] for r in regions12 for j in items12)
m12_lp += pulp.lpSum(x12[r][j] for r in regions12 for j in items12) <= 50000
for r in regions12:
    m12_lp += pulp.lpSum(x12[r][j] for j in items12) >= 5000
    m12_lp += pulp.lpSum(x12[r][j] for j in items12) <= 12000
m12_lp += pulp.lpSum(x12[r]['H'] for r in regions12) >= 12000
m12_lp.solve(pulp.PULP_CBC_CMD(msg=False))
Z12_lp = pulp.value(m12_lp.objective)
print(f"  LP Z* = {Z12_lp:.0f}")
for r in regions12:
    alloc_r = {j: pulp.value(x12[r][j]) for j in items12}
    tot_r = sum(alloc_r.values())
    print(f"  {r}: I={alloc_r['I']:>6.0f}  D={alloc_r['D']:>6.0f}  "
          f"AI={alloc_r['AI']:>6.0f}  H={alloc_r['H']:>6.0f}  Tổng={tot_r:>7.0f}")

# ── M4: Thị trường lao động ──────────────────────────────────────────
print("\n[M4] Thị trường lao động (LP NetJob):")
a1_12=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
b1_12=np.array([45,28,35,32,22,30,20,55])
c1_12=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
d1_12=np.array([50,32,42,38,26,36,24,62])
risk12=np.array([18,42,25,38,52,35,28,22])/100
coeff12=a1_12-c1_12*risk12
c_obj12=np.concatenate([-coeff12,-b1_12])
A1_12=np.concatenate([np.ones(8),np.ones(8)]).reshape(1,-1)
A2_12=np.zeros((8,16))
for i in range(8): A2_12[i,i]=-coeff12[i]; A2_12[i,8+i]=-b1_12[i]
A3_12=np.zeros((8,16))
for i in range(8): A3_12[i,i]=c1_12[i]*risk12[i]; A3_12[i,8+i]=-d1_12[i]
res12_lp=linprog(c_obj12,A_ub=np.vstack([A1_12,A2_12,A3_12]),
                 b_ub=np.concatenate([[30000],np.zeros(8),np.zeros(8)]),
                 bounds=[(0,None)]*16,method='highs')
print(f"  Tổng NetJob = {-res12_lp.fun:,.0f} việc làm")

print("\n" + "═"*60)
print("  ✔  HOÀN THÀNH TẤT CẢ 12 BÀI")
print("═"*60)
