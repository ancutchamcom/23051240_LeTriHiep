# ============================================================
#  BÀI 4.4 – TỐI ƯU PHÂN BỔ NGÂN SÁCH SỐ HOÁ VIỆT NAM
#  Giải quyết đầy đủ câu 4.4.1 → 4.4.4
#  Chạy trực tiếp trên Google Colab (1 cell duy nhất)
# ============================================================

# ---------- 0. Cài đặt thư viện ----------



# ---------- 1. Import ----------
import pulp
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  DỮ LIỆU BÀI TOÁN
# ============================================================
regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
items   = ['I', 'D', 'AI', 'H']  # Infrastructure, Digital, AI, Human Capital

region_names = {
    'NMM': 'N. Midlands & Mountains',
    'RRD': 'Red River Delta',
    'NCC': 'N&S Central Coast',
    'CH' : 'Central Highlands',
    'SE' : 'Southeast',
    'MD' : 'Mekong Delta'
}
item_names = {
    'I' : 'Infrastructure (I)',
    'D' : 'Digital (D)',
    'AI': 'AI & Tech (AI)',
    'H' : 'Human Capital (H)'
}

# Hệ số tác động GDP β_{r,j}
beta = {
    ('NMM','I'):1.15, ('NMM','D'):0.85, ('NMM','AI'):0.55, ('NMM','H'):1.30,
    ('RRD','I'):0.95, ('RRD','D'):1.25, ('RRD','AI'):1.40, ('RRD','H'):1.05,
    ('NCC','I'):1.05, ('NCC','D'):0.95, ('NCC','AI'):0.85, ('NCC','H'):1.15,
    ('CH' ,'I'):1.20, ('CH' ,'D'):0.75, ('CH' ,'AI'):0.45, ('CH' ,'H'):1.35,
    ('SE' ,'I'):0.90, ('SE' ,'D'):1.30, ('SE' ,'AI'):1.55, ('SE' ,'H'):1.00,
    ('MD' ,'I'):1.10, ('MD' ,'D'):0.85, ('MD' ,'AI'):0.65, ('MD' ,'H'):1.25,
}

D0    = {'NMM':38, 'RRD':78, 'NCC':55, 'CH':32, 'SE':82, 'MD':48}
gamma = 0.002  # Hệ số cải thiện chỉ số số hoá
# ---------------------------------------------------------------
# GHI CHÚ FEASIBILITY: Với D0_max=82(SE), D0_min=32(CH), gamma=0.002
#   D_CH_max = 32 + 0.002*12000 = 56.0
#   D_SE_min = 82 (SE không cần đầu tư D để đã cao)
#   ratio_max = 56/82 ≈ 0.683 → λ phải ≤ 0.683 để feasible
#   → Chọn λ = 0.65 (vẫn đảm bảo công bằng ≥ 65% mức cao nhất)
# ---------------------------------------------------------------
lam = 0.65  # Ngưỡng công bằng λ (đã điều chỉnh cho feasibility)

BUDGET = 50_000  # Tổng ngân sách (tỷ VND)
MIN_R  =  5_000  # Ngân sách tối thiểu mỗi vùng
MAX_R  = 12_000  # Ngân sách tối đa mỗi vùng
MIN_H  = 12_000  # Tổng chi tối thiểu cho Nhân lực (H)

# ============================================================
#  HÀM TIỆN ÍCH
# ============================================================
def extract_matrix_pulp(x_var, regs, its):
    M = np.zeros((len(regs), len(its)))
    for i, r in enumerate(regs):
        for j, it in enumerate(its):
            val = pulp.value(x_var[r][it])
            M[i, j] = val if val is not None else 0.0
    return M

def print_allocation_matrix(matrix, regs, its, title, z_star):
    df = pd.DataFrame(np.round(matrix, 1), index=regs, columns=its)
    df['TOTAL'] = df.sum(axis=1)
    df.loc['TOTAL'] = df.sum()
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"  Z* (GDP gain) = {z_star:,.2f} tỷ VND")
    print(f"{'='*65}")
    print(df.to_string())
    print(f"{'='*65}")

# ============================================================
#  CÂU 4.4.1 – PuLP / CBC
# ============================================================
print("\n" + "▓"*65)
print("  CÂU 4.4.1 – MÔ HÌNH PuLP (Solver: CBC)")
print("▓"*65)

m_pulp = pulp.LpProblem('VN_Digital_Budget_PuLP', pulp.LpMaximize)

# Biến quyết định x_{r,j} ≥ 0
x_p = pulp.LpVariable.dicts('x', (regions, items), lowBound=0)

# Biến phụ M = D_max (chỉ số số hoá cao nhất sau đầu tư)
M_p = pulp.LpVariable('Dmax', lowBound=0)

# Hàm mục tiêu: max Σ β_{r,j} · x_{r,j}
m_pulp += pulp.lpSum(beta[(r,j)] * x_p[r][j]
                     for r in regions for j in items), "GDP_gain"

# C1: Tổng ngân sách ≤ 50.000 tỷ
m_pulp += (pulp.lpSum(x_p[r][j] for r in regions for j in items)
           <= BUDGET), "C1_budget"

# C2: Ngân sách tối thiểu mỗi vùng ≥ 5.000
for r in regions:
    m_pulp += (pulp.lpSum(x_p[r][j] for j in items)
               >= MIN_R), f"C2_min_{r}"

# C3: Ngân sách tối đa mỗi vùng ≤ 12.000
for r in regions:
    m_pulp += (pulp.lpSum(x_p[r][j] for j in items)
               <= MAX_R), f"C3_max_{r}"

# C4: Tổng chi nhân lực H ≥ 12.000
m_pulp += (pulp.lpSum(x_p[r]['H'] for r in regions)
           >= MIN_H), "C4_human"

# C5: Công bằng số hoá (linearization qua biến phụ M = Dmax)
#   Ý nghĩa: Chỉ số số hoá thấp nhất phải ≥ λ × chỉ số cao nhất
#   Linearization:
#     (a) D0[r] + γ·x_{r,D} ≤ M   ∀r  →  M = max của tất cả chỉ số số hoá
#     (b) D0[r] + γ·x_{r,D} ≥ λ·M ∀r  →  min ≥ λ·max
#   Thêm upper bound cho M để tránh M→∞ kéo nghiệm thành infeasible:
#     (c) M ≤ max(D0) + γ·MAX_R   (M không thể vượt mức tối đa lý thuyết)
for r in regions:
    m_pulp += (D0[r] + gamma * x_p[r]['D']
               <= M_p), f"C5a_upper_{r}"
for r in regions:
    m_pulp += (D0[r] + gamma * x_p[r]['D']
               >= lam * M_p), f"C5b_lower_{r}"
# Upper bound cho M (ngăn M phình to vô hạn)
M_theoretical_max = max(D0.values()) + gamma * MAX_R  # = 82 + 24 = 106
m_pulp += M_p <= M_theoretical_max, "C5c_Mbound"

# Giải
status_pulp = m_pulp.solve(pulp.PULP_CBC_CMD(msg=0))
z_pulp      = pulp.value(m_pulp.objective)
mat_pulp    = extract_matrix_pulp(x_p, regions, items)

print_allocation_matrix(mat_pulp, regions, items,
                        "PuLP – Phân bổ tối ưu x_{j,r} (tỷ VND)", z_pulp)
print(f"\nTrạng thái solver : {pulp.LpStatus[status_pulp]}")
print(f"Dmax (M*)         : {pulp.value(M_p):.2f}")
print(f"λ áp dụng         : {lam}  (ngưỡng feasibility tối đa ≈ 0.683)")

# Kiểm tra C5
print(f"\nKiểm tra ràng buộc C5 (D0 + γ·xD):")
d_vals = {}
for i, r in enumerate(regions):
    d_vals[r] = D0[r] + gamma * mat_pulp[i, 1]
    print(f"  {r}: {d_vals[r]:.2f}")
dmax_check = max(d_vals.values())
dmin_check = min(d_vals.values())
print(f"  min/max = {dmin_check:.2f}/{dmax_check:.2f} = {dmin_check/dmax_check:.4f}  "
      f"(λ={lam} → {'✅ THỎA' if dmin_check/dmax_check >= lam - 0.001 else '❌ VI PHẠM'})")

# ============================================================
#  CÂU 4.4.2 – CVXPY (tự động chọn solver LP tốt nhất)
# ============================================================
print("\n" + "▓"*65)
print("  CÂU 4.4.2 – MÔ HÌNH CVXPY")
print("▓"*65)

R = len(regions)
J = len(items)

beta_mat = np.array([[beta[(r,j)] for j in items] for r in regions])  # (6,4)
D0_vec   = np.array([D0[r] for r in regions], dtype=float)            # (6,)

x_c  = cp.Variable((R, J), nonneg=True)
Mmax = cp.Variable(nonneg=True)

objective_c = cp.Maximize(cp.sum(cp.multiply(beta_mat, x_c)))

constraints_c = [
    cp.sum(x_c)          <= BUDGET,              # C1
    cp.sum(x_c, axis=1)  >= MIN_R,               # C2
    cp.sum(x_c, axis=1)  <= MAX_R,               # C3
    cp.sum(x_c[:, 3])    >= MIN_H,               # C4  (H = col 3)
    D0_vec + gamma * x_c[:, 1] <= Mmax,          # C5a (D = col 1)
    D0_vec + gamma * x_c[:, 1] >= lam * Mmax,    # C5b
    Mmax <= M_theoretical_max,                    # C5c upper bound cho M
]

prob_cvxpy = cp.Problem(objective_c, constraints_c)

# Thử các solver theo thứ tự ưu tiên (tương thích Colab)
_solver_used = None
for _solver in [cp.GLPK, cp.SCIPY, cp.SCS, cp.ECOS]:
    try:
        prob_cvxpy.solve(solver=_solver, verbose=False)
        if prob_cvxpy.status in ["optimal", "optimal_inaccurate"]:
            _solver_used = str(_solver)
            break
    except Exception:
        continue

if prob_cvxpy.value is None:
    # Fallback: để CVXPY tự chọn solver
    prob_cvxpy.solve(verbose=False)
    _solver_used = "auto"

z_cvxpy   = prob_cvxpy.value
mat_cvxpy = np.clip(x_c.value, 0, None) if x_c.value is not None else np.zeros((R, J))

print_allocation_matrix(mat_cvxpy, regions, items,
                        "CVXPY – Phân bổ tối ưu x_{j,r} (tỷ VND)", z_cvxpy)
print(f"\nTrạng thái solver : {prob_cvxpy.status}  (solver dùng: {_solver_used})")

# So sánh PuLP vs CVXPY
diff_z   = abs(z_pulp - z_cvxpy)
diff_mat = np.abs(mat_pulp - mat_cvxpy).max()
print(f"\n{'─'*58}")
print(f"  SO SÁNH PuLP vs CVXPY")
print(f"{'─'*58}")
print(f"  Z* (PuLP)              = {z_pulp:>12,.2f} tỷ VND")
print(f"  Z* (CVXPY)             = {z_cvxpy:>12,.2f} tỷ VND")
print(f"  Chênh lệch Z*          = {diff_z:>12.4f} tỷ VND")
print(f"  Chênh lệch ma trận max = {diff_mat:>12.4f} tỷ VND")
if diff_z < 10:
    print("  ✅ Hai phương pháp cho KẾT QUẢ NHẤT QUÁN")
    print("     (sai số nhỏ do solver khác nhau: CBC vs LP continuous)")
else:
    print("  ⚠️  Sai lệch đáng kể – kiểm tra lại ràng buộc")
print(f"{'─'*58}")

# ============================================================
#  CÂU 4.4.3 – HEATMAP PHÂN BỔ TỐI ƯU
# ============================================================
print("\n" + "▓"*65)
print("  CÂU 4.4.3 – HEATMAP PHÂN BỔ TỐI ƯU")
print("▓"*65)

region_labels = [region_names[r] for r in regions]
item_labels   = [item_names[j] for j in items]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Phân bổ Ngân sách Số hoá Tối ưu – Việt Nam 2024\n(Đơn vị: tỷ VND)",
             fontsize=14, fontweight='bold')

# --- (a) Heatmap giá trị tuyệt đối ---
ax1 = axes[0]
df_heat = pd.DataFrame(mat_pulp, index=region_labels, columns=item_labels)
sns.heatmap(df_heat, ax=ax1, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.6, linecolor='white',
            cbar_kws={'label': 'Tỷ VND', 'shrink': 0.8},
            annot_kws={'size': 11, 'weight': 'bold'})
ax1.set_title("(a) Phân bổ tuyệt đối (tỷ VND)", fontsize=12, pad=10)
ax1.set_xlabel("Hạng mục đầu tư", fontsize=10)
ax1.set_ylabel("Vùng kinh tế", fontsize=10)
ax1.tick_params(axis='x', rotation=15)
ax1.tick_params(axis='y', rotation=0)

# --- (b) Heatmap tỷ lệ % trong ngân sách từng vùng ---
ax2 = axes[1]
row_sum = mat_pulp.sum(axis=1, keepdims=True)
df_pct  = pd.DataFrame(mat_pulp / row_sum * 100,
                        index=region_labels, columns=item_labels)
sns.heatmap(df_pct, ax=ax2, annot=True, fmt='.1f', cmap='Blues',
            linewidths=0.6, linecolor='white',
            cbar_kws={'label': '% ngân sách vùng', 'shrink': 0.8},
            annot_kws={'size': 11, 'weight': 'bold'})
ax2.set_title("(b) Tỷ trọng trong ngân sách từng vùng (%)", fontsize=12, pad=10)
ax2.set_xlabel("Hạng mục đầu tư", fontsize=10)
ax2.set_ylabel("Vùng kinh tế", fontsize=10)
ax2.tick_params(axis='x', rotation=15)
ax2.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig("heatmap_allocation.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Đã lưu: heatmap_allocation.png")

# --- Biểu đồ cột tổng hợp ---
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))
fig2.suptitle("Tổng hợp phân bổ ngân sách tối ưu (PuLP/CBC)",
              fontsize=13, fontweight='bold')

# Stacked bar theo vùng
ax_r = axes2[0]
df_reg  = pd.DataFrame(mat_pulp, index=regions, columns=items)
colors  = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
bottom  = np.zeros(R)
for it, col in zip(items, colors):
    ax_r.bar(regions, df_reg[it].values, bottom=bottom,
             label=item_names[it], color=col, edgecolor='white', linewidth=0.5)
    bottom += df_reg[it].values
ax_r.axhline(MAX_R, color='red',    linestyle='--', lw=1.2, label=f'Trần {MAX_R:,} tỷ')
ax_r.axhline(MIN_R, color='orange', linestyle=':',  lw=1.2, label=f'Sàn {MIN_R:,} tỷ')
ax_r.set_title("Phân bổ theo Vùng (tỷ VND)", fontsize=11)
ax_r.set_ylabel("Tỷ VND"); ax_r.set_xlabel("Vùng kinh tế")
ax_r.legend(loc='upper right', fontsize=8)
ax_r.set_ylim(0, MAX_R * 1.2)
for i, v in enumerate(df_reg.sum(axis=1)):
    ax_r.text(i, v + 150, f"{v:,.0f}", ha='center', fontsize=8.5, fontweight='bold')

# Bar theo hạng mục
ax_j = axes2[1]
totals = mat_pulp.sum(axis=0)
bars   = ax_j.bar(item_labels, totals, color=colors, edgecolor='white', lw=0.8)
ax_j.set_title("Tổng theo Hạng mục (tỷ VND)", fontsize=11)
ax_j.set_ylabel("Tỷ VND"); ax_j.set_xlabel("Hạng mục đầu tư")
ax_j.tick_params(axis='x', rotation=10)
for bar, v in zip(bars, totals):
    ax_j.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
              f"{v:,.0f}", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("bar_allocation.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ Đã lưu: bar_allocation.png")

# Phân tích văn bản
total_by_region = mat_pulp.sum(axis=1)
top_r_idx       = np.argmax(total_by_region)
print(f"\n  📊 PHÂN TÍCH HEATMAP:")
print(f"  • Vùng nhận nhiều nhất : {regions[top_r_idx]} ({region_names[regions[top_r_idx]]}) "
      f"– {total_by_region[top_r_idx]:,.0f} tỷ VND")
print(f"  • Hạng mục ưu tiên từng vùng:")
for i, r in enumerate(regions):
    top_j = items[np.argmax(mat_pulp[i])]
    pct   = mat_pulp[i, np.argmax(mat_pulp[i])] / mat_pulp[i].sum() * 100
    print(f"      {r:4s} ({region_names[r]:26s}): {item_names[top_j]} ({pct:.1f}%)")

# ============================================================
#  CÂU 4.4.4 – SO SÁNH MÔ HÌNH CÓ / KHÔNG CÓ RÀNG BUỘC C5
# ============================================================
print("\n" + "▓"*65)
print("  CÂU 4.4.4 – CHI PHÍ KINH TẾ CỦA CÔNG BẰNG VÙNG MIỀN")
print("▓"*65)

# --- Mô hình KHÔNG có C5 ---
m_nc5 = pulp.LpProblem('VN_Budget_NoC5', pulp.LpMaximize)
x_nc  = pulp.LpVariable.dicts('xnc', (regions, items), lowBound=0)

m_nc5 += pulp.lpSum(beta[(r,j)] * x_nc[r][j] for r in regions for j in items)
m_nc5 += pulp.lpSum(x_nc[r][j] for r in regions for j in items) <= BUDGET
for r in regions:
    m_nc5 += pulp.lpSum(x_nc[r][j] for j in items) >= MIN_R
    m_nc5 += pulp.lpSum(x_nc[r][j] for j in items) <= MAX_R
m_nc5 += pulp.lpSum(x_nc[r]['H'] for r in regions) >= MIN_H
# (C5 bị bỏ hoàn toàn)

m_nc5.solve(pulp.PULP_CBC_CMD(msg=0))
z_nc5   = pulp.value(m_nc5.objective)
mat_nc5 = extract_matrix_pulp(x_nc, regions, items)

print_allocation_matrix(mat_nc5, regions, items,
                        "PuLP Không C5 – Phân bổ tối ưu (tỷ VND)", z_nc5)

# Đối chiếu
equity_cost = z_nc5 - z_pulp
print(f"\n  {'─'*60}")
print(f"  BẢNG ĐỐI CHIẾU: CÓ C5 vs. KHÔNG CÓ C5")
print(f"  {'─'*60}")
print(f"  Z* CÓ ràng buộc công bằng (C5)       = {z_pulp:>12,.2f} tỷ VND")
print(f"  Z* KHÔNG ràng buộc công bằng (No-C5) = {z_nc5:>12,.2f} tỷ VND")
print(f"  {'─'*60}")
print(f"  ► Chi phí kinh tế của công bằng       = {equity_cost:>12,.2f} tỷ VND")
print(f"    (GDP gain bị hy sinh để đảm bảo công bằng số hoá vùng miền)")
print(f"  ► Tỷ lệ tổn thất                      = {equity_cost/z_nc5*100:>11.3f} %")
print(f"  {'─'*60}")

# Phân tích phân phối lại
diff_mat_c5 = mat_nc5 - mat_pulp
print(f"\n  THAY ĐỔI PHÂN BỔ KHI BỎ C5 (No-C5 − Có-C5, tỷ VND):")
df_diff = pd.DataFrame(diff_mat_c5.round(1), index=regions, columns=items)
df_diff['TOTAL'] = df_diff.sum(axis=1)
print(df_diff.to_string())

# Chỉ số số hoá so sánh
print(f"\n  CHỈ SỐ SỐ HOÁ SAU ĐẦU TƯ (D0 + γ·xD):")
print(f"  {'Vùng':6s} {'D0':>6s} {'D_C5':>10s} {'D_noC5':>10s} {'Δ':>10s}")
print("  " + "─"*46)
d_c5_vec  = []
d_nc5_vec = []
for i, r in enumerate(regions):
    dc5  = D0[r] + gamma * mat_pulp[i, 1]
    dnc5 = D0[r] + gamma * mat_nc5[i, 1]
    d_c5_vec.append(dc5); d_nc5_vec.append(dnc5)
    print(f"  {r:6s} {D0[r]:>6.0f} {dc5:>10.2f} {dnc5:>10.2f} {dnc5-dc5:>+10.2f}")

max_c5  = max(d_c5_vec);  min_c5  = min(d_c5_vec)
max_nc5 = max(d_nc5_vec); min_nc5 = min(d_nc5_vec)
print(f"\n  Tỷ lệ công bằng (min/max chỉ số số hoá):")
print(f"  • Có  C5  : {min_c5:.2f} / {max_c5:.2f} = {min_c5/max_c5:.4f}  "
      f"(λ={lam} → {'✅ THỎA' if min_c5/max_c5 >= lam-0.001 else '❌ VI PHẠM'})")
print(f"  • Không C5: {min_nc5:.2f} / {max_nc5:.2f} = {min_nc5/max_nc5:.4f}  "
      f"(không bị ràng buộc)")

# --- Biểu đồ đối chiếu ---
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle("Đối chiếu: CÓ ràng buộc công bằng C5 vs. KHÔNG CÓ C5",
              fontsize=13, fontweight='bold')

total_c5_r  = mat_pulp.sum(axis=1)
total_nc5_r = mat_nc5.sum(axis=1)

# (a) Ngân sách tổng theo vùng
ax3a = axes3[0]
xpos = np.arange(R); w = 0.35
ax3a.bar(xpos - w/2, total_c5_r,  w, label='Có C5',    color='#1976D2', alpha=0.85)
ax3a.bar(xpos + w/2, total_nc5_r, w, label='Không C5', color='#F44336', alpha=0.85)
ax3a.set_xticks(xpos); ax3a.set_xticklabels(regions)
ax3a.axhline(MAX_R, color='gray', linestyle='--', lw=0.8, alpha=0.7)
ax3a.set_title("Tổng ngân sách theo Vùng (tỷ VND)", fontsize=10)
ax3a.set_ylabel("Tỷ VND"); ax3a.legend(fontsize=9)
ax3a.set_ylim(0, MAX_R * 1.2)

# (b) Chỉ số số hoá
ax3b = axes3[1]
ax3b.plot(regions, d_c5_vec,  'o-',  color='#1976D2', lw=2, ms=8, label='Có C5')
ax3b.plot(regions, d_nc5_vec, 's--', color='#F44336', lw=2, ms=8, label='Không C5')
ax3b.axhline(lam * max_c5, color='#1976D2', linestyle=':', alpha=0.5,
             label=f'Ngưỡng λ·Dmax={lam*max_c5:.1f}')
ax3b.set_title("Chỉ số số hoá sau đầu tư", fontsize=10)
ax3b.set_ylabel("Chỉ số số hoá"); ax3b.legend(fontsize=8)
ax3b.set_ylim(20)

# (c) Chi phí công bằng
ax3c = axes3[2]
cats   = ['Z* Có C5\n(với công bằng)', 'Z* Không C5\n(không công bằng)', 'Chi phí\ncông bằng Δ']
vals   = [z_pulp, z_nc5, equity_cost]
bcolors= ['#1976D2', '#F44336', '#FF9800']
bars_c = ax3c.bar(cats, vals, color=bcolors, edgecolor='white', lw=0.8)
ax3c.set_title("Chi phí kinh tế của công bằng", fontsize=10)
ax3c.set_ylabel("Tỷ VND")
for bar, v in zip(bars_c, vals):
    ax3c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
              f"{v:,.0f}", ha='center', fontsize=9, fontweight='bold')
ax3c.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig("comparison_c5_vs_no_c5.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n  ✅ Đã lưu: comparison_c5_vs_no_c5.png")

# ============================================================
#  TÓM TẮT CUỐI
# ============================================================
print("\n" + "█"*65)
print("  TÓM TẮT KẾT QUẢ – BÀI 4.4")
print("█"*65)
print(f"""
  4.4.1  PuLP/CBC  Z* = {z_pulp:,.2f} tỷ VND  [{pulp.LpStatus[status_pulp]}]
  4.4.2  CVXPY     Z* = {z_cvxpy:,.2f} tỷ VND  [{prob_cvxpy.status}]
         → Hai phương pháp: {'NHẤT QUÁN ✅' if diff_z < 10 else 'SÁI LỆCH ⚠️'} (ΔZ* = {diff_z:.2f})

  4.4.3  Vùng nhận nhiều nhất : {regions[top_r_idx]} ({region_names[regions[top_r_idx]]})
           = {total_by_region[top_r_idx]:,.0f} tỷ VND
         Hạng mục ưu tiên: H (Nhân lực) ở vùng kém phát triển,
                           AI ở vùng phát triển (SE, RRD)

  4.4.4  Chi phí kinh tế công bằng = {equity_cost:,.2f} tỷ VND
         ({equity_cost/z_nc5*100:.3f}% GDP gain bị hy sinh để đảm bảo
          tỷ lệ công bằng số hoá min/max ≥ λ = {lam})
         Chỉ số số hoá cải thiện vùng kém nhất (CH):
           {min(d_c5_vec):.2f} (có C5) vs {min(d_nc5_vec):.2f} (không C5)
""")
print("█"*65)