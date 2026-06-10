# ============================================================
# BÀI 7.4 - TỐI ƯU HÓA ĐẦU TƯ SỐ HÓA VIỆT NAM (NSGA-II + TOPSIS)
# Môn: Các Mô Hình Ra Quyết Định
# Chạy trên Google Colab - tất cả trong 1 cell
# ============================================================

# ── 0. CÀI ĐẶT THƯ VIỆN ──────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

print("=" * 65)
print("  BÀI 7.4: TỐI ƯU HÓA ĐẦU TƯ SỐ HÓA VIỆT NAM – NSGA-II")
print("=" * 65)

# ── 1. DỮ LIỆU THỰC TẾ TỪ CSV ────────────────────────────────
# 6 vùng: Northern Midlands, Red River Delta, North Central Coast,
#         Central Highlands, Southeast, Mekong Delta

# Hệ số beta (6 vùng × 4 hạng mục đầu tư: I=Hạ tầng, D=Số hóa, AI=Trí tuệ nhân tạo, H=Nhân lực)
# Được tính từ: grdp_growth, digital_index, ai_readiness, trained_labor
# Beta thể hiện "tỷ suất sinh lợi GDP" của mỗi đồng đầu tư theo vùng-hạng mục

# Dữ liệu vùng từ vietnam_regions_2024.csv
regions = {
    "name": ["North Midlands", "Red River Delta", "N.Central Coast",
              "Central Highlands", "Southeast", "Mekong Delta"],
    "grdp_growth":    [8.5,  7.9,  6.85, 7.2,  7.5,  7.3],   # %
    "digital_index":  [38,   78,   55,   32,   82,   48],     # 0-100
    "ai_readiness":   [22,   68,   40,   18,   75,   30],     # 0-100
    "trained_labor":  [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],  # %
    "gini":           [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
    "grdp_trillion":  [810,  3580, 1820, 420,  3050, 1409],   # VND nghìn tỷ
    "fdi":            [3.5,  20.0, 8.2,  0.8,  18.5, 2.1],   # tỷ USD
}

# Ma trận beta (6×4): hệ số lợi ích GDP theo [I, D, AI, H]
# Vùng kém phát triển (Central Highlands, North Midlands) có beta I cao (cần hạ tầng)
# Vùng phát triển (Southeast, Red River) có beta D, AI cao
beta_I  = np.array([0.000085, 0.000045, 0.000065, 0.000092, 0.000040, 0.000078])
beta_D  = np.array([0.000072, 0.000095, 0.000082, 0.000058, 0.000105, 0.000068])
beta_AI = np.array([0.000055, 0.000115, 0.000078, 0.000042, 0.000125, 0.000060])
beta_H  = np.array([0.000068, 0.000088, 0.000075, 0.000062, 0.000095, 0.000055])

beta = np.column_stack([beta_I, beta_D, beta_AI, beta_H])   # shape (6, 4)

# Hệ số phát thải carbon (tỷ lệ thuận với hạ tầng truyền thống & AI data center)
# e thấp ở vùng đã có hạ tầng xanh, cao ở vùng phụ thuộc than
e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])

# Hệ số rủi ro phụ thuộc công nghệ nước ngoài (AI)
rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])

# Hệ số an ninh từ đầu tư nhân lực (H) – giảm rủi ro
sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

# Ngân sách tổng: 60,000 tỷ VND (chia 4 hạng mục × 6 vùng = 24 biến)
TOTAL_BUDGET = 60000   # tỷ VND
N_REGIONS    = 6
N_CATEGORIES = 4      # I=Hạ tầng, D=Số hóa, AI=AI, H=Nhân lực

# Giới hạn đầu tư: mỗi biến x_{ij} ∈ [0, 12000] tỷ VND
XL = np.zeros(24)
XU = np.ones(24) * 12000

print(f"\n📊 Cấu hình bài toán:")
print(f"   • Biến quyết định: 24 (6 vùng × 4 hạng mục)")
print(f"   • Hàm mục tiêu  : 4 (GDP, Gini, CO₂, An ninh)")
print(f"   • Ràng buộc     : 12 (ngân sách + sàn đầu tư)")
print(f"   • Ngân sách tổng: {TOTAL_BUDGET:,} tỷ VND")
print(f"   • Ma trận beta  :\n{beta.round(6)}")


# ── 2. ĐỊNH NGHĨA BÀI TOÁN (Câu 7.4.1) ──────────────────────
class VietnamDigitalProblem(ElementwiseProblem):
    """
    Bài toán tối ưu đa mục tiêu phân bổ ngân sách số hóa Việt Nam.

    Biến:  x ∈ R^24 = [x_{11},...,x_{64}] (vùng i, hạng mục j)
           Reshape thành ma trận X (6×4): [I, D, AI, H]

    Mục tiêu:
      f1 = -Σ β_{ij}·x_{ij}        → MIN (maximize GDP gain)
      f2 = MAD(Σ_j x_{ij})         → MIN (minimize regional inequality)
      f3 = Σ e_i·(x_{i1}+x_{i3})   → MIN (minimize CO₂ from I & AI)
      f4 = Σ ρ_i·x_{i3} - σ_i·x_{i4} → MIN (minimize net security risk)

    Ràng buộc:
      G1:  Σ x ≤ TOTAL_BUDGET              (ngân sách tổng)
      G2:  Σ x ≥ 0.85·TOTAL_BUDGET         (tránh tiết kiệm quá mức)
      G3-8: Σ_j x_{ij} ≥ B_min_i / vùng   (sàn tối thiểu mỗi vùng)
      G9-12: Σ_i x_{ij} ≥ C_min_j / hạng mục (sàn tối thiểu mỗi hạng mục)
    """

    def __init__(self):
        super().__init__(
            n_var=24,
            n_obj=4,
            n_ieq_constr=12,
            xl=XL,
            xu=XU
        )
        self.beta = beta
        self.e    = e
        self.rho  = rho
        self.sig  = sig

        # Ngân sách tối thiểu mỗi vùng (tỷ lệ với quy mô GRDP)
        grdp = np.array([810, 3580, 1820, 420, 3050, 1409], dtype=float)
        grdp_share = grdp / grdp.sum()
        self.region_min = grdp_share * TOTAL_BUDGET * 0.50   # ít nhất 50% thị phần GRDP

        # Ngân sách tối thiểu mỗi hạng mục (đảm bảo cân bằng)
        self.cat_min = np.array([8000, 10000, 12000, 8000])  # [I, D, AI, H]

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape(N_REGIONS, N_CATEGORIES)   # (6, 4): [I, D, AI, H]

        # ─ Hàm mục tiêu ─────────────────────────────────
        # f1: GDP gain (maximize → negate)
        f1 = -(self.beta * X).sum()

        # f2: Bất bình đẳng vùng (MAD của tổng đầu tư mỗi vùng)
        sums = X.sum(axis=1)                     # (6,)
        f2   = np.abs(sums - sums.mean()).mean()

        # f3: Phát thải CO₂ từ hạ tầng (I) và AI data center
        f3 = (self.e * (X[:, 0] + X[:, 2])).sum()

        # f4: Rủi ro an ninh công nghệ ròng
        f4 = (self.rho * X[:, 2]).sum() - (self.sig * X[:, 3]).sum()

        out['F'] = [f1, f2, f3, f4]

        # ─ Ràng buộc (G ≤ 0) ────────────────────────────
        total = X.sum()
        g = []
        g.append(total - TOTAL_BUDGET)            # G1: tổng ≤ 60000
        g.append(0.85 * TOTAL_BUDGET - total)     # G2: tổng ≥ 51000

        # G3-G8: mỗi vùng nhận ít nhất region_min
        for i in range(N_REGIONS):
            g.append(self.region_min[i] - X[i, :].sum())

        # G9-G12: mỗi hạng mục nhận ít nhất cat_min
        for j in range(N_CATEGORIES):
            g.append(self.cat_min[j] - X[:, j].sum())

        out['G'] = g


# ── 3. CHẠY NSGA-II (Câu 7.4.1) ──────────────────────────────
print("\n" + "─" * 65)
print("🚀 Đang chạy NSGA-II (pop=100, gen=200) ...")
print("─" * 65)

problem   = VietnamDigitalProblem()
algorithm = NSGA2(pop_size=100)
termination = get_termination("n_gen", 200)

res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    verbose=False,
    save_history=False
)

print(f"✅ Hoàn tất! Tổng nghiệm Pareto: {len(res.F)}")


# ── 4. TRÍCH XUẤT TẬP PARETO (Câu 7.4.2) ─────────────────────
F = res.F.copy()   # shape (n_pareto, 4)
X_pareto = res.X.copy()

# Chuyển f1 về dương (GDP gain thực)
GDP_gain  = -F[:, 0]      # maximize
Gini_disp =  F[:, 1]      # minimize
CO2_emit  =  F[:, 2]      # minimize
Sec_risk  =  F[:, 3]      # minimize

print(f"\n📈 Thống kê tập Pareto ({len(F)} nghiệm):")
labels = ["GDP gain (nghìn tỷ VND)", "Bất bình đẳng (MAD)", "Phát thải CO₂", "Rủi ro an ninh"]
for i, lbl in enumerate(labels):
    col = -F[:, 0] if i == 0 else F[:, i]
    print(f"   {lbl:30s}: min={col.min():.3f}  max={col.max():.3f}  mean={col.mean():.3f}")


# ── 5. VẼ BIỂU ĐỒ (Câu 7.4.2) ────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f1117')
fig.suptitle("BÀI 7.4 – TỐI ƯU HÓA ĐẦU TƯ SỐ HÓA VIỆT NAM\nNSGA-II Pareto Front Analysis",
             fontsize=16, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

DARK   = '#0f1117'
ACCENT = '#00d4aa'
WARM   = '#ff6b6b'
BLUE   = '#4a9eff'
GOLD   = '#ffd700'
TEXT   = '#e0e0e0'

# ── 5A. Scatter 3D: f1, f2, f3 ──────────────────────────────
ax3d = fig.add_subplot(gs[0, 0], projection='3d')
ax3d.set_facecolor('#1a1d27')
scatter = ax3d.scatter(
    GDP_gain, Gini_disp, CO2_emit,
    c=Sec_risk, cmap='plasma',
    s=40, alpha=0.8, edgecolors='none'
)
cbar = plt.colorbar(scatter, ax=ax3d, shrink=0.6, pad=0.12)
cbar.set_label('Rủi ro an ninh (f₄)', color=TEXT, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT)

ax3d.set_xlabel('GDP gain (−f₁)', color=ACCENT, fontsize=9, labelpad=8)
ax3d.set_ylabel('Bất bình đẳng (f₂)', color=WARM, fontsize=9, labelpad=8)
ax3d.set_zlabel('Phát thải CO₂ (f₃)', color=BLUE, fontsize=9, labelpad=8)
ax3d.set_title('Pareto Front 3D\n(f₁, f₂, f₃; màu = f₄)', color=TEXT, fontsize=10, pad=10)
ax3d.tick_params(colors=TEXT, labelsize=7)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.grid(True, alpha=0.2)

# ── 5B. Pareto Front 2D: f1 vs f2 ───────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#1a1d27')
sc2 = ax2.scatter(GDP_gain, Gini_disp, c=CO2_emit, cmap='YlOrRd',
                  s=35, alpha=0.8, edgecolors='none')
cbar2 = plt.colorbar(sc2, ax=ax2)
cbar2.set_label('CO₂ (f₃)', color=TEXT, fontsize=8)
cbar2.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color=TEXT)
ax2.set_xlabel('GDP gain (−f₁)', color=ACCENT, fontsize=10)
ax2.set_ylabel('Bất bình đẳng (f₂)', color=WARM, fontsize=10)
ax2.set_title('Trade-off: Tăng trưởng vs Bao trùm\n(màu = phát thải CO₂)',
              color=TEXT, fontsize=10)
ax2.tick_params(colors=TEXT, labelsize=8)
for spine in ax2.spines.values():
    spine.set_edgecolor('#333')
ax2.grid(True, alpha=0.2, color='#333')

# ── 5C. Parallel Coordinates (4 mục tiêu) ───────────────────
ax_pc = fig.add_subplot(gs[1, :])
ax_pc.set_facecolor('#1a1d27')

# Chuẩn hóa 0-1 để vẽ parallel coordinates
F_plot = np.column_stack([GDP_gain, Gini_disp, CO2_emit, Sec_risk])
F_norm = (F_plot - F_plot.min(axis=0)) / (F_plot.max(axis=0) - F_plot.min(axis=0) + 1e-10)

obj_labels = ['f₁: GDP gain\n(tối đa hóa →)', 'f₂: Bất bình đẳng\n(← tối thiểu hóa)',
              'f₃: Phát thải CO₂\n(← tối thiểu hóa)', 'f₄: Rủi ro an ninh\n(← tối thiểu hóa)']

# Vẽ các đường Pareto (lấy mẫu 120 nghiệm để tránh chồng chéo)
n_show = min(120, len(F_norm))
idx_show = np.random.choice(len(F_norm), n_show, replace=False)
colors_pc = plt.cm.viridis(F_norm[idx_show, 0])

for k, i in enumerate(idx_show):
    ax_pc.plot(range(4), F_norm[i], color=colors_pc[k], alpha=0.35, linewidth=0.8)

# Highlight nghiệm tốt nhất f1 (GDP cao nhất)
best_f1_idx = np.argmax(GDP_gain)
ax_pc.plot(range(4), F_norm[best_f1_idx], color=GOLD, linewidth=3.0,
           alpha=1.0, label='Nghiệm GDP cao nhất', zorder=5)
ax_pc.scatter(range(4), F_norm[best_f1_idx], color=GOLD, s=100, zorder=6)

ax_pc.set_xticks(range(4))
ax_pc.set_xticklabels(obj_labels, color=TEXT, fontsize=9)
ax_pc.set_ylabel('Giá trị chuẩn hóa [0, 1]', color=TEXT, fontsize=10)
ax_pc.set_title('Parallel Coordinates – Toàn bộ tập Pareto (4 mục tiêu)',
                color=TEXT, fontsize=11)
ax_pc.tick_params(colors=TEXT, labelsize=8)
ax_pc.set_xlim(-0.1, 3.1)
ax_pc.set_ylim(-0.05, 1.05)
for spine in ax_pc.spines.values():
    spine.set_edgecolor('#444')
ax_pc.grid(True, alpha=0.15, color='#444', axis='x')
ax_pc.legend(facecolor='#1a1d27', edgecolor='#555', labelcolor=TEXT, fontsize=9)

# Thêm colorbar cho parallel coordinates
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(GDP_gain.min(), GDP_gain.max()))
sm.set_array([])
cbar_pc = plt.colorbar(sm, ax=ax_pc, orientation='vertical', shrink=0.8, pad=0.01)
cbar_pc.set_label('GDP gain (f₁)', color=TEXT, fontsize=8)
cbar_pc.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(plt.getp(cbar_pc.ax.axes, 'yticklabels'), color=TEXT)

plt.savefig('pareto_analysis.png',
            dpi=150, bbox_inches='tight', facecolor=DARK)
plt.show()
print("✅ Đã lưu: pareto_analysis.png")


# ── 6. TOPSIS (Câu 7.4.3) ────────────────────────────────────
print("\n" + "─" * 65)
print("📐 TOPSIS – Chọn nghiệm thỏa hiệp")
print("─" * 65)

# Trọng số chính sách (theo đề bài)
# Tăng trưởng(f1)=0.40, Bao trùm(f2)=0.25, Môi trường(f3)=0.20, An ninh(f4)=0.15
weights = np.array([0.40, 0.25, 0.20, 0.15])
print(f"   Trọng số: Tăng trưởng={weights[0]}, Bao trùm={weights[1]}, "
      f"Môi trường={weights[2]}, An ninh={weights[3]}")

# Ma trận quyết định: tất cả chuyển về "nhỏ hơn tốt hơn"
# f1 đã là -GDP_gain (minimize), còn f2, f3, f4 đã là minimize
decision_matrix = F.copy()   # shape (n, 4), tất cả minimize

# Bước 1: Chuẩn hóa vector (Euclidean norm)
norm_factors = np.sqrt((decision_matrix ** 2).sum(axis=0))
R = decision_matrix / norm_factors   # Normalized matrix

# Bước 2: Ma trận trọng số
V = R * weights   # Weighted normalized matrix

# Bước 3: Ideal best (A+) và Ideal worst (A-)
# Vì tất cả minimize → A+ = min, A- = max
A_plus  = V.min(axis=0)
A_minus = V.max(axis=0)

# Bước 4: Khoảng cách Euclidean đến A+ và A-
d_plus  = np.sqrt(((V - A_plus)  ** 2).sum(axis=1))
d_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

# Bước 5: Điểm TOPSIS (closeness coefficient)
C_score = d_minus / (d_plus + d_minus)   # Lớn hơn = tốt hơn

best_topsis_idx = np.argmax(C_score)

print(f"\n   ✅ Nghiệm TOPSIS tốt nhất: index = {best_topsis_idx}")
print(f"   ✅ TOPSIS score = {C_score[best_topsis_idx]:.4f}")

# Kết quả nghiệm TOPSIS
x_topsis = X_pareto[best_topsis_idx]
X_topsis_mat = x_topsis.reshape(N_REGIONS, N_CATEGORIES)

topsis_gdp  = -F[best_topsis_idx, 0]
topsis_gini =  F[best_topsis_idx, 1]
topsis_co2  =  F[best_topsis_idx, 2]
topsis_sec  =  F[best_topsis_idx, 3]

print(f"\n{'='*55}")
print(f"  NGHIỆM THỎA HIỆP (TOPSIS)")
print(f"{'='*55}")
print(f"  f₁ GDP gain    : {topsis_gdp:.4f}  (nghìn tỷ VND tương đương)")
print(f"  f₂ Bất bình đẳng: {topsis_gini:.2f}   (MAD tỷ VND)")
print(f"  f₃ Phát thải CO₂: {topsis_co2:.2f}")
print(f"  f₄ Rủi ro an ninh: {topsis_sec:.2f}")
print(f"{'='*55}")
print(f"\n  Phân bổ ngân sách (tỷ VND) – TOPSIS:")
print(f"  {'Vùng':<22} {'Hạ tầng':>10} {'Số hóa':>10} {'AI':>10} {'Nhân lực':>10} {'Tổng':>10}")
print(f"  {'-'*72}")
for i, rname in enumerate(regions["name"]):
    row = X_topsis_mat[i]
    print(f"  {rname:<22} {row[0]:>10.0f} {row[1]:>10.0f} {row[2]:>10.0f} {row[3]:>10.0f} {row.sum():>10.0f}")
print(f"  {'-'*72}")
totals = X_topsis_mat.sum(axis=0)
print(f"  {'TỔNG':<22} {totals[0]:>10.0f} {totals[1]:>10.0f} {totals[2]:>10.0f} {totals[3]:>10.0f} {totals.sum():>10.0f}")


# ── 7. PHÂN TÍCH CHI PHÍ CƠ HỘI (Câu 7.4.4) ─────────────────
print("\n" + "─" * 65)
print("💸 PHÂN TÍCH CHI PHÍ CƠ HỘI")
print("─" * 65)

# Nghiệm GDP cao nhất
best_growth_idx = np.argmax(GDP_gain)
x_growth        = X_pareto[best_growth_idx]
X_growth_mat    = x_growth.reshape(N_REGIONS, N_CATEGORIES)

growth_gdp  = GDP_gain[best_growth_idx]
growth_gini = Gini_disp[best_growth_idx]
growth_co2  = CO2_emit[best_growth_idx]
growth_sec  = Sec_risk[best_growth_idx]

print(f"\n  NGHIỆM TĂNG TRƯỞNG CAO NHẤT:")
print(f"    f₁ GDP gain     : {growth_gdp:.4f}")
print(f"    f₂ Bất bình đẳng : {growth_gini:.2f}")
print(f"    f₃ Phát thải CO₂ : {growth_co2:.2f}")
print(f"    f₄ Rủi ro an ninh: {growth_sec:.2f}")

print(f"\n  NGHIỆM THỎA HIỆP (TOPSIS):")
print(f"    f₁ GDP gain     : {topsis_gdp:.4f}")
print(f"    f₂ Bất bình đẳng : {topsis_gini:.2f}")
print(f"    f₃ Phát thải CO₂ : {topsis_co2:.2f}")
print(f"    f₄ Rủi ro an ninh: {topsis_sec:.2f}")

# Chi phí cơ hội: so sánh giá trị của "nghiệm GDP cao nhất" vs "nghiệm thỏa hiệp"
# Nếu tập Pareto đã đủ tốt, growth_gdp ≥ topsis_gdp (GDP tốt hơn nhưng hy sinh các mục tiêu khác)

delta_gdp  = (growth_gdp  - topsis_gdp)  / abs(topsis_gdp)  * 100 if topsis_gdp  != 0 else 0
delta_gini = (growth_gini - topsis_gini) / abs(topsis_gini) * 100 if topsis_gini != 0 else 0
delta_co2  = (growth_co2  - topsis_co2)  / abs(topsis_co2)  * 100 if topsis_co2  != 0 else 0
delta_sec  = (growth_sec  - topsis_sec)  / abs(topsis_sec)  * 100 if topsis_sec  != 0 else 0

print(f"\n{'='*60}")
print(f"  CHI PHÍ CƠ HỘI – Nghiệm GDP Cao Nhất vs Thỏa Hiệp")
print(f"{'='*60}")
print(f"  Lợi ích tăng trưởng thêm  : {delta_gdp:+.2f}%")
print(f"  Hy sinh bao trùm (Gini)   : {delta_gini:+.2f}%  {'(tệ hơn)' if delta_gini > 0 else '(tốt hơn)'}")
print(f"  Hy sinh môi trường (CO₂)  : {delta_co2:+.2f}%  {'(tệ hơn)' if delta_co2  > 0 else '(tốt hơn)'}")
print(f"  Thay đổi an ninh          : {delta_sec:+.2f}%  {'(tệ hơn)' if delta_sec  > 0 else '(tốt hơn)'}")
print(f"{'='*60}")

# Diễn giải bằng lời
print(f"""
  📝 DIỄN GIẢI:
  Để đạt GDP cao nhất (tăng thêm {delta_gdp:.1f}%), nghiệm tăng trưởng
  phải trả giá:
    • Bao trùm xã hội: {'tệ hơn' if delta_gini > 0 else 'tốt hơn'} {abs(delta_gini):.1f}%
      (bất bình đẳng vùng tăng)
    • Môi trường/CO₂:  {'tệ hơn' if delta_co2  > 0 else 'tốt hơn'} {abs(delta_co2):.1f}%
      (phát thải từ hạ tầng và AI data center)
    • Rủi ro an ninh:  {'tệ hơn' if delta_sec  > 0 else 'tốt hơn'} {abs(delta_sec):.1f}%

  → Nghiệm TOPSIS (trọng số chính sách) cân bằng tốt hơn
    khi ưu tiên tăng trưởng 40%, bao trùm 25%,
    môi trường 20%, an ninh 15%.
""")


# ── 8. BIỂU ĐỒ CHI PHÍ CƠ HỘI ───────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.patch.set_facecolor(DARK)
fig2.suptitle("Chi Phí Cơ Hội – Nghiệm GDP Cao Nhất vs Nghiệm Thỏa Hiệp (TOPSIS)",
              color=TEXT, fontsize=13, fontweight='bold')

# ── 8A. TOPSIS score distribution ──
ax = axes[0]
ax.set_facecolor('#1a1d27')
ax.hist(C_score, bins=30, color=BLUE, alpha=0.8, edgecolor='white', linewidth=0.5)
ax.axvline(C_score[best_topsis_idx], color=GOLD, linewidth=2.5, linestyle='--',
           label=f'TOPSIS best={C_score[best_topsis_idx]:.3f}')
ax.axvline(C_score[best_growth_idx], color=WARM, linewidth=2.5, linestyle='-.',
           label=f'GDP best={C_score[best_growth_idx]:.3f}')
ax.set_xlabel('TOPSIS Score', color=TEXT)
ax.set_ylabel('Số nghiệm', color=TEXT)
ax.set_title('Phân phối TOPSIS Score', color=TEXT, fontsize=11)
ax.tick_params(colors=TEXT)
ax.legend(facecolor='#1a1d27', edgecolor='#555', labelcolor=TEXT)
for spine in ax.spines.values(): spine.set_edgecolor('#444')

# ── 8B. Spider/Radar chart so sánh 2 nghiệm ──
ax_radar = axes[1]
ax_radar.set_facecolor('#1a1d27')

# Chuẩn hóa 4 mục tiêu theo tập Pareto
mins = np.array([GDP_gain.min(), Gini_disp.min(), CO2_emit.min(), Sec_risk.min()])
maxs = np.array([GDP_gain.max(), Gini_disp.max(), CO2_emit.max(), Sec_risk.max()])

def normalize_for_radar(vals, mins, maxs):
    # f1 (GDP): larger is better → normalized as (val-min)/(max-min)
    # f2,f3,f4: smaller is better → normalized as 1-(val-min)/(max-min)
    norms = np.zeros(4)
    norms[0] = (vals[0] - mins[0]) / (maxs[0] - mins[0] + 1e-10)          # GDP: bigger=better
    norms[1] = 1 - (vals[1] - mins[1]) / (maxs[1] - mins[1] + 1e-10)      # Gini: smaller=better
    norms[2] = 1 - (vals[2] - mins[2]) / (maxs[2] - mins[2] + 1e-10)      # CO2: smaller=better
    norms[3] = 1 - (vals[3] - mins[3]) / (maxs[3] - mins[3] + 1e-10)      # Sec: smaller=better
    return norms

topsis_norm = normalize_for_radar([topsis_gdp, topsis_gini, topsis_co2, topsis_sec], mins, maxs)
growth_norm = normalize_for_radar([growth_gdp, growth_gini, growth_co2, growth_sec], mins, maxs)

categories  = ['Tăng trưởng\n(f₁)', 'Bao trùm\n(−f₂)', 'Môi trường\n(−f₃)', 'An ninh\n(−f₄)']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

# Vẽ radar chart thủ công trên ax thường
for idx, (vals, color, label, ls) in enumerate([
    (topsis_norm, GOLD, 'TOPSIS (thỏa hiệp)', '-'),
    (growth_norm, WARM, 'GDP cao nhất', '--'),
]):
    # Chuyển thành tọa độ Cartesian
    x_cart = [v * np.cos(a) for v, a in zip(vals, angles)]
    y_cart = [v * np.sin(a) for v, a in zip(vals, angles)]
    x_cart.append(x_cart[0])
    y_cart.append(y_cart[0])
    ax_radar.plot(x_cart, y_cart, color=color, linewidth=2.5,
                  linestyle=ls, label=label, alpha=0.9)
    ax_radar.fill(x_cart[:-1], y_cart[:-1], color=color, alpha=0.12)

# Vẽ lưới và trục
for angle, cat in zip(angles, categories):
    ax_radar.plot([0, np.cos(angle)], [0, np.sin(angle)],
                  color='#444', linewidth=0.8, linestyle=':')
    ax_radar.text(1.18 * np.cos(angle), 1.18 * np.sin(angle), cat,
                  ha='center', va='center', color=TEXT, fontsize=8)

for r in [0.25, 0.5, 0.75, 1.0]:
    circle_x = [r * np.cos(a) for a in np.linspace(0, 2*np.pi, 100)]
    circle_y = [r * np.sin(a) for a in np.linspace(0, 2*np.pi, 100)]
    ax_radar.plot(circle_x, circle_y, color='#333', linewidth=0.6, linestyle='-')

ax_radar.set_xlim(-1.4, 1.4)
ax_radar.set_ylim(-1.4, 1.4)
ax_radar.set_aspect('equal')
ax_radar.axis('off')
ax_radar.set_title('So sánh 2 nghiệm\n(Radar chart, 0=tệ, 1=tốt)', color=TEXT, fontsize=10)
ax_radar.legend(loc='lower right', facecolor='#1a1d27', edgecolor='#555',
                labelcolor=TEXT, fontsize=8)

# ── 8C. Bar chart phân bổ ngân sách ──
ax_bar = axes[2]
ax_bar.set_facecolor('#1a1d27')

cat_names = ['Hạ tầng\n(I)', 'Số hóa\n(D)', 'AI', 'Nhân lực\n(H)']
topsis_cat = X_topsis_mat.sum(axis=0)
growth_cat = X_growth_mat.sum(axis=0)

x_pos = np.arange(4)
w = 0.38
bars1 = ax_bar.bar(x_pos - w/2, topsis_cat, w, color=GOLD, alpha=0.85, label='TOPSIS', edgecolor='white', linewidth=0.5)
bars2 = ax_bar.bar(x_pos + w/2, growth_cat, w, color=WARM, alpha=0.85, label='GDP cao nhất', edgecolor='white', linewidth=0.5)

for bar in bars1:
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{bar.get_height():.0f}', ha='center', va='bottom', color=GOLD, fontsize=7)
for bar in bars2:
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{bar.get_height():.0f}', ha='center', va='bottom', color=WARM, fontsize=7)

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(cat_names, color=TEXT, fontsize=9)
ax_bar.set_ylabel('Tổng đầu tư (tỷ VND)', color=TEXT)
ax_bar.set_title('Phân bổ ngân sách theo hạng mục\n(Tổng 2 nghiệm)', color=TEXT, fontsize=10)
ax_bar.tick_params(colors=TEXT)
ax_bar.legend(facecolor='#1a1d27', edgecolor='#555', labelcolor=TEXT, fontsize=9)
for spine in ax_bar.spines.values(): spine.set_edgecolor('#444')
ax_bar.grid(True, alpha=0.2, axis='y', color='#444')

plt.tight_layout()
plt.savefig('opportunity_cost.png',
            dpi=150, bbox_inches='tight', facecolor=DARK)
plt.show()
print("✅ Đã lưu: opportunity_cost.png")


# ── 9. BẢNG TÓM TẮT CUỐI ─────────────────────────────────────
print("\n" + "=" * 65)
print("  TÓM TẮT KẾT QUẢ BÀI 7.4")
print("=" * 65)
summary = pd.DataFrame({
    'Chỉ tiêu': ['GDP gain', 'Bất bình đẳng (Gini MAD)', 'Phát thải CO₂', 'Rủi ro an ninh'],
    'TOPSIS (thỏa hiệp)': [f'{topsis_gdp:.4f}', f'{topsis_gini:.2f}',
                            f'{topsis_co2:.2f}', f'{topsis_sec:.2f}'],
    'GDP cao nhất':       [f'{growth_gdp:.4f}', f'{growth_gini:.2f}',
                            f'{growth_co2:.2f}', f'{growth_sec:.2f}'],
    'Chi phí cơ hội (%)': [f'{delta_gdp:+.1f}%', f'{delta_gini:+.1f}%',
                            f'{delta_co2:+.1f}%', f'{delta_sec:+.1f}%'],
})
print(summary.to_string(index=False))
print("\n✅ Hoàn tất tất cả 4 câu hỏi bài 7.4!")
print("   • pareto_analysis.png   – Câu 7.4.2 (3D scatter + parallel coordinates)")
print("   • opportunity_cost.png  – Câu 7.4.3/7.4.4 (TOPSIS + chi phí cơ hội)")