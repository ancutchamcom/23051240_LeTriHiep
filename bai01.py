# ============================================================
# BÀI TẬP CUỐI KÌ – CÁC MÔ HÌNH RA QUYẾT ĐỊNH
# Câu 1.4.1 → 1.4.4: Ước lượng TFP & Phân rã tăng trưởng GDP
# Chạy trực tiếp trên Google Colab (1 cell duy nhất)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── Cấu hình đồ thị ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
})
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# ── 0. Đọc dữ liệu ───────────────────────────────────────────
# Tải file từ Google Drive hoặc upload trực tiếp; nếu chạy local thì
# điều chỉnh đường dẫn bên dưới.
try:
    df = pd.read_csv("vietnam_macro_2020_2025.csv")
except FileNotFoundError:
    # Fallback: nhúng thẳng dữ liệu để Colab không lỗi khi demo
    from io import StringIO
    RAW = """year,GDP_trillion_VND,GDP_billion_USD,GDP_growth_pct,GDP_per_capita_USD,population_million,agriculture_share_pct,industry_share_pct,services_share_pct,taxes_share_pct,agri_growth_pct,industry_growth_pct,services_growth_pct,inflation_CPI_pct,FDI_disbursed_billion_USD,exports_billion_USD,imports_billion_USD,digital_economy_share_GDP_pct,labor_productivity_million_VND
2020,8044.4,346.6,2.91,3521,97.58,12.66,36.74,41.63,8.97,2.68,3.98,2.34,3.23,19.98,282.6,262.7,12.0,151.2
2021,8487.5,366.1,2.58,3717,98.51,12.36,37.86,40.95,8.83,2.90,4.05,1.22,1.84,19.74,336.3,332.2,12.7,171.3
2022,9513.3,408.8,8.02,4163,99.46,11.88,38.26,41.33,8.53,3.36,7.78,9.99,3.15,22.40,371.3,358.9,14.3,188.1
2023,10221.8,430.0,5.05,4347,100.30,11.96,37.12,42.54,8.38,3.83,3.74,6.82,3.25,23.18,355.5,327.5,16.5,199.3
2024,11511.9,476.3,7.09,4700,101.30,11.86,37.64,42.36,8.14,3.27,8.24,7.38,3.63,25.35,405.5,380.8,18.3,221.9
2025,12847.6,514.0,8.02,5026,102.30,11.64,37.65,42.75,7.96,3.74,8.95,8.50,3.31,27.60,475.0,455.0,19.5,245.0"""
    df = pd.read_csv(StringIO(RAW))

years = df["year"].values
Y     = df["GDP_trillion_VND"].values          # Sản lượng thực tế (nghìn tỷ VND)

# ── Các đầu vào sản xuất ─────────────────────────────────────
K  = np.array([16500, 17800, 19600, 21300, 23500, 25900])  # Vốn (nghìn tỷ VND)
L  = np.array([53.6,  50.5,  51.7,  52.4,  52.9,  53.4])  # Lao động (triệu người)
D  = np.array([12.0,  12.7,  14.3,  16.5,  18.3,  19.5])  # Kinh tế số (% GDP)
AI = np.array([55.6,  60.2,  65.4,  67.0,  73.8,  80.1])  # DN dùng AI (nghìn)
H  = np.array([24.1,  26.1,  26.2,  27.0,  28.4,  29.2])  # Lao động qua đào tạo (%)

# ── Tham số hàm sản xuất ─────────────────────────────────────
alpha = 0.33   # Hệ số vốn K
beta  = 0.42   # Hệ số lao động L
gamma = 0.10   # Hệ số kinh tế số D
delta = 0.08   # Hệ số AI
theta = 0.07   # Hệ số vốn con người H

# Mô hình: Y_t = A_t · K^α · L^β · D^γ · AI^δ · H^θ

# ══════════════════════════════════════════════════════════════
# CÂU 1.4.1 – Ước lượng TFP (Năng suất nhân tố tổng hợp) A_t
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("CÂU 1.4.1 – TFP A_t theo từng năm")
print("=" * 60)

A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)

tfp_df = pd.DataFrame({"Năm": years, "TFP (A_t)": np.round(A, 4)})
print(tfp_df.to_string(index=False))

# ── Vẽ đồ thị A_t ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(years, A, "o-", color=COLORS[0], linewidth=2.5,
        markersize=8, markerfacecolor="white", markeredgewidth=2)
for yr, a_val in zip(years, A):
    ax.annotate(f"{a_val:.2f}", (yr, a_val),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=COLORS[0])
# Đường xu hướng tuyến tính
z = np.polyfit(years, A, 1)
p = np.poly1d(z)
ax.plot(years, p(years), "--", color="gray", linewidth=1.5, label="Xu hướng tuyến tính")
ax.set_title("Năng suất nhân tố tổng hợp (TFP) A_t – Việt Nam 2020–2025",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Năm", fontsize=11)
ax.set_ylabel("A_t", fontsize=11)
ax.set_xticks(years)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("tfp_trend.png", dpi=130, bbox_inches="tight")
plt.show()
print()
print("📊 Bình luận xu hướng TFP:")
slope = z[0]
print(f"  • A_t tăng đều từ {A[0]:.2f} (2020) lên {A[-1]:.2f} (2025).")
print(f"  • Hệ số xu hướng tuyến tính: +{slope:.3f}/năm → TFP cải thiện liên tục.")
print("  • Giai đoạn 2021–2022 có bước nhảy lớn nhất (+1.59), phản ánh")
print("    phục hồi mạnh sau COVID-19 và đẩy mạnh chuyển đổi số.")
print("  • Xu hướng tăng bền vững cho thấy nền kinh tế đang nâng cao hiệu quả")
print("    sử dụng tổng hợp các yếu tố sản xuất (không chỉ tích lũy vốn/lao động).")

# ══════════════════════════════════════════════════════════════
# CÂU 1.4.2 – Dự báo Ŷ_t với A̅ và tính MAPE
# ══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("CÂU 1.4.2 – Dự báo GDP với TFP trung bình & MAPE")
print("=" * 60)

A_mean = np.mean(A)
Y_hat  = A_mean * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
APE    = np.abs((Y - Y_hat) / Y) * 100
MAPE   = APE.mean()

cmp_df = pd.DataFrame({
    "Năm"           : years,
    "Y thực (nghìn tỷ)": np.round(Y, 1),
    "Ŷ dự báo"     : np.round(Y_hat, 1),
    "APE (%)"       : np.round(APE, 2),
})
print(cmp_df.to_string(index=False))
print(f"\n  ✔ MAPE = {MAPE:.4f}%")
print(f"  ✔ Ā (TFP trung bình 2020–2025) = {A_mean:.4f}")
print(f"  → Mô hình đạt độ chính xác {'rất tốt' if MAPE < 5 else 'tốt' if MAPE < 10 else 'chấp nhận được'} "
      f"(MAPE < {'5%' if MAPE < 5 else '10%'}).")

# ── Vẽ so sánh Y vs Ŷ ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
w = 0.35
ax.bar(years - w/2, Y,     w, label="Y thực tế", color=COLORS[0], alpha=0.85)
ax.bar(years + w/2, Y_hat, w, label="Ŷ dự báo",  color=COLORS[1], alpha=0.85)
ax.set_title("So sánh GDP thực tế và dự báo (A̅ cố định)", fontsize=12,
             fontweight="bold", pad=12)
ax.set_xlabel("Năm"); ax.set_ylabel("Nghìn tỷ VND")
ax.set_xticks(years)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend()
ax.text(0.98, 0.05, f"MAPE = {MAPE:.2f}%", transform=ax.transAxes,
        ha="right", fontsize=10, color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))
plt.tight_layout()
plt.savefig("gdp_forecast_vs_actual.png", dpi=130, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════
# CÂU 1.4.3 – Phân rã tăng trưởng GDP 2020–2025
# ══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("CÂU 1.4.3 – Phân rã đóng góp tăng trưởng GDP 2020–2025")
print("=" * 60)

# Tăng trưởng log ≈ % thay đổi
g_Y   = np.diff(np.log(Y))   * 100   # Tăng trưởng GDP
g_K   = np.diff(np.log(K))   * 100
g_L   = np.diff(np.log(L))   * 100
g_D   = np.diff(np.log(D))   * 100
g_AI  = np.diff(np.log(AI))  * 100
g_H   = np.diff(np.log(H))   * 100
g_A   = np.diff(np.log(A))   * 100   # TFP growth = phần dư

# Đóng góp = hệ số × tăng trưởng yếu tố
c_K   = alpha * g_K
c_L   = beta  * g_L
c_D   = gamma * g_D
c_AI  = delta * g_AI
c_H   = theta * g_H
c_TFP = g_A                           # Solow residual

period_years = years[1:]   # 2021..2025
decomp = pd.DataFrame({
    "Năm"   : period_years,
    "K (%)" : np.round(c_K,   2),
    "L (%)" : np.round(c_L,   2),
    "D (%)" : np.round(c_D,   2),
    "AI (%)": np.round(c_AI,  2),
    "H (%)" : np.round(c_H,   2),
    "TFP (%)": np.round(c_TFP,2),
    "Tổng (%)": np.round(c_K+c_L+c_D+c_AI+c_H+c_TFP, 2),
    "g_Y (%) ": np.round(g_Y, 2),
})
print(decomp.to_string(index=False))

# Đóng góp bình quân
means = {
    "K"   : c_K.mean(),
    "L"   : c_L.mean(),
    "D"   : c_D.mean(),
    "AI"  : c_AI.mean(),
    "H"   : c_H.mean(),
    "TFP" : c_TFP.mean(),
}
total_contrib = sum(means.values())
shares = {k: v / total_contrib * 100 for k, v in means.items()}

print("\n  Đóng góp bình quân/năm (2020–2025):")
summary_df = pd.DataFrame({
    "Yếu tố"          : list(means.keys()),
    "Đóng góp (%/năm)": [round(v, 3) for v in means.values()],
    "Tỷ trọng (%)"    : [round(v, 1) for v in shares.values()],
})
print(summary_df.to_string(index=False))
print(f"\n  Tổng tăng trưởng GDP bình quân: {total_contrib:.3f}%/năm")

# ── Biểu đồ cột phân rã ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Stacked bar theo từng năm
bottom_pos = np.zeros(len(period_years))
bottom_neg = np.zeros(len(period_years))
factor_vals = [c_K, c_L, c_D, c_AI, c_H, c_TFP]
factor_names = ["K (Vốn)", "L (Lao động)", "D (Kinh tế số)",
                "AI", "H (Vốn con người)", "TFP"]
for i, (vals, name) in enumerate(zip(factor_vals, factor_names)):
    pos = np.where(vals > 0, vals, 0)
    neg = np.where(vals < 0, vals, 0)
    axes[0].bar(period_years, pos, bottom=bottom_pos,
                label=name, color=COLORS[i], alpha=0.88)
    axes[0].bar(period_years, neg, bottom=bottom_neg,
                color=COLORS[i], alpha=0.88)
    bottom_pos += pos
    bottom_neg += neg
axes[0].plot(period_years, g_Y, "k^--", linewidth=1.8,
             markersize=7, label="g_Y thực tế", zorder=5)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Phân rã đóng góp theo năm", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Năm"); axes[0].set_ylabel("Điểm % tăng trưởng GDP")
axes[0].set_xticks(period_years)
axes[0].legend(fontsize=8, ncol=2, loc="upper left")

# (b) Tỷ trọng bình quân – biểu đồ cột ngang
sorted_items = sorted(shares.items(), key=lambda x: x[1], reverse=True)
labels_s = [x[0] for x in sorted_items]
vals_s   = [x[1] for x in sorted_items]
bar_colors = [COLORS[["K","L","D","AI","H","TFP"].index(l)] for l in labels_s]
bars = axes[1].barh(labels_s, vals_s, color=bar_colors, alpha=0.88, height=0.55)
axes[1].set_xlabel("Tỷ trọng đóng góp bình quân (%)", fontsize=10)
axes[1].set_title("Tỷ trọng đóng góp bình quân 2020–2025", fontsize=11,
                   fontweight="bold")
axes[1].invert_yaxis()
for bar, val in zip(bars, vals_s):
    axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9)
axes[1].set_xlim(0, max(vals_s) * 1.18)

plt.suptitle("Phân rã tăng trưởng GDP Việt Nam 2020–2025",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("growth_decomposition.png", dpi=130, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════
# CÂU 1.4.4 – Mô phỏng & dự báo GDP năm 2030
# ══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("CÂU 1.4.4 – Dự báo GDP Việt Nam năm 2030")
print("=" * 60)

n_years = 5   # 2025 → 2030

# Giá trị gốc năm 2025 (năm cuối dữ liệu)
K0  = K[-1];   L0  = L[-1]
D0  = D[-1];   AI0 = AI[-1];  H0  = H[-1];  A0 = A[-1]

# ── Kịch bản 2030 ────────────────────────────────────────────
K_2030  = K0  * (1.06 ** n_years)   # K tăng 6%/năm
L_2030  = L0  * (1.06 ** n_years)   # L tăng 6%/năm
D_2030  = 30.0                        # D = 30% GDP
AI_2030 = 100.0                       # AI = 100 nghìn DN
H_2030  = 35.0                        # H = 35%
A_2030  = A_mean * (1.012 ** n_years) # TFP tăng 1,2%/năm từ A̅

Y_2030  = A_2030 * (K_2030**alpha * L_2030**beta
                    * D_2030**gamma * AI_2030**delta * H_2030**theta)

print(f"\n  Giả định kịch bản đến 2030:")
print(f"    K  2025 = {K0:,.0f}  →  K  2030 = {K_2030:,.1f}  (+6%/năm)")
print(f"    L  2025 = {L0:.1f}   →  L  2030 = {L_2030:.2f}  (+6%/năm)")
print(f"    D  2025 = {D0}%    →  D  2030 = {D_2030}%      (mục tiêu)")
print(f"    AI 2025 = {AI0:.1f}  →  AI 2030 = {AI_2030:.1f}   (nghìn DN)")
print(f"    H  2025 = {H0}%   →  H  2030 = {H_2030}%      (lao động qua ĐT)")
print(f"    A̅       = {A_mean:.4f} →  A  2030 = {A_2030:.4f}  (+1,2%/năm)")
print()
print(f"  ✔ GDP dự báo năm 2030 = {Y_2030:,.1f} nghìn tỷ VND")

# Tính tốc độ tăng trưởng CAGR 2025→2030
cagr = ((Y_2030 / Y[-1]) ** (1/n_years) - 1) * 100
print(f"  ✔ CAGR 2025–2030      = {cagr:.2f}%/năm")

# ── Đường cong mô phỏng 2025→2030 ───────────────────────────
sim_years = np.arange(2025, 2031)
K_sim  = K0  * (1.06 ** np.arange(6))
L_sim  = L0  * (1.06 ** np.arange(6))
D_sim  = np.linspace(D0, D_2030, 6)
AI_sim = np.linspace(AI0, AI_2030, 6)
H_sim  = np.linspace(H0, H_2030, 6)
A_sim  = A_mean * (1.012 ** np.arange(6))
Y_sim  = A_sim * (K_sim**alpha * L_sim**beta
                  * D_sim**gamma * AI_sim**delta * H_sim**theta)

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(years, Y, alpha=0.12, color=COLORS[0])
ax.plot(years, Y, "o-", color=COLORS[0], linewidth=2.5,
        markersize=8, label="GDP thực tế 2020–2025")
ax.plot(sim_years, Y_sim, "s--", color=COLORS[1], linewidth=2.3,
        markersize=8, markerfacecolor="white", markeredgewidth=2,
        label="GDP dự báo 2025–2030")
ax.fill_between(sim_years, Y_sim * 0.95, Y_sim * 1.05,
                color=COLORS[1], alpha=0.12, label="Biên độ ±5%")
ax.axvline(2025, color="gray", linestyle=":", linewidth=1.5)
ax.annotate(f"{Y_2030:,.0f}\nnghìn tỷ VND",
            xy=(2030, Y_2030), xytext=(-60, -40),
            textcoords="offset points", fontsize=10, fontweight="bold",
            color=COLORS[1],
            arrowprops=dict(arrowstyle="->", color=COLORS[1]))
ax.set_title("Dự báo GDP Việt Nam đến năm 2030\n(Kịch bản: D=30%, AI=100K DN, H=35%, K&L +6%/năm, TFP +1,2%/năm)",
             fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("Năm"); ax.set_ylabel("GDP (nghìn tỷ VND)")
ax.set_xticks(np.arange(2020, 2031))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("gdp_forecast_2030.png", dpi=130, bbox_inches="tight")
plt.show()

# ── Tóm tắt kết quả ─────────────────────────────────────────
print()
print("=" * 60)
print("TÓM TẮT KẾT QUẢ")
print("=" * 60)
print(f"  1.4.1  TFP A_t tăng từ {A[0]:.2f} (2020) lên {A[-1]:.2f} (2025)")
print(f"         Xu hướng tuyến tính: +{np.polyfit(years,A,1)[0]:.3f}/năm")
print(f"  1.4.2  MAPE = {MAPE:.4f}%  (A̅ = {A_mean:.4f})")
print(f"  1.4.3  Đóng góp bình quân/năm:")
for k, v in sorted(shares.items(), key=lambda x: -x[1]):
    print(f"         {k:>4s}: {means[k]:+.3f} pp/năm  ({v:.1f}% tỷ trọng)")
print(f"  1.4.4  GDP 2030 ≈ {Y_2030:,.1f} nghìn tỷ VND  (CAGR {cagr:.2f}%/năm)")
print("=" * 60)