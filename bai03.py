# ================================================================
#  BÀI TOÁN 3.4 – MÔ HÌNH RA QUYẾT ĐỊNH
#  Phân tích ưu tiên đầu tư ngành kinh tế Việt Nam 2024
#  Chạy trên Google Colab – tất cả trong 1 cell
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from io import StringIO

# ── Dữ liệu nhúng sẵn (thay bằng pd.read_csv nếu có file trên Drive) ──
RAW = """sector_id,sector_name_en,gdp_share_2024_pct,growth_rate_2024_pct,labor_million,labor_share_pct,export_billion_USD,digital_index_0_100,ai_readiness_0_100,fdi_attraction_billion_USD,spillover_coef_0_1,automation_risk_pct,rd_intensity_pct
1,Agriculture-Forestry-Fishery,11.86,3.27,13.2,26.5,40.5,28,15,2.1,0.35,18,0.15
2,Manufacturing,24.1,9.64,11.5,23.1,290.9,68,55,18.6,0.78,42,0.62
3,Construction,7.04,7.45,4.8,9.6,2.5,35,20,0.8,0.42,25,0.18
4,Mining,3.36,-1.2,0.3,0.6,8.2,50,30,0.5,0.3,55,0.22
5,Wholesale-Retail,9.85,7.1,7.8,15.7,5.5,72,48,3.2,0.55,38,0.1
6,Finance-Banking-Insurance,5.12,7.36,0.55,1.1,1.2,82,72,1.8,0.85,52,0.45
7,Logistics-Transport-Warehousing,5.45,9.93,1.95,3.9,3.1,65,42,2.4,0.72,35,0.2
8,Information-Communication-IT,3.85,7.85,0.62,1.2,178,92,88,4.6,0.92,28,1.2
9,Education-Training,3.85,6.42,2.15,4.3,0,55,38,0.4,0.65,22,0.3
10,Healthcare,2.85,6.85,0.75,1.5,0,58,45,1.2,0.6,18,0.55"""

VI_NAMES = {
    "Agriculture-Forestry-Fishery":   "Nông-Lâm-Thủy sản",
    "Manufacturing":                   "Công nghiệp chế biến",
    "Construction":                    "Xây dựng",
    "Mining":                          "Khai khoáng",
    "Wholesale-Retail":                "Bán buôn-Bán lẻ",
    "Finance-Banking-Insurance":       "Tài chính-Ngân hàng",
    "Logistics-Transport-Warehousing": "Logistics-Vận tải",
    "Information-Communication-IT":    "CNTT-Truyền thông",
    "Education-Training":              "Giáo dục-Đào tạo",
    "Healthcare":                      "Y tế",
}

# ── Đọc dữ liệu ──────────────────────────────────────────────────
df = pd.read_csv(StringIO(RAW))
# Nếu chạy trên Colab có file riêng, dùng:
# df = pd.read_csv('vietnam_sectors_2024.csv')

df["sector_name_vi"] = df["sector_name_en"].map(VI_NAMES)

# ── Định nghĩa các cột chỉ tiêu ──────────────────────────────────
COLS_GOOD = [
    "growth_rate_2024_pct",   # C1 – Tăng trưởng       (a1=0.15)
    "gdp_share_2024_pct",     # C2 – GDP Share          (a2=0.15)
    "spillover_coef_0_1",     # C3 – Lan tỏa            (a3=0.20)
    "export_billion_USD",     # C4 – Xuất khẩu          (a4=0.15)
    "labor_million",          # C5 – Việc làm           (a5=0.10)
    "ai_readiness_0_100",     # C6 – AI Readiness       (a6=0.20)
]
COL_BAD = "automation_risk_pct"   # C7 – Rủi ro tự động hóa  (a7=0.15, trừ đi)

COL_LABELS = [
    "C1 Tăng trưởng", "C2 GDP Share", "C3 Lan tỏa",
    "C4 Xuất khẩu",   "C5 Việc làm", "C6 AI Ready", "C7 Risk↓",
]

SEP = "\n" + "═" * 65 + "\n"

# ── Hàm chuẩn hóa ────────────────────────────────────────────────
def norm_good(s):
    """Min-max, chiều tốt: cao = tốt."""
    return (s - s.min()) / (s.max() - s.min())

def norm_bad(s):
    """Min-max đảo dấu, chiều xấu: cao = tốt (rủi ro thấp)."""
    return (s.max() - s) / (s.max() - s.min())

# ── Hàm tính Priority ─────────────────────────────────────────────
def calc_priority(df_in, w_goods, w_risk):
    """
    Priority_i = sum(a_j * X_ij) - a7 * X_i7
    Trong đó X_i7 là chuẩn hóa bình thường (chưa đảo) rồi trừ đi.
    """
    Xg = df_in[COLS_GOOD].apply(norm_good).values        # shape (10,6)
    Xb = norm_good(df_in[COL_BAD]).values                 # chưa đảo, sẽ trừ
    return Xg @ np.array(w_goods) - w_risk * Xb          # công thức đề bài


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CÂU 3.4.1 – Chuẩn hóa min-max và in ma trận                   ║
# ╚══════════════════════════════════════════════════════════════════╝
print(SEP + "CÂU 3.4.1 – MA TRẬN CHUẨN HÓA MIN-MAX (7 CHỈ TIÊU)" + SEP)

norm_good_df = df[COLS_GOOD].apply(norm_good)
norm_bad_col = norm_bad(df[COL_BAD])   # đảo dấu để hiển thị "càng cao càng tốt"

norm_matrix = norm_good_df.copy()
norm_matrix["C7_risk_inv"] = norm_bad_col
norm_matrix.columns = COL_LABELS
norm_matrix.index   = df["sector_name_vi"]

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 130)
print(norm_matrix.to_string())
print("\n* Cột C7 Risk↓ đã đảo dấu: giá trị cao = rủi ro THẤP (tốt hơn).")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CÂU 3.4.2 – Tính Priority với bộ trọng số mặc định            ║
# ╚══════════════════════════════════════════════════════════════════╝
print(SEP + "CÂU 3.4.2 – XẾP HẠNG 10 NGÀNH (TRỌNG SỐ MẶC ĐỊNH)" + SEP)

# Trọng số đề cho: a1..a6 cho cột tốt, a7 cho cột xấu
# Tổng = 0.15+0.15+0.20+0.15+0.10+0.20+0.15 = 1.10
# Đây là hệ số của công thức tuyến tính, KHÔNG bắt buộc tổng = 1
a1, a2, a3, a4, a5, a6, a7 = 0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15
w_goods_def = [a1, a2, a3, a4, a5, a6]
w_risk_def  = a7

df["Priority"] = calc_priority(df, w_goods_def, w_risk_def)
rank_df = (df[["sector_name_vi", "Priority"]]
           .sort_values("Priority", ascending=False)
           .reset_index(drop=True))
rank_df.index += 1

print(f"{'Hạng':<6} {'Ngành':<32} {'Priority':>10}")
print("─" * 52)
for hng, row in rank_df.iterrows():
    mark = " 🏆" if hng <= 3 else ""
    print(f"  {hng:<4} {row['sector_name_vi']:<32} {row['Priority']:>10.4f}{mark}")

top3_default = list(rank_df.head(3)["sector_name_vi"])
print(f"\n  Trọng số: a1={a1}, a2={a2}, a3={a3}, a4={a4}, a5={a5}, a6={a6}, a7={a7}")
print(f"  TOP-3: {top3_default}")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CÂU 3.4.3 – Phân tích độ nhạy theo a6 (AI Readiness)          ║
# ╚══════════════════════════════════════════════════════════════════╝
print(SEP + "CÂU 3.4.3 – PHÂN TÍCH ĐỘ NHẠY a6 ∈ [0.05, 0.40]" + SEP)

a6_range    = np.arange(0.05, 0.41, 0.05).round(2)
sectors_vi  = list(df["sector_name_vi"])
rank_records = {}   # {a6_val: Series hạng theo sector}

for a6_val in a6_range:
    # Giữ nguyên tỷ lệ 6 hệ số còn lại, chuẩn hóa lại để tổng 7 hệ số = 1.10
    # (tức là tổng 6 hệ số tốt + a7 = 1.10 - a6_val + a6_val = const)
    # Cách đơn giản hơn: giảm/tăng đều các a khác, giữ a7 cố định
    # → chuẩn hóa để tổng 7 trọng số = 1 (scale toàn bộ)
    base_others = np.array([a1, a2, a3, a4, a5, a7])   # 6 cái còn lại (trừ a6)
    # Tổng mới = 1 → scale others
    remaining = 1.0 - a6_val
    scale = remaining / base_others.sum()
    w_sc  = base_others * scale
    w_g   = [w_sc[0], w_sc[1], w_sc[2], w_sc[3], w_sc[4], a6_val]
    w_r   = w_sc[5]

    prio  = calc_priority(df, w_g, w_r)
    ranks = pd.Series(prio, index=sectors_vi).rank(ascending=False).astype(int)
    rank_records[a6_val] = ranks
    top3  = list(ranks.sort_values().head(3).index)
    print(f"  a6={a6_val:.2f}  Tổng w_good={sum(w_g):.3f}  w_risk={w_r:.3f}  →  Top-3: {top3}")

# ── Heatmap ──────────────────────────────────────────────────────
rank_matrix = pd.DataFrame(rank_records, index=sectors_vi)
# index=sector, columns=a6_val

fig1, ax1 = plt.subplots(figsize=(13, 6))
sns.heatmap(
    rank_matrix,
    annot=True, fmt="d",
    linewidths=0.5,
    cmap="YlOrRd_r",
    cbar_kws={"label": "Hạng (1 = tốt nhất)"},
    ax=ax1,
    annot_kws={"size": 10},
)
ax1.set_title(
    "Phân tích độ nhạy – Hạng ngành khi thay đổi a₆ (AI Readiness)\n"
    "(Màu xanh = hạng cao, màu đỏ = hạng thấp)",
    fontsize=12, fontweight="bold", pad=10,
)
ax1.set_xlabel("Trọng số a₆", fontsize=11)
ax1.set_ylabel("Ngành kinh tế", fontsize=11)
ax1.set_xticklabels([f"{v:.2f}" for v in a6_range], rotation=0)
ax1.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig("sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  → Biểu đồ heatmap đã lưu: sensitivity_heatmap.png")

# Nhận xét top-3
top3_sets = [
    frozenset(rank_records[v].nsmallest(3).index) for v in a6_range
]
unique_top3 = list(dict.fromkeys(top3_sets))
print(f"\n  Số tập top-3 khác nhau: {len(unique_top3)}")
if len(unique_top3) == 1:
    print("  → Top-3 KHÔNG thay đổi khi a6 thay đổi từ 0.05 đến 0.40.")
else:
    print("  → Top-3 CÓ thay đổi:")
    prev = None
    for v, t3 in zip(a6_range, top3_sets):
        if t3 != prev:
            print(f"     a6={v:.2f}: {sorted(t3)}")
            prev = t3


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CÂU 3.4.4 – So sánh hai bộ trọng số chiến lược                ║
# ╚══════════════════════════════════════════════════════════════════╝
print(SEP + "CÂU 3.4.4 – SO SÁNH HAI BỘ TRỌNG SỐ CHIẾN LƯỢC" + SEP)

# ── Bộ A: Định hướng TĂNG TRƯỞNG ────────────────────────────────
# Ưu tiên: Tăng trưởng(C1), GDP(C2), Xuất khẩu(C4), AI(C6)
WA_goods = [0.25, 0.20, 0.10, 0.25, 0.05, 0.10]   # C1..C6
WA_risk  = 0.05

# ── Bộ B: Định hướng BAO TRÙM ────────────────────────────────────
# Ưu tiên: Việc làm(C5), Lan tỏa(C3), Giảm rủi ro(C7)
WB_goods = [0.10, 0.10, 0.20, 0.10, 0.25, 0.10]   # C1..C6
WB_risk  = 0.15

prioA = calc_priority(df, WA_goods, WA_risk)
prioB = calc_priority(df, WB_goods, WB_risk)

rankA = pd.Series(prioA, index=df["sector_name_vi"]).sort_values(ascending=False)
rankB = pd.Series(prioB, index=df["sector_name_vi"]).sort_values(ascending=False)
top3A = list(rankA.head(3).index)
top3B = list(rankB.head(3).index)

# In bảng trọng số
print(f"  {'Chỉ tiêu':<22} {'Tăng trưởng':>14} {'Bao trùm':>10}")
print("  " + "─" * 48)
lbl7  = ["C1 Tăng trưởng", "C2 GDP Share", "C3 Lan tỏa",
         "C4 Xuất khẩu",   "C5 Việc làm",  "C6 AI Ready", "C7 Risk (trừ)"]
wA_all = WA_goods + [WA_risk]
wB_all = WB_goods + [WB_risk]
for lbl, wa, wb in zip(lbl7, wA_all, wB_all):
    print(f"  {lbl:<22} {wa:>14.2f} {wb:>10.2f}")
print(f"  {'Tổng':<22} {sum(wA_all):>14.2f} {sum(wB_all):>10.2f}")

# In bảng xếp hạng song song
print(f"\n  {'Hạng A':<8} {'Ngành (A)':<32} {'Score A':>9}   "
      f"{'Hạng B':<8} {'Ngành (B)':<32} {'Score B':>9}")
print("  " + "─" * 100)
for i in range(10):
    sA, vA = rankA.index[i], rankA.iloc[i]
    sB, vB = rankB.index[i], rankB.iloc[i]
    mA = "🏆" if i < 3 else "  "
    mB = "🏆" if i < 3 else "  "
    print(f"  {mA} {i+1:<5} {sA:<32} {vA:>9.4f}   "
          f"{mB} {i+1:<5} {sB:<32} {vB:>9.4f}")

print(f"\n  🏆 Top-3 TĂNG TRƯỞNG : {top3A}")
print(f"  🌱 Top-3 BAO TRÙM    : {top3B}")

if set(top3A) == set(top3B):
    print("  → Hai bộ cho TOP-3 GIỐNG NHAU.")
else:
    only_A = [s for s in top3A if s not in top3B]
    only_B = [s for s in top3B if s not in top3A]
    print("  → TOP-3 KHÁC NHAU!")
    if only_A: print(f"     Chỉ trong TT : {only_A}")
    if only_B: print(f"     Chỉ trong BT : {only_B}")

# ── Biểu đồ so sánh bar chart ─────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
pairs = [
    (axes[0], rankA, top3A, "#27ae60", "#abebc6", "📈 Định hướng TĂNG TRƯỞNG"),
    (axes[1], rankB, top3B, "#e67e22", "#fad7a0", "🌱 Định hướng BAO TRÙM"),
]
for ax, rank_s, top3, c_top, c_rest, title in pairs:
    colors = [c_top if s in top3 else c_rest for s in rank_s.index[::-1]]
    bars = ax.barh(rank_s.index[::-1], rank_s.values[::-1],
                   color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Priority Score", fontsize=10)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    for bar, val in zip(bars, rank_s.values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)

plt.suptitle("So sánh Xếp hạng Ngành theo Hai Định hướng Chiến lược",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("comparison_two_weights.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  → Biểu đồ so sánh đã lưu: comparison_two_weights.png")

# ── Radar chart trọng số ──────────────────────────────────────
CAT_LABELS = ["C1\nTăng trưởng", "C2\nGDP", "C3\nLan tỏa",
              "C4\nXuất khẩu",   "C5\nViệc làm", "C6\nAI", "C7\nRisk"]
N = len(CAT_LABELS)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig3, ax3 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for vals, color, label, marker in [
    (wA_all, "#27ae60", "Tăng trưởng", "o"),
    (wB_all, "#e67e22", "Bao trùm",    "s"),
]:
    v = vals + [vals[0]]
    ax3.plot(angles, v, f"{marker}-", linewidth=2.2, color=color, label=label)
    ax3.fill(angles, v, alpha=0.18, color=color)

ax3.set_thetagrids(np.degrees(angles[:-1]), CAT_LABELS, fontsize=10)
ax3.set_ylim(0, 0.30)
ax3.set_title("Radar Trọng số – Hai Bộ Chiến lược",
              fontsize=13, fontweight="bold", pad=22)
ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=10)
plt.tight_layout()
plt.savefig("radar_weights.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → Biểu đồ radar đã lưu: radar_weights.png")

print(SEP + "✅ HOÀN THÀNH TẤT CẢ CÂU 3.4.1 – 3.4.4" + SEP)