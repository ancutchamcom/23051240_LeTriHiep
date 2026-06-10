# ============================================================
#  AIDEOM-VN — Hệ thống Mô hình Ra quyết định Phát triển
#  Kinh tế Việt Nam trong Kỷ nguyên AI
#  Giao diện kiểm tra bài tập cho Giảng viên chấm điểm
#  Sinh viên: Trần Hoàng Bách
#  Viện Quản trị Kinh doanh — Đại học Kinh tế, ĐHQGHN
#  Chạy:  streamlit run app.py
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AIDEOM-VN", page_icon="🇻🇳",
                   layout="wide", initial_sidebar_state="expanded")

ACCENT = "#ff4d6d"

# ============================================================
#  HÀM NẠP DỮ LIỆU CHUNG (đọc 3 CSV gốc cùng thư mục app.py)
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))


def _find(fname):
    """Tìm file CSV ở thư mục app.py hoặc thư mục data/."""
    for cand in (os.path.join(HERE, fname), os.path.join(HERE, "data", fname), fname):
        if os.path.exists(cand):
            return cand
    return fname


@st.cache_data
def load_macro():
    df = pd.read_csv(_find("vietnam_macro_2020_2025.csv"))
    return df.sort_values("year").reset_index(drop=True)


@st.cache_data
def load_sectors():
    return pd.read_csv(_find("vietnam_sectors_2024.csv"))


@st.cache_data
def load_regions():
    return pd.read_csv(_find("vietnam_regions_2024.csv"))


# Tên tiếng Việt cho 10 ngành
VI_SECTOR = {
    "Agriculture-Forestry-Fishery": "Nông-Lâm-Thủy sản",
    "Manufacturing": "CN chế biến chế tạo",
    "Construction": "Xây dựng",
    "Mining": "Khai khoáng",
    "Wholesale-Retail": "Bán buôn-Bán lẻ",
    "Finance-Banking-Insurance": "Tài chính-Ngân hàng",
    "Logistics-Transport-Warehousing": "Logistics-Vận tải",
    "Information-Communication-IT": "CNTT-Truyền thông",
    "Education-Training": "Giáo dục-Đào tạo",
    "Healthcare": "Y tế",
}

# ============================================================
#  CSS giao diện tối giản, accent crimson/pink
# ============================================================
st.markdown(f"""
<style>
.stApp {{ background:#0e1117; }}
h1,h2,h3 {{ color:#fafafa; }}
.big-metric {{ font-size:2.0rem; font-weight:800; color:{ACCENT}; }}
.sub {{ color:#9aa0a6; font-size:0.9rem; }}
div[data-testid="stMetricValue"] {{ color:{ACCENT}; }}
</style>
""", unsafe_allow_html=True)


def header(title, goal):
    """In tiêu đề bài + mục tiêu kinh tế ngắn gọn."""
    st.markdown(f"## {title}")
    st.markdown(f"<span class='sub'>🎯 <b>Mục tiêu:</b> {goal}</span>",
                unsafe_allow_html=True)
    st.divider()


# ============================================================
#  SIDEBAR ĐIỀU HƯỚNG
# ============================================================
st.sidebar.markdown(f"<h2 style='color:{ACCENT}'>🇻🇳 AIDEOM-VN</h2>",
                    unsafe_allow_html=True)
st.sidebar.markdown("<span class='sub'>Mô hình ra quyết định phát triển "
                    "kinh tế VN trong kỉ nguyên AI</span>",
                    unsafe_allow_html=True)
st.sidebar.divider()

PAGES = [
    "🏠 Trang chủ Tổng quan Hệ thống",
    "🔹 Bài 1",
    "🔹 Bài 2",
    "🔹 Bài 3",
    "🔹 Bài 4",
    "🔹 Bài 5",
    "🔹 Bài 6",
    "🔹 Bài 7",
    "🔹 Bài 8",
    "🔹 Bài 9",
    "🔹 Bài 10",
    "🔹 Bài 11",
    "🚀 Bài 12",
]
choice = st.sidebar.radio("Điều hướng", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown(
    "<span class='sub'><b>Sinh viên:</b> Trần Hoàng Bách<br>"
    "Viện Quản trị Kinh doanh<br>Đại học Kinh tế, ĐHQGHN</span>",
    unsafe_allow_html=True)


# ============================================================
#  TRANG CHỦ
# ============================================================
def page_home():
    st.markdown(f"<h1 style='color:{ACCENT}'>AIDEOM-VN</h1>",
                unsafe_allow_html=True)
    st.markdown("#### AI-Driven Economic Decision Optimization Model for Vietnam")
    st.write("Hệ thống giải **12 bài toán mô hình ra quyết định** phát triển "
             "kinh tế Việt Nam trong kỷ nguyên AI — sử dụng **dữ liệu thực "
             "2020–2025** (GSO, MoST, MIC, MPI, World Bank, GII). Chính sách "
             "tham chiếu: Nghị quyết 57-NQ/TW và các QĐ 749/127/411/QĐ-TTg.")

    macro = load_macro()
    last = macro.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP 2025", f"{last['GDP_billion_USD']:.1f} tỷ USD",
              f"+{last['GDP_growth_pct']:.2f}%")
    c2.metric("Kinh tế số / GDP", f"≈{last['digital_economy_share_GDP_pct']:.1f}%")
    c3.metric("FDI giải ngân 2025", f"{last['FDI_disbursed_billion_USD']:.1f} tỷ USD")
    c4.metric("GDP/người 2025", f"{last['GDP_per_capita_USD']:,.0f} USD")

    st.divider()
    st.markdown("### 📁 Kiểm tra nạp dữ liệu (3 tệp CSV gốc)")
    files = [
        ("vietnam_macro_2020_2025.csv", macro),
        ("vietnam_sectors_2024.csv", load_sectors()),
        ("vietnam_regions_2024.csv", load_regions()),
    ]
    for name, df in files:
        st.markdown(f"**{name}** — Shape: `{df.shape[0]} dòng × {df.shape[1]} cột`")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🗂️ 12 bài toán theo 4 cấp độ")
    tiers = pd.DataFrame({
        "Cấp độ": ["DỄ", "DỄ", "DỄ", "TB", "TB", "TB",
                   "KHÁ KHÓ", "KHÁ KHÓ", "KHÁ KHÓ", "KHÓ", "KHÓ", "KHÓ"],
        "Bài": [f"Bài {i}" for i in range(1, 13)],
        "Nội dung": [
            "Hàm sản xuất Cobb-Douglas mở rộng + AI — TFP & growth accounting",
            "LP phân bổ ngân sách 4 hạng mục — scipy, shadow price",
            "Chỉ số ưu tiên 10 ngành — min-max norm, weighted scoring",
            "LP phân bổ ngân sách ngành-vùng — PuLP + CVXPY",
            "MIP lựa chọn 15 dự án chuyển đổi số — PuLP/CBC",
            "TOPSIS xếp hạng 6 vùng — Entropy weight + AHP",
            "NSGA-II Pareto 4 mục tiêu — pymoo + TOPSIS thỏa hiệp",
            "Tối ưu động liên thời gian 2026-2035 — SLSQP",
            "Tác động AI tới lao động — NetJob LP",
            "Quy hoạch ngẫu nhiên 2 giai đoạn — VSS/EVPI",
            "Q-learning chính sách kinh tế thích nghi — MDP",
            "Đồ án tích hợp 6 module + dashboard 5 kịch bản",
        ],
    })
    st.dataframe(tiers, use_container_width=True, hide_index=True)


# ============================================================
#  DỮ LIỆU CHUỖI THỜI GIAN BÀI 1 (lấy từ đề bài — vốn, LĐ,
#  số hóa, AI, nhân lực số 2020-2025; Y lấy từ CSV macro)
# ============================================================
K_SER = np.array([16500, 17800, 19600, 21300, 23500, 25900], float)
L_SER = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], float)
D_SER = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5], float)
AI_SER = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], float)
H_SER = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], float)
ALPHA, BETA, GAMMA, DELTA, THETA = 0.33, 0.42, 0.10, 0.08, 0.07


# ============================================================
#  BÀI 1 — Cobb-Douglas mở rộng: TFP, MAPE, phân rã, dự báo 2030
# ============================================================
def page_bai1():
    header("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng + AI",
           "Ước lượng TFP, phân rã tăng trưởng GDP và dự báo GDP 2030.")
    macro = load_macro()
    years = macro["year"].values
    Y = macro["GDP_trillion_VND"].values

    A = Y / (K_SER**ALPHA * L_SER**BETA * D_SER**GAMMA *
             AI_SER**DELTA * H_SER**THETA)
    A_mean = A.mean()
    Y_hat = A_mean * (K_SER**ALPHA * L_SER**BETA * D_SER**GAMMA *
                      AI_SER**DELTA * H_SER**THETA)
    MAPE = (np.abs((Y - Y_hat) / Y) * 100).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("TFP trung bình Ā", f"{A_mean:.3f}")
    c2.metric("MAPE dự báo", f"{MAPE:.2f}%")
    c3.metric("Xu hướng TFP", f"+{np.polyfit(years, A, 1)[0]:.3f}/năm")

    # 1.4.1 — TFP theo năm
    st.markdown("#### 1.4.1 — Năng suất nhân tố tổng hợp (TFP) Aₜ")
    df_tfp = pd.DataFrame({"Năm": years, "TFP (Aₜ)": A.round(4)})
    cc = st.columns([1, 2])
    cc[0].dataframe(df_tfp, hide_index=True, use_container_width=True)
    fig = px.line(df_tfp, x="Năm", y="TFP (Aₜ)", markers=True,
                  template="plotly_dark", title="TFP Aₜ — Việt Nam 2020-2025")
    fig.update_traces(line_color=ACCENT)
    cc[1].plotly_chart(fig, use_container_width=True)

    # 1.4.2 — So sánh Y vs Ŷ
    st.markdown("#### 1.4.2 — GDP thực tế vs dự báo (Ā cố định)")
    df_cmp = pd.DataFrame({"Năm": years, "Y thực tế": Y.round(1),
                           "Ŷ dự báo": Y_hat.round(1)})
    fig2 = px.bar(df_cmp, x="Năm", y=["Y thực tế", "Ŷ dự báo"],
                  barmode="group", template="plotly_dark",
                  color_discrete_sequence=[ACCENT, "#4a9eff"],
                  title=f"So sánh GDP (MAPE = {MAPE:.2f}%)")
    st.plotly_chart(fig2, use_container_width=True)

    # 1.4.3 — Phân rã tăng trưởng
    st.markdown("#### 1.4.3 — Phân rã đóng góp tăng trưởng 2020-2025")
    g = lambda x: np.diff(np.log(x)) * 100
    contrib = {
        "K": ALPHA * g(K_SER), "L": BETA * g(L_SER), "D": GAMMA * g(D_SER),
        "AI": DELTA * g(AI_SER), "H": THETA * g(H_SER), "TFP": g(A),
    }
    means = {k: v.mean() for k, v in contrib.items()}
    total = sum(means.values())
    shares = {k: v / total * 100 for k, v in means.items()}
    df_sh = pd.DataFrame({
        "Yếu tố": list(means.keys()),
        "Đóng góp (pp/năm)": [round(v, 3) for v in means.values()],
        "Tỷ trọng (%)": [round(v, 1) for v in shares.values()],
    }).sort_values("Tỷ trọng (%)", ascending=False)
    cc = st.columns([1, 2])
    cc[0].dataframe(df_sh, hide_index=True, use_container_width=True)
    fig3 = px.bar(df_sh, x="Tỷ trọng (%)", y="Yếu tố", orientation="h",
                  template="plotly_dark", color="Tỷ trọng (%)",
                  color_continuous_scale="reds",
                  title="Tỷ trọng đóng góp bình quân/năm")
    cc[1].plotly_chart(fig3, use_container_width=True)

    # 1.4.4 — Dự báo 2030
    st.markdown("#### 1.4.4 — Dự báo GDP năm 2030")
    n = 5
    K30 = K_SER[-1] * 1.06**n; L30 = L_SER[-1] * 1.06**n
    A30 = A_mean * 1.012**n
    Y30 = A30 * (K30**ALPHA * L30**BETA * 30.0**GAMMA *
                 100.0**DELTA * 35.0**THETA)
    cagr = ((Y30 / Y[-1])**(1/n) - 1) * 100
    sim_years = np.arange(2025, 2031)
    Ksim = K_SER[-1] * 1.06**np.arange(6)
    Lsim = L_SER[-1] * 1.06**np.arange(6)
    Dsim = np.linspace(D_SER[-1], 30, 6); AIsim = np.linspace(AI_SER[-1], 100, 6)
    Hsim = np.linspace(H_SER[-1], 35, 6); Asim = A_mean * 1.012**np.arange(6)
    Ysim = Asim * (Ksim**ALPHA * Lsim**BETA * Dsim**GAMMA *
                   AIsim**DELTA * Hsim**THETA)
    c1, c2 = st.columns(2)
    c1.metric("GDP dự báo 2030", f"{Y30:,.0f} ngh.tỷ VND")
    c2.metric("CAGR 2025-2030", f"{cagr:.2f}%/năm")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=years, y=Y, mode="lines+markers",
                              name="Thực tế 2020-2025", line=dict(color=ACCENT)))
    fig4.add_trace(go.Scatter(x=sim_years, y=Ysim, mode="lines+markers",
                              name="Dự báo 2025-2030",
                              line=dict(color="#4a9eff", dash="dash")))
    fig4.update_layout(template="plotly_dark", title="Dự báo GDP đến 2030",
                       yaxis_title="GDP (ngh.tỷ VND)")
    st.plotly_chart(fig4, use_container_width=True)


# ============================================================
#  BÀI 2 — LP phân bổ ngân sách 4 hạng mục (scipy)
# ============================================================
from scipy.optimize import linprog


def _solve_lp_bai2(budget=100, x3_min=20):
    c = [-0.85, -1.20, -0.95, -1.35]
    A_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0],
            [0, 0, -1, 0], [0, 0, 0, -1], [0.35, -0.65, 0.35, -0.65]]
    b_ub = [budget, -25, -15, -x3_min, -10, 0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    if res.success:
        return True, -res.fun, res.x, -res.ineqlin.marginals
    return False, None, None, None


def page_bai2():
    header("Bài 2 — LP phân bổ ngân sách 4 hạng mục đầu tư số",
           "Tối đa hóa tăng GDP kỳ vọng từ 100 ngh.tỷ VND, phân tích shadow price.")
    var = ["x1 Hạ tầng số", "x2 R&D AI", "x3 Nhân lực số", "x4 Ứng dụng AI"]
    ok, Z, x, duals = _solve_lp_bai2()

    # 2.4.1
    st.markdown("#### 2.4.1 — Nghiệm tối ưu (scipy.linprog)")
    c1, c2 = st.columns([1, 2])
    c1.metric("Z* (tăng GDP)", f"{Z:.4f}")
    df_x = pd.DataFrame({"Hạng mục": var, "Phân bổ (ngh.tỷ)": x.round(2),
                         "Hệ số ROI": [0.85, 1.20, 0.95, 1.35]})
    c1.dataframe(df_x, hide_index=True, use_container_width=True)
    fig = px.pie(df_x, names="Hạng mục", values="Phân bổ (ngh.tỷ)",
                 template="plotly_dark", hole=0.4,
                 color_discrete_sequence=px.colors.sequential.Reds_r,
                 title="Cơ cấu phân bổ tối ưu")
    c2.plotly_chart(fig, use_container_width=True)

    # 2.4.2 — shadow price
    st.markdown("#### 2.4.2 — Giá đối ngẫu (shadow price)")
    cons = ["R1 Ngân sách tổng", "R2 x1≥25", "R3 x2≥15", "R4 x3≥20",
            "R5 x4≥10", "R6 Cân bằng CN chiến lược"]
    df_sp = pd.DataFrame({"Ràng buộc": cons, "Shadow price": duals.round(4)})
    st.dataframe(df_sp, hide_index=True, use_container_width=True)
    st.info(f"💡 Shadow price ngân sách tổng = **{duals[0]:.4f}** — tăng 1 ngh.tỷ "
            f"ngân sách làm Z* tăng thêm {duals[0]:.4f} đơn vị (chi phí cơ hội biên của vốn công).")

    # 2.4.3 — sensitivity Z*(B)
    st.markdown("#### 2.4.3 — Đường cong độ nhạy Z*(B)")
    budgets = list(range(70, 181, 5))
    rows = [(B, _solve_lp_bai2(B)[1]) for B in budgets]
    df_s = pd.DataFrame(rows, columns=["Ngân sách B", "Z*"])
    fig3 = px.line(df_s, x="Ngân sách B", y="Z*", markers=True,
                   template="plotly_dark", title="Z* theo ngân sách tổng B")
    fig3.update_traces(line_color=ACCENT)
    st.plotly_chart(fig3, use_container_width=True)

    # 2.4.4 — x3>=30
    st.markdown("#### 2.4.4 — Ưu tiên nhân lực số (x3 ≥ 30)")
    ok30, Z30, x30, _ = _solve_lp_bai2(x3_min=30)
    c1, c2, c3 = st.columns(3)
    c1.metric("Khả thi?", "✅ Có" if ok30 else "❌ Không")
    c2.metric("Z* (x3≥30)", f"{Z30:.4f}")
    c3.metric("Chi phí cơ hội ΔZ*", f"{Z30 - Z:.4f}", f"{(Z30-Z)/Z*100:.2f}%")


# ============================================================
#  BÀI 3 — Chỉ số ưu tiên 10 ngành (min-max + weighted scoring)
# ============================================================
def _priority_data():
    df = load_sectors().copy()
    df["sector_name_vi"] = df["sector_name_en"].map(VI_SECTOR)
    return df


COLS_GOOD = ["growth_rate_2024_pct", "gdp_share_2024_pct", "spillover_coef_0_1",
             "export_billion_USD", "labor_million", "ai_readiness_0_100"]
COL_BAD = "automation_risk_pct"


def _norm_good(s): return (s - s.min()) / (s.max() - s.min())


def _calc_priority(df, w_goods, w_risk):
    Xg = df[COLS_GOOD].apply(_norm_good).values
    Xb = _norm_good(df[COL_BAD]).values
    return Xg @ np.array(w_goods) - w_risk * Xb


def page_bai3():
    header("Bài 3 — Chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành",
           "Xếp hạng ngành nên ưu tiên chuyển đổi số/AI bằng weighted scoring.")
    df = _priority_data()
    w_goods = [0.15, 0.15, 0.20, 0.15, 0.10, 0.20]
    w_risk = 0.15

    # 3.4.2 — ranking
    df["Priority"] = _calc_priority(df, w_goods, w_risk)
    rank = df[["sector_name_vi", "Priority"]].sort_values(
        "Priority", ascending=False).reset_index(drop=True)
    rank.index += 1
    st.markdown("#### 3.4.2 — Xếp hạng theo trọng số mặc định")
    c1, c2 = st.columns([1, 2])
    c1.dataframe(rank.round(4).rename(columns={"sector_name_vi": "Ngành"}),
                 use_container_width=True)
    fig = px.bar(rank.reset_index(), x="Priority", y="sector_name_vi",
                 orientation="h", template="plotly_dark", color="Priority",
                 color_continuous_scale="reds", title="Chỉ số ưu tiên 10 ngành")
    fig.update_layout(yaxis_title="", yaxis=dict(autorange="reversed"))
    c2.plotly_chart(fig, use_container_width=True)
    st.success(f"🏆 TOP-3 ưu tiên: **{', '.join(rank.head(3)['sector_name_vi'])}**")

    # 3.4.1 — normalized matrix
    with st.expander("3.4.1 — Ma trận chuẩn hóa min-max"):
        nm = df[COLS_GOOD].apply(_norm_good)
        nm["risk_inv"] = (df[COL_BAD].max() - df[COL_BAD]) / \
            (df[COL_BAD].max() - df[COL_BAD].min())
        nm.index = df["sector_name_vi"]
        st.dataframe(nm.round(3), use_container_width=True)

    # 3.4.4 — so sánh 2 bộ trọng số
    st.markdown("#### 3.4.4 — So sánh hai định hướng chính sách")
    pa = _calc_priority(df, [0.25, 0.20, 0.10, 0.25, 0.05, 0.10], 0.05)
    pb = _calc_priority(df, [0.10, 0.10, 0.20, 0.10, 0.25, 0.10], 0.15)
    cmp = pd.DataFrame({"Ngành": df["sector_name_vi"],
                        "Tăng trưởng": pa.round(4), "Bao trùm": pb.round(4)})
    fig2 = px.bar(cmp, x="Ngành", y=["Tăng trưởng", "Bao trùm"],
                  barmode="group", template="plotly_dark",
                  color_discrete_sequence=[ACCENT, "#4a9eff"])
    fig2.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig2, use_container_width=True)
    t3a = list(cmp.sort_values("Tăng trưởng", ascending=False).head(3)["Ngành"])
    t3b = list(cmp.sort_values("Bao trùm", ascending=False).head(3)["Ngành"])
    c1, c2 = st.columns(2)
    c1.info("**Định hướng tăng trưởng** TOP-3: " + ", ".join(t3a))
    c2.info("**Định hướng bao trùm** TOP-3: " + ", ".join(t3b))


# ============================================================
#  BÀI 4 — LP phân bổ ngân sách ngành-vùng (PuLP)
# ============================================================
REGIONS4 = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_FULL = {"NMM": "Trung du-Miền núi PB", "RRD": "ĐB sông Hồng",
               "NCC": "BTB-DH Trung Bộ", "CH": "Tây Nguyên",
               "SE": "Đông Nam Bộ", "MD": "ĐB sông Cửu Long"}
ITEMS4 = ["I", "D", "AI", "H"]
BETA4 = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
    ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
    ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
    ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25,
}
D0_4 = {"NMM": 38, "RRD": 78, "NCC": 55, "CH": 32, "SE": 82, "MD": 48}
GAMMA4, LAM4 = 0.002, 0.65
BUDGET4, MINR, MAXR, MINH = 50000, 5000, 12000, 12000


@st.cache_data
def _solve_bai4(with_c5=True):
    import pulp
    m = pulp.LpProblem("VN_Budget", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (REGIONS4, ITEMS4), lowBound=0)
    m += pulp.lpSum(BETA4[(r, j)] * x[r][j] for r in REGIONS4 for j in ITEMS4)
    m += pulp.lpSum(x[r][j] for r in REGIONS4 for j in ITEMS4) <= BUDGET4
    for r in REGIONS4:
        m += pulp.lpSum(x[r][j] for j in ITEMS4) >= MINR
        m += pulp.lpSum(x[r][j] for j in ITEMS4) <= MAXR
    m += pulp.lpSum(x[r]["H"] for r in REGIONS4) >= MINH
    if with_c5:
        Mv = pulp.LpVariable("Dmax", lowBound=0)
        for r in REGIONS4:
            m += D0_4[r] + GAMMA4 * x[r]["D"] <= Mv
            m += D0_4[r] + GAMMA4 * x[r]["D"] >= LAM4 * Mv
        m += Mv <= max(D0_4.values()) + GAMMA4 * MAXR
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    mat = np.array([[pulp.value(x[r][j]) or 0 for j in ITEMS4] for r in REGIONS4])
    return mat, pulp.value(m.objective)


def page_bai4():
    header("Bài 4 — LP phân bổ ngân sách số ngành-vùng",
           "Phân bổ 50.000 tỷ cho 6 vùng × 4 hạng mục, đảm bảo công bằng vùng miền.")
    mat, Z = _solve_bai4(True)
    mat0, Z0 = _solve_bai4(False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Z* (có công bằng C5)", f"{Z:,.0f} tỷ")
    c2.metric("Z* (không C5)", f"{Z0:,.0f} tỷ")
    c3.metric("Chi phí công bằng", f"{Z0 - Z:,.0f} tỷ", f"{(Z0-Z)/Z0*100:.2f}%")

    st.markdown("#### 4.4.1 + 4.4.3 — Phân bổ tối ưu & heatmap (PuLP/CBC)")
    dfm = pd.DataFrame(mat.round(0), index=[REGION_FULL[r] for r in REGIONS4],
                       columns=["Hạ tầng (I)", "Số hóa (D)", "AI", "Nhân lực (H)"])
    c1, c2 = st.columns([1, 1])
    c1.dataframe(dfm, use_container_width=True)
    fig = px.imshow(mat, x=["I", "D", "AI", "H"],
                    y=[REGION_FULL[r] for r in REGIONS4],
                    color_continuous_scale="reds", text_auto=".0f",
                    template="plotly_dark", title="Heatmap phân bổ (tỷ VND)")
    c2.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 4.4.4 — Chi phí kinh tế của công bằng vùng miền")
    tot_c5 = mat.sum(1); tot_nc5 = mat0.sum(1)
    dfb = pd.DataFrame({"Vùng": [REGION_FULL[r] for r in REGIONS4],
                        "Có C5": tot_c5, "Không C5": tot_nc5})
    fig2 = px.bar(dfb, x="Vùng", y=["Có C5", "Không C5"], barmode="group",
                  template="plotly_dark", color_discrete_sequence=[ACCENT, "#4a9eff"],
                  title="Tổng ngân sách theo vùng")
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
#  BÀI 5 — MIP lựa chọn 15 dự án (PuLP/CBC)
# ============================================================
C5_COST = {1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200, 6: 5800, 7: 6500,
           8: 15000, 9: 2500, 10: 7200, 11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500}
C5_C1 = {1: 8500, 2: 7500, 3: 12000, 4: 3500, 5: 2500, 6: 4000, 7: 4500, 8: 9000,
         9: 1800, 10: 5000, 11: 3500, 12: 5500, 13: 13000, 14: 2800, 15: 1200}
C5_B = {1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800, 6: 11400, 7: 12200, 8: 28500,
        9: 5800, 10: 13800, 11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800}
C5_NAME = {1: "TT dữ liệu Hòa Lạc", 2: "TT dữ liệu phía Nam", 3: "5G toàn quốc",
           4: "VNeID 2.0", 5: "Cổng DVC v3", 6: "Y tế số", 7: "Giáo dục số K-12",
           8: "TT AI quốc gia", 9: "Sandbox fintech", 10: "Logistics thông minh",
           11: "Nông nghiệp số ĐBSCL", 12: "Đào tạo 50k kỹ sư AI", 13: "KCN bán dẫn",
           14: "An ninh mạng SOC", 15: "Open Data"}


@st.cache_data
def _solve_bai5(b1=80000, force_p1p2=False, expected=False):
    import pulp
    P = list(range(1, 16))
    p_risk = {i: 0.85 if i in {3, 8, 14} else 0.75 if i in {1, 2, 12}
              else 0.65 if i in {7, 10, 13} else 0.80 for i in P}
    m = pulp.LpProblem("Sel", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("y", P, cat="Binary")
    if expected:
        m += pulp.lpSum(p_risk[i] * C5_B[i] * y[i] for i in P)
    else:
        m += pulp.lpSum(C5_B[i] * y[i] for i in P)
    m += pulp.lpSum(C5_COST[i] * y[i] for i in P) <= b1
    m += pulp.lpSum(C5_C1[i] * y[i] for i in P) <= 40000
    if force_p1p2:
        m += y[1] == 1; m += y[2] == 1
    else:
        m += y[1] + y[2] <= 1
    m += y[8] <= y[12]; m += y[13] <= y[12]
    m += y[4] + y[5] >= 1; m += y[14] >= 1
    m += pulp.lpSum(y[i] for i in P) >= 7
    m += pulp.lpSum(y[i] for i in P) <= 11
    status = m.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        return None, None
    sel = [i for i in P if (y[i].value() or 0) > 0.5]
    return sel, pulp.value(m.objective)


def page_bai5():
    header("Bài 5 — MIP lựa chọn dự án chuyển đổi số",
           "Chọn tập dự án tối ưu trong 15 ứng cử với ràng buộc ngân sách & cấu trúc.")
    sel, Z = _solve_bai5()
    sel2, Z2 = _solve_bai5(b1=100000)
    sel3, Z3 = _solve_bai5(force_p1p2=True)
    sel4, Z4 = _solve_bai5(expected=True)

    st.markdown("#### 5.4.1 — Bài toán gốc (GĐ1: 80k, GĐ2: 40k tỷ)")
    df = pd.DataFrame({
        "Mã": [f"P{i}" for i in sel],
        "Dự án": [C5_NAME[i] for i in sel],
        "Chi phí": [C5_COST[i] for i in sel],
        "Lợi ích B": [C5_B[i] for i in sel],
    })
    c1, c2 = st.columns([1.3, 1])
    c1.dataframe(df, hide_index=True, use_container_width=True)
    c2.metric("Tổng lợi ích Z*", f"{Z:,.0f} tỷ")
    c2.metric("Số dự án", len(sel))
    c2.metric("Tổng chi phí", f"{sum(C5_COST[i] for i in sel):,.0f} tỷ")
    fig = px.bar(df, x="Mã", y="Lợi ích B", template="plotly_dark",
                 color="Lợi ích B", color_continuous_scale="reds",
                 title="Lợi ích NPV các dự án được chọn")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### So sánh 4 kịch bản")
    rows = [("5.4.1 Gốc 80k", Z, sel), ("5.4.2 Nới 100k", Z2, sel2),
            ("5.4.3 Bắt buộc P1+P2", Z3, sel3 or []),
            ("5.4.4 Rủi ro E[Z]", Z4, sel4)]
    df_s = pd.DataFrame({
        "Kịch bản": [r[0] for r in rows],
        "Z* / E[Z]*": [f"{r[1]:,.0f}" if r[1] else "INFEASIBLE" for r in rows],
        "Số dự án": [len(r[2]) if r[1] else "—" for r in rows],
    })
    st.dataframe(df_s, hide_index=True, use_container_width=True)


# ============================================================
#  BÀI 6 — TOPSIS xếp hạng 6 vùng (Entropy + AHP)
# ============================================================
TOPSIS_CRIT = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD",
               "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct",
               "rd_intensity_pct", "internet_penetration_pct", "gini_coef"]
TOPSIS_BEN = [True, True, True, True, True, True, True, False]
CRIT_ABBR = ["GRDP/cap", "FDI", "Digital", "AI-Ready", "Trained Labor",
             "R&D", "Internet", "Gini"]


def _topsis(X, w, ben):
    R = X / np.sqrt((X**2).sum(0)); V = R * w
    ben = np.array(ben)
    Ap = np.where(ben, V.max(0), V.min(0))
    An = np.where(ben, V.min(0), V.max(0))
    Sp = np.sqrt(((V - Ap)**2).sum(1)); Sn = np.sqrt(((V - An)**2).sum(1))
    return Sn / (Sp + Sn)


def _entropy_w(X):
    P = X / X.sum(0); k = 1 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), 0)
    d = 1 - E
    return d / d.sum()


def page_bai6():
    header("Bài 6 — TOPSIS xếp hạng 6 vùng ưu tiên đầu tư AI",
           "Xếp hạng vùng triển khai trung tâm AI bằng MCDM TOPSIS + Entropy.")
    reg = load_regions()
    X = reg[TOPSIS_CRIT].values.astype(float)
    w_exp = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    C_exp = _topsis(X, w_exp, TOPSIS_BEN)
    w_ent = _entropy_w(X)
    C_ent = _topsis(X, w_ent, TOPSIS_BEN)

    names = reg["region_name_en"].tolist()
    df = pd.DataFrame({"Vùng": names,
                       "C* Expert": C_exp.round(4),
                       "C* Entropy": C_ent.round(4)})
    df["Hạng Expert"] = df["C* Expert"].rank(ascending=False).astype(int)
    df["Hạng Entropy"] = df["C* Entropy"].rank(ascending=False).astype(int)
    dfs = df.sort_values("Hạng Expert")

    st.markdown("#### 6.4.1 + 6.4.2 — TOPSIS trọng số Expert vs Entropy")
    c1, c2 = st.columns([1.2, 1])
    c1.dataframe(dfs, hide_index=True, use_container_width=True)
    fig = px.bar(dfs, x="C* Expert", y="Vùng", orientation="h",
                 template="plotly_dark", color="C* Expert",
                 color_continuous_scale="reds", title="Điểm TOPSIS (Expert)")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    c2.plotly_chart(fig, use_container_width=True)
    st.success(f"🏆 TOP-3 vùng ưu tiên AI: **{', '.join(dfs.head(3)['Vùng'])}**")

    st.markdown("#### Trọng số Entropy (khách quan)")
    dfw = pd.DataFrame({"Tiêu chí": CRIT_ABBR, "Expert": w_exp.round(3),
                        "Entropy": w_ent.round(3)})
    fig2 = px.bar(dfw, x="Tiêu chí", y=["Expert", "Entropy"], barmode="group",
                  template="plotly_dark", color_discrete_sequence=[ACCENT, "#4a9eff"])
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
#  BÀI 7 — NSGA-II Pareto 4 mục tiêu (pymoo)
# ============================================================
BETA7 = np.column_stack([
    [0.000085, 0.000045, 0.000065, 0.000092, 0.000040, 0.000078],
    [0.000072, 0.000095, 0.000082, 0.000058, 0.000105, 0.000068],
    [0.000055, 0.000115, 0.000078, 0.000042, 0.000125, 0.000060],
    [0.000068, 0.000088, 0.000075, 0.000062, 0.000095, 0.000055]])
E7 = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
RHO7 = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
SIG7 = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])
TOTAL7 = 60000


@st.cache_data
def _solve_bai7(n_gen=120):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as moo_min
    from pymoo.termination import get_termination

    grdp = np.array([810, 3580, 1820, 420, 3050, 1409], float)
    region_min = grdp / grdp.sum() * TOTAL7 * 0.50
    cat_min = np.array([8000, 10000, 12000, 8000])

    class Prob(ElementwiseProblem):
        def __init__(s):
            super().__init__(n_var=24, n_obj=4, n_ieq_constr=12,
                             xl=np.zeros(24), xu=np.ones(24) * 12000)

        def _evaluate(s, x, out, *a, **k):
            X = x.reshape(6, 4)
            f1 = -(BETA7 * X).sum()
            sums = X.sum(1); f2 = np.abs(sums - sums.mean()).mean()
            f3 = (E7 * (X[:, 0] + X[:, 2])).sum()
            f4 = (RHO7 * X[:, 2]).sum() - (SIG7 * X[:, 3]).sum()
            out["F"] = [f1, f2, f3, f4]
            g = [X.sum() - TOTAL7, 0.85 * TOTAL7 - X.sum()]
            for i in range(6):
                g.append(region_min[i] - X[i].sum())
            for j in range(4):
                g.append(cat_min[j] - X[:, j].sum())
            out["G"] = g

    res = moo_min(Prob(), NSGA2(pop_size=100), get_termination("n_gen", n_gen),
                  seed=42, verbose=False)
    return res.F, res.X


def page_bai7():
    header("Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II",
           "4 mục tiêu xung đột: tăng trưởng, bao trùm, môi trường, an ninh dữ liệu.")
    with st.spinner("Đang chạy NSGA-II (pop=100, gen=120)..."):
        F, X = _solve_bai7()
    GDP = -F[:, 0]; Gini = F[:, 1]; CO2 = F[:, 2]; Sec = F[:, 3]
    st.metric("Số nghiệm Pareto", len(F))

    # TOPSIS chọn nghiệm thỏa hiệp
    w = np.array([0.40, 0.25, 0.20, 0.15])
    norm = np.sqrt((F**2).sum(0)); V = F / norm * w
    Ap = V.min(0); An = V.max(0)
    dp = np.sqrt(((V - Ap)**2).sum(1)); dn = np.sqrt(((V - An)**2).sum(1))
    C = dn / (dp + dn)
    bi = int(np.argmax(C)); gi = int(np.argmax(GDP))

    st.markdown("#### 7.4.2 — Đường biên Pareto: Tăng trưởng vs Bao trùm")
    dfp = pd.DataFrame({"GDP gain": GDP, "Bất bình đẳng (Gini)": Gini,
                        "CO₂": CO2, "An ninh": Sec})
    fig = px.scatter(dfp, x="GDP gain", y="Bất bình đẳng (Gini)", color="CO₂",
                     color_continuous_scale="plasma", template="plotly_dark",
                     title="Trade-off tăng trưởng ↔ bao trùm (màu = phát thải)")
    fig.add_trace(go.Scatter(x=[GDP[bi]], y=[Gini[bi]], mode="markers",
                             marker=dict(size=16, color=ACCENT, symbol="star"),
                             name="Nghiệm thỏa hiệp (TOPSIS)"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 7.4.3 — Nghiệm thỏa hiệp (trọng số 40/25/20/15)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP gain", f"{GDP[bi]:.3f}")
    c2.metric("Bất bình đẳng", f"{Gini[bi]:.1f}")
    c3.metric("Phát thải CO₂", f"{CO2[bi]:.1f}")
    c4.metric("Rủi ro an ninh", f"{Sec[bi]:.1f}")

    # Radar so sánh nghiệm GDP cao nhất vs thỏ hiệp
    def nrm(v):
        mins = np.array([GDP.min(), Gini.min(), CO2.min(), Sec.min()])
        maxs = np.array([GDP.max(), Gini.max(), CO2.max(), Sec.max()])
        out = np.zeros(4)
        out[0] = (v[0] - mins[0]) / (maxs[0] - mins[0] + 1e-9)
        for i in (1, 2, 3):
            out[i] = 1 - (v[i] - mins[i]) / (maxs[i] - mins[i] + 1e-9)
        return out
    cats = ["Tăng trưởng", "Bao trùm", "Môi trường", "An ninh"]
    rt = nrm([GDP[bi], Gini[bi], CO2[bi], Sec[bi]])
    rg = nrm([GDP[gi], Gini[gi], CO2[gi], Sec[gi]])
    figr = go.Figure()
    figr.add_trace(go.Scatterpolar(r=list(rt) + [rt[0]], theta=cats + [cats[0]],
                                   fill="toself", name="Thỏa hiệp",
                                   line_color=ACCENT))
    figr.add_trace(go.Scatterpolar(r=list(rg) + [rg[0]], theta=cats + [cats[0]],
                                   fill="toself", name="GDP cao nhất",
                                   line_color="#4a9eff"))
    figr.update_layout(template="plotly_dark", title="So sánh đánh đổi đa mục tiêu",
                       polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(figr, use_container_width=True)


# ============================================================
#  BÀI 8 — Tối ưu động liên thời gian 2026-2035 (SLSQP)
# ============================================================
from scipy.optimize import minimize as sp_min

A8 = dict(alpha=0.33, beta=0.42, g=0.10, d=0.08, th=0.07,
          dK=0.05, dD=0.12, dAI=0.15, thH=0.8, mu=0.02,
          phi1=0.003, phi2=0.002, phi3=0.004, rho=0.97, gcr=1.5)
T8 = 10
K08, L08, D08, AI08, H08, Y08 = 27500.0, 53.9, 20.3, 86.0, 30.0, 12847.6


@st.cache_data
def _solve_bai8():
    A0 = Y08 / (K08**A8["alpha"] * L08**A8["beta"] * D08**A8["g"] *
                AI08**A8["d"] * H08**A8["th"])
    L = np.array([L08 * 1.009**t for t in range(T8 + 1)])

    def traj(u):
        IK, ID, IAI, IH = u[0::4], u[1::4], u[2::4], u[3::4]
        K = np.zeros(T8 + 1); D = np.zeros(T8 + 1); AI = np.zeros(T8 + 1)
        H = np.zeros(T8 + 1); Adyn = np.zeros(T8 + 1); Y = np.zeros(T8 + 1)
        C = np.zeros(T8)
        K[0], D[0], AI[0], H[0], Adyn[0] = K08, D08, AI08, H08, A0
        for t in range(T8):
            Y[t] = Adyn[t] * K[t]**A8["alpha"] * L[t]**A8["beta"] * \
                D[t]**A8["g"] * AI[t]**A8["d"] * H[t]**A8["th"]
            C[t] = Y[t] - IK[t] - ID[t] - IAI[t] - IH[t]
            if C[t] <= 0:
                return None
            K[t+1] = (1 - A8["dK"]) * K[t] + IK[t]
            D[t+1] = (1 - A8["dD"]) * D[t] + ID[t]
            AI[t+1] = (1 - A8["dAI"]) * AI[t] + IAI[t]
            H[t+1] = H[t] + A8["thH"] * IH[t] - A8["mu"] * H[t]
            Adyn[t+1] = Adyn[t] * (1 + A8["phi1"]*(D[t]/100) +
                                   A8["phi2"]*(AI[t]/100) + A8["phi3"]*(H[t]/100))
        Y[T8] = Adyn[T8] * K[T8]**A8["alpha"] * L[T8]**A8["beta"] * \
            D[T8]**A8["g"] * AI[T8]**A8["d"] * H[T8]**A8["th"]
        return K, D, AI, H, Y, C, Adyn

    def welfare(u):
        r = traj(u)
        if r is None or np.any(r[5] <= 0):
            return 1e15
        C = r[5]
        return -sum(A8["rho"]**t * (C[t]**(1 - A8["gcr"]) - 1) / (1 - A8["gcr"])
                    for t in range(T8))

    ti = 14000 * 0.15
    u0 = np.zeros(T8 * 4)
    for t in range(T8):
        u0[t*4:t*4+4] = [ti*0.40, ti*0.25, ti*0.20, ti*0.15]

    def cons(u):
        r = traj(u)
        return -1e10 if r is None else min(r[5]) - 1
    res = sp_min(welfare, u0, method="SLSQP", bounds=[(0, None)] * (T8 * 4),
                 constraints=[{"type": "ineq", "fun": cons}],
                 options={"maxiter": 600, "ftol": 1e-8})
    return traj(res.x), -res.fun, res.x


def page_bai8():
    header("Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
           "Thiết kế quỹ đạo phân bổ vốn dài hạn tối đa hóa phúc lợi xã hội.")
    with st.spinner("Đang tối ưu SLSQP..."):
        (K, D, AI, H, Y, C, Adyn), W, u = _solve_bai8()
    years = np.arange(2026, 2037)
    st.metric("Phúc lợi tổng W*", f"{W:.2f}")

    st.markdown("#### 8.3.2 — Quỹ đạo tối ưu K, D, AI, H, Y, C")
    df = pd.DataFrame({"Năm": years, "K": K.round(0), "D": D.round(1),
                       "AI": AI.round(1), "H": H.round(1), "Y": Y.round(0),
                       "TFP": Adyn.round(2)})
    st.dataframe(df, hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=years, y=Y, name="Y (GDP)", line=dict(color=ACCENT)))
    fig1.add_trace(go.Scatter(x=years[:T8], y=C, name="C (tiêu dùng)",
                              line=dict(color="#4a9eff")))
    fig1.update_layout(template="plotly_dark", title="GDP & tiêu dùng",
                       yaxis_title="ngh.tỷ VND")
    c1.plotly_chart(fig1, use_container_width=True)
    fig2 = go.Figure()
    for col, nm in [(K, "K"), (D*100, "D×100"), (AI*10, "AI×10"), (H*100, "H×100")]:
        fig2.add_trace(go.Scatter(x=years, y=col, name=nm))
    fig2.update_layout(template="plotly_dark", title="Tích lũy vốn (đã scale)")
    c2.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Tỷ lệ đầu tư theo năm (% GDP)")
    inv = u.reshape(T8, 4)
    df_inv = pd.DataFrame(inv / Y[:T8, None] * 100, columns=["IK", "ID", "IAI", "IH"])
    df_inv["Năm"] = years[:T8]
    fig3 = px.bar(df_inv, x="Năm", y=["IK", "ID", "IAI", "IH"],
                  template="plotly_dark", title="Cơ cấu đầu tư/GDP (%)",
                  color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(fig3, use_container_width=True)


# ============================================================
#  BÀI 9 — Tác động AI tới lao động (NetJob LP)
# ============================================================
SECT9 = ["Nông-LT", "CN chế biến", "Xây dựng", "Bán buôn-bán lẻ",
         "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục-ĐT"]
L9 = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
RISK9 = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
A1_9 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
B1_9 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
C1_9 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
D1_9 = np.array([50, 32, 42, 38, 26, 36, 24, 62])


@st.cache_data
def _solve_bai9(cap5=False):
    N = 8
    coeff = A1_9 - C1_9 * RISK9
    c_obj = np.concatenate([-coeff, -B1_9])
    A1 = np.concatenate([np.ones(N), np.ones(N)]).reshape(1, -1)
    A1b = np.concatenate([-np.ones(N), np.zeros(N)]).reshape(1, -1)
    A2 = np.zeros((N, 2*N)); A3 = np.zeros((N, 2*N))
    for i in range(N):
        A2[i, i] = -coeff[i]; A2[i, N+i] = -B1_9[i]
        A3[i, i] = C1_9[i] * RISK9[i]; A3[i, N+i] = -D1_9[i]
    A_ub = np.vstack([A1, A1b, A2, A3])
    b_ub = np.concatenate([[30000], [-9000], np.zeros(N), np.zeros(N)])
    if cap5:
        A4 = np.zeros((N, 2*N))
        for i in range(N):
            A4[i, i] = C1_9[i] * RISK9[i]
        A_ub = np.vstack([A_ub, A4]); b_ub = np.concatenate([b_ub, 0.05 * L9 * 1e6])
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * (2*N), method="highs")
    if not res.success:
        return None
    xA, xH = res.x[:N], res.x[N:]
    return xA, xH, coeff * xA + B1_9 * xH, C1_9 * RISK9 * xA, -res.fun


def page_bai9():
    header("Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
           "Phân bổ 30.000 tỷ cho AI & đào tạo để tối đa hóa việc làm ròng NetJob.")
    xA, xH, NJ, Disp, tot = _solve_bai9()
    st.metric("Tổng NetJob ròng", f"{tot:,.0f} việc làm")

    st.markdown("#### 9.4.1 — Phân bổ tối ưu & NetJob theo ngành")
    df = pd.DataFrame({"Ngành": SECT9, "x_AI": xA.round(0), "x_H": xH.round(0),
                       "Displaced": Disp.round(0), "NetJob": NJ.round(0)})
    c1, c2 = st.columns([1.2, 1])
    c1.dataframe(df, hide_index=True, use_container_width=True)
    fig = px.bar(df, x="Ngành", y="NetJob", template="plotly_dark",
                 color="NetJob", color_continuous_scale="reds",
                 title="Việc làm ròng theo ngành")
    fig.update_layout(xaxis_tickangle=-40)
    c2.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 9.4.3 — Cơ cấu đầu tư AI vs đào tạo (H)")
    fig2 = px.bar(df, x="Ngành", y=["x_AI", "x_H"], barmode="stack",
                  template="plotly_dark", color_discrete_sequence=[ACCENT, "#4a9eff"],
                  title="Phân bổ ngân sách theo ngành")
    fig2.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 9.4.4 — Thêm ràng buộc Displaced ≤ 5% lao động")
    r5 = _solve_bai9(cap5=True)
    if r5:
        c1, c2 = st.columns(2)
        c1.metric("NetJob (không ràng buộc)", f"{tot:,.0f}")
        c2.metric("NetJob (ràng buộc 5%)", f"{r5[4]:,.0f}",
                  f"-{(tot-r5[4])/tot*100:.1f}%")
    else:
        st.warning("Bài toán không khả thi với ràng buộc 5%.")


# ============================================================
#  BÀI 10 — Quy hoạch ngẫu nhiên 2 giai đoạn (port sang linprog)
# ============================================================
J10 = ["I", "D", "AI", "H"]
S10 = ["s1", "s2", "s3", "s4"]
P10 = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
BBASE10 = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
BS10 = {
    ("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
    ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
    ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
    ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10,
}


def _solve_sp10():
    # Biến: x_I,x_D,x_AI,x_H + y[s,j] (4 + 16 = 20). Tối đa hóa.
    nx, ny = 4, 16
    n = nx + ny
    c = np.zeros(n)
    for k, j in enumerate(J10):
        c[k] = -BBASE10[j]
    for si, s in enumerate(S10):
        for k, j in enumerate(J10):
            c[nx + si*4 + k] = -P10[s] * BS10[(s, j)]
    A_ub, b_ub = [], []
    # Σx ≤ 65000
    row = np.zeros(n); row[:4] = 1; A_ub.append(row); b_ub.append(65000)
    # Σy_s ≤ 15000 ∀s ; y_AI^s ≤ 0.5 x_H
    for si in range(4):
        r = np.zeros(n); r[nx + si*4: nx + si*4 + 4] = 1
        A_ub.append(r); b_ub.append(15000)
        r2 = np.zeros(n); r2[nx + si*4 + 2] = 1; r2[3] = -0.5
        A_ub.append(r2); b_ub.append(0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * n, method="highs")
    x = {j: res.x[k] for k, j in enumerate(J10)}
    y = {s: {j: res.x[nx + si*4 + k] for k, j in enumerate(J10)}
         for si, s in enumerate(S10)}
    return x, y, -res.fun


def _solve_det10(s):
    # Một kịch bản duy nhất, prob=1
    nx, ny = 4, 4
    n = nx + ny
    c = np.zeros(n)
    for k, j in enumerate(J10):
        c[k] = -BBASE10[j]; c[nx + k] = -BS10[(s, j)]
    A_ub, b_ub = [], []
    row = np.zeros(n); row[:4] = 1; A_ub.append(row); b_ub.append(65000)
    r = np.zeros(n); r[nx:nx+4] = 1; A_ub.append(r); b_ub.append(15000)
    r2 = np.zeros(n); r2[nx + 2] = 1; r2[3] = -0.5; A_ub.append(r2); b_ub.append(0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * n, method="highs")
    return {j: res.x[k] for k, j in enumerate(J10)}, -res.fun


def page_bai10():
    header("Bài 10 — Quy hoạch ngẫu nhiên 2 giai đoạn dưới bất định",
           "Quyết định ngân sách first-stage robust trước 4 kịch bản, đo VSS & EVPI.")
    x_sp, y_sp, Z_SP = _solve_sp10()
    det = {s: _solve_det10(s) for s in S10}

    # EV solution: beta trung bình
    beta_avg = {j: sum(P10[s] * BS10[(s, j)] for s in S10) for j in J10}
    c_ev = np.zeros(4)
    for k, j in enumerate(J10):
        c_ev[k] = -beta_avg[j]
    res_ev = linprog(c_ev, A_ub=[[1, 1, 1, 1]], b_ub=[65000],
                     bounds=[(0, None)] * 4, method="highs")
    x_ev = {j: res_ev.x[k] for k, j in enumerate(J10)}
    # Z_EV: dùng x_ev cố định, tối ưu y mỗi kịch bản
    Z_EV = sum(BBASE10[j] * x_ev[j] for j in J10)
    for s in S10:
        # max Σ βs y, y≤15000, y_AI ≤ 0.5 x_H
        cc = np.array([-BS10[(s, j)] for j in J10])
        A = [[1, 1, 1, 1]]; b = [15000]
        A.append([0, 0, 1, 0]); b.append(0.5 * x_ev["H"])
        r = linprog(cc, A_ub=A, b_ub=b, bounds=[(0, None)] * 4, method="highs")
        Z_EV += P10[s] * (-r.fun)
    Z_WS = sum(P10[s] * det[s][1] for s in S10)
    VSS = Z_SP - Z_EV
    EVPI = Z_WS - Z_SP

    st.markdown("#### 10.5.1 — Quyết định first-stage tối ưu (SP)")
    df = pd.DataFrame({"Hạng mục": ["Hạ tầng I", "Số hóa D", "AI", "Nhân lực H"],
                       "x* (tỷ VND)": [x_sp[j] for j in J10]})
    c1, c2 = st.columns([1, 1])
    c1.metric("Z* Stochastic", f"{Z_SP:,.0f}")
    c1.dataframe(df.round(0), hide_index=True, use_container_width=True)
    fig = px.pie(df, names="Hạng mục", values="x* (tỷ VND)", hole=0.4,
                 template="plotly_dark", color_discrete_sequence=px.colors.sequential.Reds_r,
                 title="Cơ cấu phân bổ first-stage")
    c2.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 10.5.3 — VSS & EVPI")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Z_SP (Stochastic)", f"{Z_SP:,.0f}")
    c2.metric("Z_EV (Expected Value)", f"{Z_EV:,.0f}")
    c3.metric("VSS", f"{VSS:,.0f}", f"{VSS/Z_SP*100:.2f}%")
    c4.metric("EVPI", f"{EVPI:,.0f}", f"{EVPI/Z_SP*100:.2f}%")
    st.info("**VSS > 0**: việc cân nhắc bất định khi quyết định có giá trị. "
            "**EVPI**: giá trị của thông tin hoàn hảo về kịch bản tương lai.")

    st.markdown("#### So sánh nghiệm xác định từng kịch bản")
    dfd = pd.DataFrame({"Kịch bản": [f"{s} (p={P10[s]})" for s in S10],
                        "Z*": [det[s][1] for s in S10]})
    fig2 = px.bar(dfd, x="Kịch bản", y="Z*", template="plotly_dark",
                  color="Z*", color_continuous_scale="reds")
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
#  BÀI 11 — Q-learning chính sách kinh tế (MDP thuần Python)
# ============================================================
ALLOC11 = {0: np.array([0.70, 0.10, 0.10, 0.10]),
           1: np.array([0.40, 0.25, 0.15, 0.20]),
           2: np.array([0.25, 0.45, 0.15, 0.15]),
           3: np.array([0.20, 0.20, 0.45, 0.15]),
           4: np.array([0.30, 0.20, 0.10, 0.40])}
ACT11 = ["Truyền thống", "Cân bằng", "Số hóa nhanh", "AI dẫn dắt", "Bao trùm"]
W11 = np.array([0.40, 0.25, 0.20, 0.15])


class _EconEnv:
    """MDP port thuần Python (không cần gymnasium)."""
    T = 10

    def reset(self, rng, state=None):
        self.state = np.array(state) if state is not None else rng.integers(0, 3, 4)
        self.t = 0
        self.K, self.D, self.AI, self.H, self.Yprev = 27500., 20.3, 86., 30., 12847.6
        return self.state.copy()

    def step(self, a):
        al = ALLOC11[a]; b = 2100.
        self.K = 0.95 * self.K + al[0] * b
        self.D = 0.88 * self.D + al[1] * b * 0.01
        self.AI = 0.85 * self.AI + al[2] * b * 0.05
        self.H = self.H + 0.8 * al[3] * b * 0.01 - 0.02 * self.H
        A = 33.70 * (1 + 0.003*(self.D/100) + 0.002*(self.AI/100) +
                     0.004*(self.H/100))**self.t
        L = 53.9 * 1.009**self.t
        Y = A * self.K**0.33 * L**0.42 * self.D**0.10 * self.AI**0.08 * self.H**0.07
        dg = (Y - self.Yprev) / self.Yprev
        du = max(0, -dg * 0.5)
        cyber = (self.AI / (self.H + 1)) * 0.01
        emis = (self.K + self.AI) * 0.0001
        r = W11[0]*dg*100 - W11[1]*du*100 - W11[2]*cyber - W11[3]*emis
        self.Yprev = Y; self.t += 1
        gl = 0 if dg < 0.03 else (1 if dg < 0.06 else 2)
        dl = 0 if self.D < 25 else (1 if self.D < 35 else 2)
        ail = 0 if self.AI < 100 else (1 if self.AI < 200 else 2)
        hl = 0 if self.H < 35 else (1 if self.H < 50 else 2)
        self.state = np.array([gl, dl, ail, hl])
        return self.state.copy(), r, self.t >= self.T


@st.cache_resource
def _train_q(n_ep=8000):
    rng = np.random.default_rng(0)
    env = _EconEnv()
    Q = np.zeros((3, 3, 3, 3, 5))
    hist = []
    for ep in range(n_ep):
        s = env.reset(rng); tot = 0
        eps = max(0.05, 1.0 - ep / 5000)
        while True:
            a = rng.integers(5) if rng.random() < eps else int(np.argmax(Q[tuple(s)]))
            s2, r, done = env.step(a)
            Q[tuple(s) + (a,)] += 0.1 * (r + 0.95*np.max(Q[tuple(s2)])*(1-done)
                                         - Q[tuple(s) + (a,)])
            tot += r; s = s2
            if done:
                break
        hist.append(tot)
    return Q, hist


def page_bai11():
    header("Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
           "Mô hình nền kinh tế VN như MDP, học chính sách phân bổ ngân sách thích nghi.")
    with st.spinner("Đang huấn luyện Q-learning (8.000 episodes)..."):
        Q, hist = _train_q()

    st.markdown("#### 11.3.4 — Learning curve")
    w = 200
    sm = np.convolve(hist, np.ones(w)/w, mode="valid")
    fig = px.line(x=np.arange(len(sm)), y=sm, template="plotly_dark",
                  labels={"x": "Episode", "y": "Phúc lợi tích lũy"},
                  title="Đường cong học (smoothed)")
    fig.update_traces(line_color=ACCENT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 11.3.3 — Chính sách π*(s) tại các trạng thái khởi đầu")
    tests = [([1, 1, 0, 1], "VN 2026 thực tế"),
             ([0, 0, 0, 2], "Kịch bản tệ"),
             ([2, 2, 2, 2], "Kịch bản tốt"),
             ([0, 1, 0, 0], "Sau khủng hoảng"),
             ([1, 0, 2, 1], "AI mạnh, D yếu")]
    rows = []
    for s, desc in tests:
        a = int(np.argmax(Q[tuple(s)]))
        rows.append({"Trạng thái": desc, "π* (hành động)": ACT11[a]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("#### So sánh π* với chính sách rule-based")
    rng = np.random.default_rng(1)
    env = _EconEnv()

    def evalp(fn, n=300):
        rs = []
        for _ in range(n):
            s = env.reset(rng); tot = 0
            while True:
                s, r, d = env.step(fn(s)); tot += r
                if d:
                    break
            rs.append(tot)
        return np.mean(rs)
    pols = {"π* (Q-learning)": lambda s: int(np.argmax(Q[tuple(s)])),
            "Luôn Cân bằng": lambda s: 1, "Luôn AI dẫn dắt": lambda s: 3,
            "Random": lambda s: rng.integers(5)}
    dfc = pd.DataFrame({"Chính sách": list(pols), "Phúc lợi TB": [evalp(f) for f in pols.values()]})
    fig2 = px.bar(dfc, x="Chính sách", y="Phúc lợi TB", template="plotly_dark",
                  color="Phúc lợi TB", color_continuous_scale="reds")
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
#  BÀI 12 — Đồ án tích hợp: dashboard 5 kịch bản chính sách
# ============================================================
SCEN12 = {
    "S1 — Truyền thống": {"K": 0.70, "D": 0.10, "AI": 0.10, "H": 0.10},
    "S2 — Số hóa nhanh": {"K": 0.25, "D": 0.45, "AI": 0.15, "H": 0.15},
    "S3 — AI dẫn dắt": {"K": 0.20, "D": 0.20, "AI": 0.45, "H": 0.15},
    "S4 — Bao trùm số": {"K": 0.30, "D": 0.20, "AI": 0.10, "H": 0.40},
    "S5 — Tối ưu cân bằng": {"K": 0.25, "D": 0.25, "AI": 0.30, "H": 0.20},
}


def _forecast12(alloc, T=4):
    a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
    K, D, AI, H, A, L0 = 27500., 20.3, 86., 30., 33.70, 53.9
    budget = 3000
    traj = [A * K**a * L0**b * D**g * AI**d * H**th]
    for t in range(T):
        K = 0.95*K + alloc["K"]*budget
        D = 0.88*D + alloc["D"]*budget*0.01
        AI = 0.85*AI + alloc["AI"]*budget*0.05
        H = H + 0.8*alloc["H"]*budget*0.01 - 0.02*H
        A = A * (1 + 0.003*(D/100) + 0.002*(AI/100) + 0.004*(H/100))
        L = L0 * 1.009**(t+1)
        traj.append(A * K**a * L**b * D**g * AI**d * H**th)
    return traj


def _netjob_scaled(alloc):
    """NetJob tổng theo trọng số AI/H của kịch bản (dùng lõi Bài 9)."""
    share_ai = alloc["AI"] / (alloc["AI"] + alloc["H"]) if (alloc["AI"]+alloc["H"]) > 0 else 0.5
    budget = 30000
    xAI_total = budget * share_ai
    xH_total = budget * (1 - share_ai)
    # phân bổ đều theo ngành tỷ lệ với hệ số hiệu quả
    coeff = A1_9 - C1_9 * RISK9
    wAI = coeff / coeff.sum()
    wH = B1_9 / B1_9.sum()
    xAI = xAI_total * wAI; xH = xH_total * wH
    return (coeff * xAI + B1_9 * xH).sum()


def page_bai12():
    header("Bài 12 — Đồ án tích hợp AIDEOM-VN",
           "Dashboard hỗ trợ ra quyết định: thử nghiệm 5 kịch bản chính sách thời gian thực.")
    scen = st.selectbox("🎛️ Chọn kịch bản chính sách", list(SCEN12.keys()))
    alloc = SCEN12[scen]

    st.markdown("##### Tỷ lệ phân bổ ngân sách")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vốn vật chất K", f"{alloc['K']*100:.0f}%")
    c2.metric("Hạ tầng số D", f"{alloc['D']*100:.0f}%")
    c3.metric("Công nghệ AI", f"{alloc['AI']*100:.0f}%")
    c4.metric("Nhân lực số H", f"{alloc['H']*100:.0f}%")
    st.divider()

    cc = st.columns(3)
    # (1) Pie cơ cấu vốn
    df_pie = pd.DataFrame({"Hạng mục": ["Vốn K", "Hạ tầng D", "AI", "Nhân lực H"],
                           "Tỷ lệ": [alloc["K"], alloc["D"], alloc["AI"], alloc["H"]]})
    fig_pie = px.pie(df_pie, names="Hạng mục", values="Tỷ lệ", hole=0.45,
                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.Reds_r,
                     title="Cơ cấu phân bổ vốn")
    cc[0].plotly_chart(fig_pie, use_container_width=True)

    # (2) Radar đa mục tiêu (proxy chuẩn hóa)
    growth = alloc["K"]*0.4 + alloc["D"]*0.7 + alloc["AI"]*0.9 + alloc["H"]*0.5
    inclus = alloc["H"]*1.0 + alloc["D"]*0.4
    green = 1 - (alloc["K"]*0.6 + alloc["AI"]*0.7)
    secure = alloc["H"]*0.8 - alloc["AI"]*0.3 + 0.5
    radar = np.clip([growth, inclus, green, secure], 0, 1.2)
    cats = ["Tăng trưởng", "Bao trùm", "Xanh hóa", "An ninh"]
    fig_r = go.Figure(go.Scatterpolar(r=list(radar) + [radar[0]],
                                      theta=cats + [cats[0]], fill="toself",
                                      line_color=ACCENT))
    fig_r.update_layout(template="plotly_dark", title="Đánh đổi đa mục tiêu",
                        polar=dict(radialaxis=dict(range=[0, 1.2])))
    cc[1].plotly_chart(fig_r, use_container_width=True)

    # (3) Bar NetJob
    nj = _netjob_scaled(alloc)
    fig_nj = go.Figure(go.Bar(x=["NetJob ròng"], y=[nj], marker_color=ACCENT,
                              text=[f"{nj:,.0f}"], textposition="outside"))
    fig_nj.update_layout(template="plotly_dark", title="Việc làm ròng (NetJob)",
                         yaxis_title="việc làm")
    cc[2].plotly_chart(fig_nj, use_container_width=True)

    st.divider()
    st.markdown("##### Dự báo GDP 2026-2030 (Cobb-Douglas — Module M1)")
    years = list(range(2026, 2031))
    traj = _forecast12(alloc)
    df_g = pd.DataFrame({"Năm": years, "GDP (ngh.tỷ VND)": np.round(traj, 0)})
    c1, c2 = st.columns([1, 2])
    c1.metric("GDP 2030 dự báo", f"{traj[-1]:,.0f} ngh.tỷ")
    c1.dataframe(df_g, hide_index=True, use_container_width=True)
    fig_g = px.line(df_g, x="Năm", y="GDP (ngh.tỷ VND)", markers=True,
                    template="plotly_dark", title=f"Quỹ đạo GDP — {scen}")
    fig_g.update_traces(line_color=ACCENT)
    c2.plotly_chart(fig_g, use_container_width=True)

    st.markdown("##### So sánh GDP 2030 giữa 5 kịch bản")
    comp = pd.DataFrame({"Kịch bản": list(SCEN12),
                         "GDP 2030": [round(_forecast12(a)[-1]) for a in SCEN12.values()]})
    fig_c = px.bar(comp, x="Kịch bản", y="GDP 2030", template="plotly_dark",
                   color="GDP 2030", color_continuous_scale="reds")
    st.plotly_chart(fig_c, use_container_width=True)


# ============================================================
#  ROUTER
# ============================================================
ROUTER = {
    PAGES[0]: page_home, PAGES[1]: page_bai1, PAGES[2]: page_bai2,
    PAGES[3]: page_bai3, PAGES[4]: page_bai4, PAGES[5]: page_bai5,
    PAGES[6]: page_bai6, PAGES[7]: page_bai7, PAGES[8]: page_bai8,
    PAGES[9]: page_bai9, PAGES[10]: page_bai10, PAGES[11]: page_bai11,
    PAGES[12]: page_bai12,
}
ROUTER[choice]()
