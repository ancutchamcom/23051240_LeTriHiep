# ============================================================
#  AIDEOM-VN — Hệ thống Mô hình Ra quyết định Phát triển
#  Kinh tế Việt Nam trong Kỷ nguyên AI
#  Giao diện kiểm tra bài tập cho Giảng viên chấm điểm
#  Sinh viên: Lê Trí Hiệp
#  Viện Quản trị Kinh doanh — Đại học Kinh tế, ĐHQGHN
#  Chạy:  streamlit run app.py
#
#  GHI CHÚ: Mỗi bài chạy TRỰC TIẾP code gốc từ notebook
#  (.ipynb) trong thư mục modules/, tái hiện y nguyên kết quả
#  in ra (print) và các biểu đồ matplotlib/seaborn gốc.
# ============================================================

import os
import io
import contextlib

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as mfig

# Plotly chỉ dùng cho dashboard tương tác Bài 12
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AIDEOM-VN", page_icon="🇻🇳",
                   layout="wide", initial_sidebar_state="expanded")

ACCENT = "#ff4d6d"
HERE = os.path.dirname(os.path.abspath(__file__))
MOD_DIR = os.path.join(HERE, "modules")

# Cho module gốc đọc CSV bằng đường dẫn tương đối
os.chdir(HERE)


# ============================================================
#  HÀM NẠP DỮ LIỆU CHUNG (3 CSV gốc)
# ============================================================
def _find(fname):
    for cand in (os.path.join(HERE, fname), os.path.join(HERE, "data", fname), fname):
        if os.path.exists(cand):
            return cand
    return fname


@st.cache_data
def load_macro():
    return pd.read_csv(_find("vietnam_macro_2020_2025.csv")).sort_values(
        "year").reset_index(drop=True)


@st.cache_data
def load_sectors():
    return pd.read_csv(_find("vietnam_sectors_2024.csv"))


@st.cache_data
def load_regions():
    return pd.read_csv(_find("vietnam_regions_2024.csv"))


# ============================================================
#  RUNNER — chạy nguyên cell notebook, thu output + figure gốc
# ============================================================
@st.cache_data(show_spinner=False)
def run_notebook_cell(module_name):
    """Exec code gốc của 1 bài, trả về (text_output, [png_bytes...]).

    Figure được chụp NGAY tại savefig() để giữ đúng thứ tự & nội dung
    như chạy trên notebook gốc.
    """
    path = os.path.join(MOD_DIR, module_name)
    captured = []
    orig_savefig = mfig.Figure.savefig

    def cap_savefig(self, *a, **k):
        b = io.BytesIO()
        try:
            orig_savefig(self, b, format="png", dpi=120,
                         bbox_inches="tight", facecolor=self.get_facecolor())
            captured.append(b.getvalue())
        except Exception:
            pass

    saved = (mfig.Figure.savefig, plt.savefig, plt.show, plt.close)
    mfig.Figure.savefig = cap_savefig
    plt.savefig = lambda *a, **k: cap_savefig(plt.gcf())
    plt.show = lambda *a, **k: None
    plt.close = lambda *a, **k: None  # giữ figure để chụp

    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf):
            code = open(path, "r", encoding="utf-8").read()
            exec(compile(code, path, "exec"), {"__name__": "__main__"})
    except Exception:
        import traceback
        err = traceback.format_exc()
    finally:
        mfig.Figure.savefig, plt.savefig, plt.show, plt.close = saved
        plt.close("all")

    text = buf.getvalue()
    if err:
        text += "\n\n[LỖI KHI CHẠY CELL]\n" + err
    return text, captured


def render_cell(title, goal, module_name, spinner):
    st.markdown(f"## {title}")
    st.markdown(f"<span style='color:#9aa0a6'>🎯 <b>Mục tiêu:</b> {goal}</span>",
                unsafe_allow_html=True)
    st.divider()

    with st.spinner(spinner):
        text, imgs = run_notebook_cell(module_name)

    if imgs:
        st.markdown("#### 📊 Biểu đồ (tái hiện y nguyên từ notebook)")
        for png in imgs:
            st.image(png, use_container_width=True)
    else:
        st.info("Bài này không xuất biểu đồ trong notebook gốc — xem kết quả số bên dưới.")

    st.markdown("#### 🖥️ Kết quả tính toán (output gốc)")
    st.code(text or "(không có output)", language="text")


# ============================================================
#  CSS
# ============================================================
st.markdown(f"""
<style>
.stApp {{ background:#0e1117; }}
h1,h2,h3 {{ color:#fafafa; }}
div[data-testid="stMetricValue"] {{ color:{ACCENT}; }}
.sub {{ color:#9aa0a6; font-size:0.9rem; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
#  SIDEBAR
# ============================================================
st.sidebar.markdown(f"<h2 style='color:{ACCENT}'>🇻🇳 AIDEOM-VN</h2>",
                    unsafe_allow_html=True)
st.sidebar.markdown("<span class='sub'>Mô hình ra quyết định phát triển kinh tế "
                    "VN trong kỉ nguyên AI</span>", unsafe_allow_html=True)
st.sidebar.divider()

PAGES = [
    "🏠 Trang chủ Tổng quan Hệ thống",
    "🔹 Bài 1", "🔹 Bài 2", "🔹 Bài 3", "🔹 Bài 4", "🔹 Bài 5", "🔹 Bài 6",
    "🔹 Bài 7", "🔹 Bài 8", "🔹 Bài 9", "🔹 Bài 10", "🔹 Bài 11", "🚀 Bài 12",
]
choice = st.sidebar.radio("Điều hướng", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown(
    "<span class='sub'><b>Sinh viên:</b> Lê Trí Hiệp<br>"
    "Viện Quản trị Kinh doanh<br>Đại học Kinh tế, ĐHQGHN</span>",
    unsafe_allow_html=True)


# ============================================================
#  TRANG CHỦ
# ============================================================
def page_home():
    st.markdown(f"<h1 style='color:{ACCENT}'>AIDEOM-VN</h1>", unsafe_allow_html=True)
    st.markdown("#### AI-Driven Economic Decision Optimization Model for Vietnam")
    st.write("Hệ thống giải **12 bài toán mô hình ra quyết định** phát triển kinh "
             "tế Việt Nam trong kỷ nguyên AI — dữ liệu thực 2020–2025 (GSO, MoST, "
             "MIC, MPI, World Bank, GII). Chính sách tham chiếu: Nghị quyết "
             "57-NQ/TW, các QĐ 749/127/411/QĐ-TTg.")

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
    for name, df in [("vietnam_macro_2020_2025.csv", macro),
                     ("vietnam_sectors_2024.csv", load_sectors()),
                     ("vietnam_regions_2024.csv", load_regions())]:
        st.markdown(f"**{name}** — Shape: `{df.shape[0]} dòng × {df.shape[1]} cột`")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🗂️ 12 bài toán theo 4 cấp độ")
    tiers = pd.DataFrame({
        "Cấp độ": ["DỄ"]*3 + ["TB"]*3 + ["KHÁ KHÓ"]*3 + ["KHÓ"]*3,
        "Bài": [f"Bài {i}" for i in range(1, 13)],
        "Nội dung": [
            "Cobb-Douglas mở rộng + AI — TFP & growth accounting",
            "LP phân bổ ngân sách 4 hạng mục — scipy, shadow price",
            "Chỉ số ưu tiên 10 ngành — min-max norm, weighted scoring",
            "LP phân bổ ngân sách ngành-vùng — PuLP + CVXPY",
            "MIP lựa chọn 15 dự án — PuLP/CBC",
            "TOPSIS xếp hạng 6 vùng — Entropy + AHP",
            "NSGA-II Pareto 4 mục tiêu — pymoo + TOPSIS",
            "Tối ưu động liên thời gian 2026-2035 — SLSQP",
            "Tác động AI tới lao động — NetJob LP",
            "Quy hoạch ngẫu nhiên 2 giai đoạn — VSS/EVPI",
            "Q-learning chính sách kinh tế thích nghi — MDP",
            "Đồ án tích hợp 6 module + dashboard 5 kịch bản",
        ],
    })
    st.dataframe(tiers, use_container_width=True, hide_index=True)


# ============================================================
#  METADATA BÀI 1–11
# ============================================================
BAI_META = {
    "🔹 Bài 1": ("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng + AI",
                 "Ước lượng TFP, phân rã tăng trưởng GDP, dự báo GDP 2030.", "bai01.py"),
    "🔹 Bài 2": ("Bài 2 — LP phân bổ ngân sách 4 hạng mục đầu tư số",
                 "Tối đa hóa tăng GDP kỳ vọng từ 100 ngh.tỷ VND, phân tích shadow price.", "bai02.py"),
    "🔹 Bài 3": ("Bài 3 — Chỉ số ưu tiên ngành cho 10 ngành Việt Nam",
                 "Xếp hạng ngành ưu tiên chuyển đổi số/AI bằng weighted scoring.", "bai03.py"),
    "🔹 Bài 4": ("Bài 4 — LP phân bổ ngân sách số ngành-vùng",
                 "Phân bổ 50.000 tỷ cho 6 vùng × 4 hạng mục, đảm bảo công bằng vùng.", "bai04.py"),
    "🔹 Bài 5": ("Bài 5 — MIP lựa chọn dự án chuyển đổi số",
                 "Chọn tập dự án tối ưu trong 15 ứng cử với ràng buộc ngân sách & cấu trúc.", "bai05.py"),
    "🔹 Bài 6": ("Bài 6 — TOPSIS xếp hạng 6 vùng ưu tiên đầu tư AI",
                 "Xếp hạng vùng triển khai trung tâm AI bằng TOPSIS + Entropy + AHP.", "bai06.py"),
    "🔹 Bài 7": ("Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II",
                 "4 mục tiêu xung đột: tăng trưởng, bao trùm, môi trường, an ninh dữ liệu.", "bai07.py"),
    "🔹 Bài 8": ("Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
                 "Thiết kế quỹ đạo phân bổ vốn dài hạn tối đa hóa phúc lợi xã hội.", "bai08.py"),
    "🔹 Bài 9": ("Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
                 "Phân bổ 30.000 tỷ cho AI & đào tạo để tối đa hóa việc làm ròng NetJob.", "bai09.py"),
    "🔹 Bài 10": ("Bài 10 — Quy hoạch ngẫu nhiên 2 giai đoạn dưới bất định",
                  "Quyết định first-stage robust trước 4 kịch bản, đo VSS & EVPI.", "bai10.py"),
    "🔹 Bài 11": ("Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
                  "Mô hình nền kinh tế VN như MDP, học chính sách phân bổ thích nghi.", "bai11.py"),
}

SPIN = {
    "🔹 Bài 7": "Đang chạy NSGA-II (pop=100, gen=200)... ~1 phút lần đầu",
    "🔹 Bài 11": "Đang huấn luyện Q-learning (10.000 episodes)... ~30s lần đầu",
}


# ============================================================
#  BÀI 12 — chạy module M1-M4 gốc + dashboard 5 kịch bản
# ============================================================
SCEN12 = {
    "S1 — Truyền thống": {"K": 0.70, "D": 0.10, "AI": 0.10, "H": 0.10},
    "S2 — Số hóa nhanh": {"K": 0.25, "D": 0.45, "AI": 0.15, "H": 0.15},
    "S3 — AI dẫn dắt": {"K": 0.20, "D": 0.20, "AI": 0.45, "H": 0.15},
    "S4 — Bao trùm số": {"K": 0.30, "D": 0.20, "AI": 0.10, "H": 0.40},
    "S5 — Tối ưu cân bằng": {"K": 0.25, "D": 0.25, "AI": 0.30, "H": 0.20},
}
_A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
_B1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
_C1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
_RISK = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100


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
        A = A*(1 + 0.003*(D/100) + 0.002*(AI/100) + 0.004*(H/100))
        L = L0*1.009**(t+1)
        traj.append(A * K**a * L**b * D**g * AI**d * H**th)
    return traj


def _netjob12(alloc):
    s_ai = alloc["AI"]/(alloc["AI"]+alloc["H"]) if (alloc["AI"]+alloc["H"]) > 0 else 0.5
    coeff = _A1 - _C1*_RISK
    xAI = 30000*s_ai*(coeff/coeff.sum())
    xH = 30000*(1-s_ai)*(_B1/_B1.sum())
    return (coeff*xAI + _B1*xH).sum()


def page_bai12():
    st.markdown("## Bài 12 — Đồ án tích hợp AIDEOM-VN")
    st.markdown("<span style='color:#9aa0a6'>🎯 <b>Mục tiêu:</b> Tích hợp 6 module "
                "(M1–M6) và dashboard hỗ trợ ra quyết định 5 kịch bản chính sách."
                "</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🎛️ Dashboard ra quyết định — 5 kịch bản chính sách (Module M6)")
    scen = st.selectbox("Chọn kịch bản", list(SCEN12.keys()))
    alloc = SCEN12[scen]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vốn vật chất K", f"{alloc['K']*100:.0f}%")
    c2.metric("Hạ tầng số D", f"{alloc['D']*100:.0f}%")
    c3.metric("Công nghệ AI", f"{alloc['AI']*100:.0f}%")
    c4.metric("Nhân lực số H", f"{alloc['H']*100:.0f}%")

    cc = st.columns(3)
    df_pie = pd.DataFrame({"Hạng mục": ["Vốn K", "Hạ tầng D", "AI", "Nhân lực H"],
                           "Tỷ lệ": [alloc["K"], alloc["D"], alloc["AI"], alloc["H"]]})
    fig_pie = px.pie(df_pie, names="Hạng mục", values="Tỷ lệ", hole=0.45,
                     template="plotly_dark",
                     color_discrete_sequence=px.colors.sequential.Reds_r,
                     title="Cơ cấu phân bổ vốn")
    cc[0].plotly_chart(fig_pie, use_container_width=True)

    growth = alloc["K"]*0.4 + alloc["D"]*0.7 + alloc["AI"]*0.9 + alloc["H"]*0.5
    inclus = alloc["H"]*1.0 + alloc["D"]*0.4
    green = 1 - (alloc["K"]*0.6 + alloc["AI"]*0.7)
    secure = alloc["H"]*0.8 - alloc["AI"]*0.3 + 0.5
    radar = np.clip([growth, inclus, green, secure], 0, 1.2)
    cats = ["Tăng trưởng", "Bao trùm", "Xanh hóa", "An ninh"]
    fig_r = go.Figure(go.Scatterpolar(r=list(radar)+[radar[0]],
                                      theta=cats+[cats[0]], fill="toself",
                                      line_color=ACCENT))
    fig_r.update_layout(template="plotly_dark", title="Đánh đổi đa mục tiêu",
                        polar=dict(radialaxis=dict(range=[0, 1.2])))
    cc[1].plotly_chart(fig_r, use_container_width=True)

    nj = _netjob12(alloc)
    fig_nj = go.Figure(go.Bar(x=["NetJob ròng"], y=[nj], marker_color=ACCENT,
                              text=[f"{nj:,.0f}"], textposition="outside"))
    fig_nj.update_layout(template="plotly_dark", title="Việc làm ròng (NetJob)",
                         yaxis_title="việc làm")
    cc[2].plotly_chart(fig_nj, use_container_width=True)

    traj = _forecast12(alloc)
    df_g = pd.DataFrame({"Năm": list(range(2026, 2031)),
                         "GDP (ngh.tỷ VND)": np.round(traj, 0)})
    fig_g = px.line(df_g, x="Năm", y="GDP (ngh.tỷ VND)", markers=True,
                    template="plotly_dark", title=f"Dự báo GDP — {scen}")
    fig_g.update_traces(line_color=ACCENT)
    st.plotly_chart(fig_g, use_container_width=True)

    comp = pd.DataFrame({"Kịch bản": list(SCEN12),
                         "GDP 2030": [round(_forecast12(a)[-1]) for a in SCEN12.values()]})
    fig_c = px.bar(comp, x="Kịch bản", y="GDP 2030", template="plotly_dark",
                   color="GDP 2030", color_continuous_scale="reds",
                   title="So sánh GDP 2030 giữa 5 kịch bản")
    st.plotly_chart(fig_c, use_container_width=True)

    st.divider()
    st.markdown("### 🖥️ Kết quả tích hợp 6 module (output gốc từ notebook)")
    with st.spinner("Đang chạy pipeline M1–M4..."):
        text, imgs = run_notebook_cell("bai12.py")
    for png in imgs:
        st.image(png, use_container_width=True)
    st.code(text or "(không có output)", language="text")


# ============================================================
#  ROUTER
# ============================================================
if choice == PAGES[0]:
    page_home()
elif choice == "🚀 Bài 12":
    page_bai12()
else:
    title, goal, mod = BAI_META[choice]
    render_cell(title, goal, mod, SPIN.get(choice, "Đang chạy code gốc từ notebook..."))
