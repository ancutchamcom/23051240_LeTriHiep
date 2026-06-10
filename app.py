# ============================================================
#  AIDEOM-VN — Hệ thống Mô hình Ra quyết định Phát triển
#  Kinh tế Việt Nam trong Kỷ nguyên AI
#  Giao diện kiểm tra bài tập cho Giảng viên chấm điểm
#  Sinh viên: Lê Trí Hiệp
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
    "🔹 Bài 1",... (Còn 9 KB)
