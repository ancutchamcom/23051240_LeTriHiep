"""
╔══════════════════════════════════════════════════════════════╗
║  BÀI TOÁN LỰA CHỌN DỰ ÁN ĐẦU TƯ CÔNG VIỆT NAM             ║
║  Môn: Các Mô Hình Ra Quyết Định                              ║
║  Giải pháp ILP bằng PuLP + Solver CBC                        ║
╠══════════════════════════════════════════════════════════════╣
║  5.4.1 – Bài toán gốc (80k / 40k tỷ VND)                    ║
║  5.4.2 – Nới ngân sách GĐ1 lên 100.000 tỷ                   ║
║  5.4.3 – Bắt buộc P1 VÀ P2 (redundancy Quốc hội)            ║
║  5.4.4 – Mở rộng rủi ro: Tối đa E[Z] = Σ pᵢ·Bᵢ·yᵢ         ║
╚══════════════════════════════════════════════════════════════╝
"""

import pulp
from pulp import (
    LpProblem, LpMaximize, LpVariable, LpStatus,
    lpSum, value, PULP_CBC_CMD
)

print("✔  PuLP đã sẵn sàng. Phiên bản:", pulp.__version__)

# ════════════════════════════════════════════════════════════════
# 1. DỮ LIỆU BÀI TOÁN – 15 dự án (P1..P15)
# ════════════════════════════════════════════════════════════════
P = list(range(1, 16))

# Chi phí giai đoạn 1 (tỷ VND)
C = {
    1: 12000, 2: 11500, 3: 18000, 4:  4500,  5: 3200,
    6:  5800, 7:  6500, 8: 15000, 9:  2500, 10: 7200,
   11:  4800,12:  8500,13: 20000,14:  3800, 15: 1500
}
# Chi phí giai đoạn 2 (tỷ VND)
C1 = {
    1:  8500, 2:  7500, 3: 12000, 4:  3500,  5: 2500,
    6:  4000, 7:  4500, 8:  9000, 9:  1800, 10: 5000,
   11:  3500,12:  5500,13: 13000,14:  2800, 15: 1200
}
# Lợi ích kinh tế – xã hội (tỷ VND)
B = {
    1: 21500, 2: 20800, 3: 32500, 4:  9200,  5: 6800,
    6: 11400, 7: 12200, 8: 28500, 9:  5800, 10:13800,
   11:  8500,12: 16200,13: 35000,14:  7500, 15: 3800
}

# ── Xác suất hoàn thành đúng tiến độ (Câu 5.4.4) ────────────
#   Hạ tầng        P3, P8, P14    → p = 0.85
#   Chính phủ số   P1, P2, P12   → p = 0.75
#   AI / Bán dẫn   P7, P10, P13  → p = 0.65
#   Còn lại                       → p = 0.80
_INFRA  = {3, 8, 14}
_EGOV   = {1, 2, 12}
_AI     = {7, 10, 13}
p_risk = {
    i: 0.85 if i in _INFRA else
       0.75 if i in _EGOV  else
       0.65 if i in _AI    else 0.80
    for i in P
}

# ════════════════════════════════════════════════════════════════
# 2. HÀM XÂY DỰNG MÔ HÌNH ILP TỔNG QUÁT
# ════════════════════════════════════════════════════════════════
def build_model(budget1=80_000, budget2=40_000,
                force_p1_and_p2=False, use_expected=False):
    """
    Xây dựng mô hình ILP lựa chọn dự án.

    Parameters
    ----------
    budget1         : Ngân sách GĐ1 tối đa (tỷ VND)
    budget2         : Ngân sách GĐ2 tối đa (tỷ VND)
    force_p1_and_p2 : True → bắt buộc y[1]=y[2]=1 (Câu 5.4.3)
    use_expected    : True → hàm mục tiêu là E[Z] = Σ pᵢBᵢyᵢ (Câu 5.4.4)

    Returns
    -------
    (LpProblem, dict[int, LpVariable])
    """
    m = LpProblem("VN_Project_Selection", LpMaximize)
    y = LpVariable.dicts("y", P, cat="Binary")

    # ── Hàm mục tiêu ────────────────────────────────────────────
    if use_expected:
        m += lpSum(p_risk[i] * B[i] * y[i] for i in P), "Max_E_Z"
    else:
        m += lpSum(B[i] * y[i] for i in P), "Max_Z"

    # ── Ngân sách ───────────────────────────────────────────────
    m += lpSum(C[i]  * y[i] for i in P) <= budget1, "Budget_GD1"
    m += lpSum(C1[i] * y[i] for i in P) <= budget2, "Budget_GD2"

    # ── Ràng buộc cấu trúc ──────────────────────────────────────
    if force_p1_and_p2:
        m += y[1] == 1, "Force_P1"            # Câu 5.4.3
        m += y[2] == 1, "Force_P2"
    else:
        m += y[1] + y[2] <= 1, "MutualExcl_P1_P2"   # Loại trừ nhau

    m += y[8]  <= y[12], "Depend_P8_req_P12"  # P8 cần P12
    m += y[13] <= y[12], "Depend_P13_req_P12" # P13 cần P12

    m += y[4] + y[5] >= 1, "Region_P4orP5"    # Ưu tiên vùng
    m += y[14] >= 1,        "Mandatory_P14"    # Cam kết quốc tế

    m += lpSum(y[i] for i in P) >= 7,  "Min_7_Projects"
    m += lpSum(y[i] for i in P) <= 11, "Max_11_Projects"

    return m, y

# ════════════════════════════════════════════════════════════════
# 3. HÀM GIẢI VÀ IN BÁO CÁO
# ════════════════════════════════════════════════════════════════
SEP  = "═" * 64
SEP2 = "─" * 64

def solve_and_report(m, y, title, use_expected=False):
    """Giải mô hình và in báo cáo chi tiết."""
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

    m.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[m.status]
    print(f"  Trạng thái solver  : {status}")

    if status != "Optimal":
        print("  ⚠  KHÔNG TÌM ĐƯỢC LỜI GIẢI TỐI ƯU!")
        if status == "Infeasible":
            # Phân tích lý do không khả thi
            c1_p1p2 = C[1] + C[2]
            c2_p1p2 = C1[1] + C1[2]
            print(f"     Chi phí P1+P2 riêng: GĐ1={c1_p1p2:,} | GĐ2={c2_p1p2:,} tỷ")
            print(f"     Kèm ràng buộc P14 + P4/P5 + min 7 dự án,")
            print(f"     ngân sách GĐ2 bị vượt khi buộc cả P1 và P2.")
        return None

    selected = [i for i in P if y[i].value() > 0.5]
    tc1 = sum(C[i]  for i in selected)
    tc2 = sum(C1[i] for i in selected)
    tb  = sum(B[i]  for i in selected)
    z   = value(m.objective)
    npv = z / (tc1 + tc2)

    print(f"  Dự án được chọn   : {[f'P{i}' for i in selected]}")
    print(f"  Số lượng dự án    : {len(selected)}")
    print(f"{SEP2}")
    print(f"  {'Tổng chi phí GĐ1 (tỷ VND)':<30}: {tc1:>12,.0f}")
    print(f"  {'Tổng chi phí GĐ2 (tỷ VND)':<30}: {tc2:>12,.0f}")
    print(f"  {'Tổng chi phí (tỷ VND)':<30}: {tc1+tc2:>12,.0f}")
    print(f"{SEP2}")
    print(f"  {'Tổng lợi ích B (tỷ VND)':<30}: {tb:>12,.0f}")
    if use_expected:
        ez = sum(p_risk[i] * B[i] for i in selected)
        print(f"  {'Lợi ích kỳ vọng E[Z]* (tỷ)':<30}: {ez:>12,.1f}")
        print(f"  {'NPV biên E[Z]*/(C+C1)':<30}: {ez/(tc1+tc2):>12.4f}")
    else:
        print(f"  {'Tổng lợi ích Z* (tỷ VND)':<30}: {z:>12,.0f}")
        print(f"  {'NPV biên Z*/(C+C1)':<30}: {npv:>12.4f}")
    print(f"{SEP2}")

    # ── Bảng chi tiết từng dự án được chọn ─────────────────────
    print(f"\n  {'Dự án':<6} {'C GĐ1':>9} {'C GĐ2':>9} "
          f"{'B':>9} {'p':>5} {'E[B]':>9}")
    print(f"  {'------':<6} {'-'*9} {'-'*9} {'-'*9} {'-'*5} {'-'*9}")
    for i in selected:
        print(f"  P{i:<5} {C[i]:>9,} {C1[i]:>9,} "
              f"{B[i]:>9,} {p_risk[i]:>5.2f} {p_risk[i]*B[i]:>9,.0f}")

    return z, selected

# ════════════════════════════════════════════════════════════════
# CÂU 5.4.1 – BÀI TOÁN GỐC
# ════════════════════════════════════════════════════════════════
print("\n" + "█"*64)
print("  PHÂN TÍCH TỐI ƯU HOÁ LỰA CHỌN DỰ ÁN ĐẦU TƯ CÔNG VIỆT NAM")
print("█"*64)

m1, y1 = build_model(budget1=80_000, budget2=40_000)
r1 = solve_and_report(m1, y1,
    "CÂU 5.4.1 – Bài toán gốc  [GĐ1: 80.000 tỷ | GĐ2: 40.000 tỷ]")

# ════════════════════════════════════════════════════════════════
# CÂU 5.4.2 – NỚI NGÂN SÁCH GĐ1 → 100.000 tỷ
# ════════════════════════════════════════════════════════════════
m2, y2 = build_model(budget1=100_000, budget2=40_000)
r2 = solve_and_report(m2, y2,
    "CÂU 5.4.2 – Nới ngân sách  [GĐ1: 100.000 tỷ | GĐ2: 40.000 tỷ]")

if r1 and r2:
    sel1, sel2 = set(r1[1]), set(r2[1])
    added   = sorted(sel2 - sel1)
    removed = sorted(sel1 - sel2)
    dz = r2[0] - r1[0]
    print(f"\n  ▶ Dự án mới thêm   : {[f'P{i}' for i in added]   or ['(không có)']}")
    print(f"  ▶ Dự án bị loại    : {[f'P{i}' for i in removed]  or ['(không có)']}")
    print(f"  ▶ Tăng Z*          : +{dz:,.0f} tỷ VND "
          f"(+{dz/r1[0]*100:.2f}% so với kịch bản gốc)")

# ════════════════════════════════════════════════════════════════
# CÂU 5.4.3 – BẮT BUỘC P1 VÀ P2 (REDUNDANCY)
# ════════════════════════════════════════════════════════════════
m3, y3 = build_model(budget1=80_000, budget2=40_000,
                     force_p1_and_p2=True)
r3 = solve_and_report(m3, y3,
    "CÂU 5.4.3 – Bắt buộc P1 VÀ P2  [GĐ1: 80.000 | GĐ2: 40.000 tỷ]")

if r3 and r1:
    dz3 = r3[0] - r1[0]
    print(f"\n  ▶ Thay đổi Z* so với bài toán gốc: "
          f"{dz3:+,.0f} tỷ ({dz3/r1[0]*100:+.2f}%)")

# ════════════════════════════════════════════════════════════════
# CÂU 5.4.4 – RỦI RO DỰ ÁN: TỐI ĐA E[Z] = Σ pᵢ Bᵢ yᵢ
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  CÂU 5.4.4 – Mở rộng: Rủi ro dự án  [GĐ1: 80.000 | GĐ2: 40.000 tỷ]")
print(SEP)
print("  Xác suất hoàn thành đúng tiến độ (pᵢ):")
print("    Hạ tầng     (P3, P8, P14)         → p = 0.85")
print("    Chính phủ số (P1, P2, P12)        → p = 0.75")
print("    AI/Bán dẫn   (P7, P10, P13)       → p = 0.65")
print("    Còn lại      (P4,P5,P6,P9,P11,P15)→ p = 0.80")

m4, y4 = build_model(budget1=80_000, budget2=40_000,
                     use_expected=True)
r4 = solve_and_report(m4, y4,
    "CÂU 5.4.4 – Tối đa E[Z] = Σ pᵢ·Bᵢ·yᵢ",
    use_expected=True)

if r4 and r1:
    sel4 = set(r4[1])
    added4   = sorted(sel4 - set(r1[1]))
    removed4 = sorted(set(r1[1]) - sel4)
    print(f"\n  ▶ So với Câu 5.4.1 (tất định):")
    print(f"     Thêm vào  : {[f'P{i}' for i in added4]   or ['(không có)']}")
    print(f"     Bỏ ra     : {[f'P{i}' for i in removed4] or ['(không có)']}")
    print(f"\n  ► Phân tích: Khi tính rủi ro, mô hình ưu tiên dự án hạ tầng")
    print(f"    (p=0.85) và hạn chế dự án AI/bán dẫn (p=0.65) dù B danh nghĩa cao.")

# ════════════════════════════════════════════════════════════════
# BẢNG TỔNG KẾT
# ════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  TỔNG KẾT SO SÁNH BỐN KỊCH BẢN")
print(SEP)
print(f"  {'Kịch bản':<44} {'Z* / E[Z]*':>14}  {'# Dự án':>7}")
print(f"  {'-'*44} {'-'*14}  {'-'*7}")

rows = [
    ("5.4.1  Gốc           (GĐ1=80k / GĐ2=40k)",  r1),
    ("5.4.2  Nới ngân sách (GĐ1=100k / GĐ2=40k)", r2),
    ("5.4.3  Bắt buộc P1+P2 (redundancy)",         r3),
    ("5.4.4  Rủi ro E[Z]   (GĐ1=80k / GĐ2=40k)", r4),
]
for lbl, res in rows:
    if res:
        print(f"  {lbl:<44} {res[0]:>14,.0f}  {len(res[1]):>7}")
    else:
        print(f"  {lbl:<44} {'INFEASIBLE':>14}  {'–':>7}")

print(f"\n{SEP}")
print("  ✔  Hoàn thành phân tích tối ưu hoá – PuLP/CBC")
print(SEP)