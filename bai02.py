import os
OUTPUT_DIR = '.'  # Trên Colab, lưu vào /content
os.makedirs(OUTPUT_DIR, exist_ok=True) if OUTPUT_DIR not in (".","") else None
# ============================================================
# BÀI TOÁN 2.4 - TỐI ƯU HÓA PHÂN BỔ NGÂN SÁCH AI QUỐC GIA
# Môn: Các Mô Hình Ra Quyết Định
# Yêu cầu: scipy + numpy + matplotlib (không cần pulp)
# ============================================================
#
# Biến quyết định:
#   x1: Hạ tầng số & Điện toán đám mây  (nghìn tỷ VND)
#   x2: Nghiên cứu & Phát triển AI       (nghìn tỷ VND)
#   x3: Đào tạo Nhân lực số             (nghìn tỷ VND)
#   x4: Ứng dụng AI vào các ngành       (nghìn tỷ VND)
#
# Hàm mục tiêu: Tối đa hóa Z = 0.85x1 + 1.20x2 + 0.95x3 + 1.35x4
# Ràng buộc:
#   (R1) x1+x2+x3+x4 <= 100         Ngân sách tổng
#   (R2) x1 >= 25                    Hạ tầng tối thiểu
#   (R3) x2 >= 15                    R&D tối thiểu
#   (R4) x3 >= 20                    Nhân lực tối thiểu
#   (R5) x4 >= 10                    Ứng dụng tối thiểu
#   (R6) 0.35(x1+x3) <= 0.65(x2+x4) Cân bằng hạ tầng/ứng dụng
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog

print("=" * 65)
print("  BÀI TOÁN PHÂN BỔ NGÂN SÁCH AI QUỐC GIA VIỆT NAM 2024")
print("=" * 65)

var_names  = ['x1 (Hạ tầng số)', 'x2 (R&D AI)', 'x3 (Nhân lực)', 'x4 (Ứng dụng AI)']
rois       = [0.85, 1.20, 0.95, 1.35]
con_labels = [
    'R1: Ngân sách tổng  (<=100)',
    'R2: Hạ tầng tối thiểu (x1>=25)',
    'R3: R&D tối thiểu    (x2>=15)',
    'R4: Nhân lực tối thiểu(x3>=20)',
    'R5: Ứng dụng tối thiểu(x4>=10)',
    'R6: Cân bằng hạ tầng/ứng dụng',
]

# --- Hàm solver dùng scipy HiGHS ---
def solve_lp(budget=100, x3_min=20, verbose=False):
    """
    Trả về (success, Z_star, x_opt, dual_values)
    dual_values: mảng shadow price theo thứ tự [R1..R6]
    """
    c     = [-0.85, -1.20, -0.95, -1.35]
    A_ub  = [
        [ 1,     1,     1,     1   ],   # R1: tổng <= budget
        [-1,     0,     0,     0   ],   # R2: x1 >= 25
        [ 0,    -1,     0,     0   ],   # R3: x2 >= 15
        [ 0,     0,    -1,     0   ],   # R4: x3 >= x3_min
        [ 0,     0,     0,    -1   ],   # R5: x4 >= 10
        [ 0.35, -0.65,  0.35, -0.65],  # R6: cân bằng
    ]
    b_ub  = [budget, -25, -15, -x3_min, -10, 0]
    bounds = [(0, None)] * 4
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success:
        # marginals của ineqlin = shadow prices (âm vì linprog minimize)
        # Để có shadow price theo nghĩa maximize, đổi dấu
        duals = -res.ineqlin.marginals  # shape (6,)
        return True, -res.fun, res.x, duals
    else:
        return False, None, None, None

# ============================================================
# CÂU 2.4.1 - SCIPY.OPTIMIZE.LINPROG
# ============================================================
print("\n" + "═"*65)
print("  CÂU 2.4.1 — SCIPY.OPTIMIZE.LINPROG")
print("═"*65)

ok, Z_base, x_base, duals_base = solve_lp(budget=100, x3_min=20)

print(f"\n✅ Trạng thái: {'Tìm được nghiệm tối ưu' if ok else 'Không khả thi'}")
print(f"\n📊 Phân bổ ngân sách tối ưu (B = 100 nghìn tỷ VND):")
print(f"   {'─'*47}")
for name, roi, val in zip(var_names, rois, x_base):
    print(f"   {name:25s}: {val:6.2f} nghìn tỷ  (ROI={roi})")
print(f"   {'─'*47}")
print(f"   {'Tổng':25s}: {sum(x_base):6.2f} nghìn tỷ")
print(f"\n🎯 Giá trị hàm mục tiêu tối ưu: Z* = {Z_base:.4f}")

print(f"\n📋 Kiểm tra ràng buộc:")
A_ub_check = [
    [ 1,     1,     1,     1   ],
    [-1,     0,     0,     0   ],
    [ 0,    -1,     0,     0   ],
    [ 0,     0,    -1,     0   ],
    [ 0,     0,     0,    -1   ],
    [ 0.35, -0.65,  0.35, -0.65],
]
b_ub_check = [100, -25, -15, -20, -10, 0]
ops        = ['<=','<=','<=','<=','<=','<=']
for label, row, bval, op in zip(con_labels, A_ub_check, b_ub_check, ops):
    lhs = np.dot(row, x_base)
    ok_c = lhs <= bval + 1e-6
    print(f"   {'✓' if ok_c else '✗'}  {label}: {lhs:.4f} {op} {bval}")

# ============================================================
# CÂU 2.4.2 - GIÁ ĐỐI NGẪU (DUAL VALUES / SHADOW PRICES)
# ============================================================
print("\n" + "═"*65)
print("  CÂU 2.4.2 — GIÁ ĐỐI NGẪU (SHADOW PRICES)")
print("═"*65)

print(f"\n💡 Giá đối ngẫu của từng ràng buộc (B=100, x3≥20):")
print(f"   {'─'*68}")
print(f"   {'Ràng buộc':<35}  {'Shadow Price':>12}  Slack")
print(f"   {'─'*68}")
A_ub_mat  = np.array([
    [ 1,     1,     1,     1   ],
    [-1,     0,     0,     0   ],
    [ 0,    -1,     0,     0   ],
    [ 0,     0,    -1,     0   ],
    [ 0,     0,     0,    -1   ],
    [ 0.35, -0.65,  0.35, -0.65],
])
b_ub_base = np.array([100, -25, -15, -20, -10, 0])
slacks    = b_ub_base - A_ub_mat @ x_base

for label, sp, sl in zip(con_labels, duals_base, slacks):
    binding = "● binding" if abs(sl) < 1e-4 else f"  slack={sl:.2f}"
    print(f"   {label:<35}  {sp:>12.4f}  {binding}")
print(f"   {'─'*68}")

sp_budget = duals_base[0]
print(f"""
🔍 Giải thích Shadow Price ràng buộc ngân sách tổng = {sp_budget:.4f}:

   ➤ Ý nghĩa kỹ thuật:
     Nếu tăng ngân sách B thêm 1 nghìn tỷ VND (trong vùng ổn định),
     giá trị tối ưu Z* sẽ tăng thêm {sp_budget:.4f} đơn vị.

   ➤ Ý nghĩa chính sách:
     • Shadow price = {sp_budget:.4f} > ROI thấp nhất (x1=0.85) cho thấy
       ngân sách đang là ràng buộc "binding" — còn dư địa hiệu quả.
     • Mỗi đồng ngân sách bổ sung tạo ra {sp_budget:.4f} đơn vị giá trị,
       tương đương ROI của danh mục tối ưu biên (x4, ROI=1.35).
     • Chính phủ nên ưu tiên xét tăng tổng ngân sách AI hơn là
       tái cơ cấu nội bộ — lợi ích biên vẫn dương và đáng kể.
     • Nếu shadow price = 0 thì ngân sách không còn là ràng buộc
       cần nới lỏng (đã đủ dư địa).""")

# ============================================================
# CÂU 2.4.3 - PHÂN TÍCH ĐỘ NHẠY: ĐƯỜNG CONG Z*(B)
# ============================================================
print("\n" + "═"*65)
print("  CÂU 2.4.3 — PHÂN TÍCH ĐỘ NHẠY: ĐƯỜNG CONG Z*(B)")
print("═"*65)

budgets = list(range(70, 181, 5))
results = {}
for B in budgets:
    ok_b, zv, xv, dv = solve_lp(budget=B, x3_min=20)
    results[B] = (ok_b, zv, xv, dv)

print(f"\n📊 Kết quả tại các mốc đặc biệt:")
print(f"   {'B':>6} | {'Z*':>10} | {'x1':>6} | {'x2':>6} | {'x3':>6} | {'x4':>6} | {'SP(B)':>8}")
print(f"   {'─'*65}")
for B_key in [100, 120, 140]:
    ok_b, zv, xv, dv = results[B_key]
    if ok_b:
        print(f"   {B_key:>6} | {zv:>10.4f} | {xv[0]:>6.2f} | {xv[1]:>6.2f} | {xv[2]:>6.2f} | {xv[3]:>6.2f} | {dv[0]:>8.4f}")

print(f"\n📈 Nhận xét độ nhạy:")
z100 = results[100][1]; z120 = results[120][1]; z140 = results[140][1]
print(f"   Tăng B từ 100→120 (+20):  ΔZ* = {z120-z100:.4f}  (bình quân {(z120-z100)/20:.4f}/đv)")
print(f"   Tăng B từ 120→140 (+20):  ΔZ* = {z140-z120:.4f}  (bình quân {(z140-z120)/20:.4f}/đv)")

# --- Vẽ đồ thị ---
valid = [(B, *results[B][1:]) for B in budgets if results[B][0]]
bv  = [r[0] for r in valid]
zv  = [r[1] for r in valid]
xvs = np.array([r[2] for r in valid])   # shape (n,4)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Phân tích Độ nhạy: Ngân sách B → Z*(B)\nBài toán Phân bổ Ngân sách AI Quốc gia Việt Nam',
             fontsize=13, fontweight='bold', y=1.01)

# Plot trái: Đường cong Z*(B)
ax1 = axes[0]
ax1.plot(bv, zv, 'b-o', linewidth=2.5, markersize=5, label='Z*(B)', zorder=3)
for B_key, col in zip([100, 120, 140], ['#e74c3c','#f39c12','#27ae60']):
    if B_key in bv:
        idx = bv.index(B_key)
        ax1.scatter([B_key], [zv[idx]], color=col, s=150, zorder=5)
        ax1.annotate(f'B={B_key}\nZ*={zv[idx]:.2f}',
                     xy=(B_key, zv[idx]), xytext=(B_key+3, zv[idx]-1.5),
                     fontsize=9, color=col, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
ax1.set_xlabel('Ngân sách tổng B (nghìn tỷ VND)', fontsize=11)
ax1.set_ylabel('Giá trị mục tiêu tối ưu Z*', fontsize=11)
ax1.set_title('Đường cong Z*(B)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3); ax1.set_facecolor('#f8f9fa')

# Plot phải: Phân bổ từng biến theo B
ax2 = axes[1]
bar_colors  = ['#3498db','#9b59b6','#2ecc71','#e67e22']
labels_plot = ['x1: Hạ tầng','x2: R&D AI','x3: Nhân lực','x4: Ứng dụng']
for i, (col, lab) in enumerate(zip(bar_colors, labels_plot)):
    ax2.plot(bv, xvs[:,i], '-o', color=col, linewidth=2, markersize=4, label=lab)
for B_key in [100,120,140]:
    ax2.axvline(x=B_key, color='gray', linestyle='--', alpha=0.4)
ax2.set_xlabel('Ngân sách tổng B (nghìn tỷ VND)', fontsize=11)
ax2.set_ylabel('Phân bổ (nghìn tỷ VND)', fontsize=11)
ax2.set_title('Phân bổ tối ưu theo B', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left'); ax2.grid(True, alpha=0.3); ax2.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Đã lưu: sensitivity_analysis.png")

# ============================================================
# CÂU 2.4.4 - RÀNG BUỘC ƯU TIÊN NHÂN LỰC SỐ: x3 >= 30
# ============================================================
print("\n" + "═"*65)
print("  CÂU 2.4.4 — RÀNG BUỘC BỔ SUNG: x3 ≥ 30 (Ưu tiên Nhân lực số)")
print("═"*65)

ok_30, Z_30, x_30, duals_30 = solve_lp(budget=100, x3_min=30)

print(f"\n✅ Bài toán với x3 ≥ 30 (B=100): {'KHẢ THI' if ok_30 else 'KHÔNG KHẢ THI'}")

if ok_30:
    print(f"\n📊 Phân bổ tối ưu (x3 ≥ 30):")
    for name, roi, val in zip(var_names, rois, x_30):
        print(f"   {name:25s}: {val:6.2f} nghìn tỷ  (ROI={roi})")
    delta = Z_30 - Z_base
    print(f"""
🎯 So sánh kết quả:
   Z* (x3 ≥ 20) = {Z_base:.4f}
   Z* (x3 ≥ 30) = {Z_30:.4f}
   ΔZ*           = {delta:.4f}  ({delta/Z_base*100:.2f}%)

📋 Phân tích tác động chính sách:
   ➤ Bài toán VẪN KHẢ THI — ràng buộc x3≥30 không làm vô nghiệm.
     Kiểm tra: x1_min+x2_min+x3_min+x4_min = 25+15+30+10 = 80 ≤ 100 ✓
   ➤ Z* giảm {abs(delta):.4f} đơn vị ({abs(delta)/Z_base*100:.2f}%) — đây là CHI PHÍ CƠ HỘI
     của chính sách ưu tiên nhân lực số.
   ➤ Nguyên nhân: buộc phân bổ thêm vào x3 (ROI=0.95) thay vì
     x4 (ROI=1.35), tức bỏ qua lợi ích chênh 0.40/đv.
   ➤ Shadow price của ràng buộc x3≥30 = {duals_30[3]:.4f}
     → Chi phí biên để thêm 1 đv x3: mất {abs(duals_30[3]):.4f} đv Z.
   ➤ Khuyến nghị chính sách:
     • Nếu ưu tiên x3 là bắt buộc (thiếu kỹ sư AI là bottleneck),
       chấp nhận thiệt hại {abs(delta):.4f} đv ngắn hạn để đảm bảo
       năng lực nhân lực dài hạn.
     • Xem xét tăng tổng ngân sách (B→110-120) để bù đắp phần
       thiệt hại Z* thay vì đánh đổi nội bộ.""")
else:
    print("\n❌ Bài toán KHÔNG KHẢ THI — các ràng buộc mâu thuẫn nhau.")

# --- Đồ thị so sánh ---
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('So sánh: Bài toán gốc (x3 ≥ 20) vs Ưu tiên Nhân lực (x3 ≥ 30)\nB = 100 nghìn tỷ VND',
              fontsize=12, fontweight='bold', y=1.01)

bar_colors  = ['#3498db','#9b59b6','#2ecc71','#e67e22']
short_names = ['Hạ tầng\n(x1)','R&D AI\n(x2)','Nhân lực\n(x3)','Ứng dụng\n(x4)']
cases = [
    ('x3 ≥ 20  (Gốc)',         x_base, Z_base, '#27ae60', [25,15,20,10]),
    ('x3 ≥ 30  (Ưu tiên NL)', x_30,   Z_30,   '#e74c3c', [25,15,30,10]),
]
for ax, (label, xvals, zval, col, mins) in zip(axes2, cases):
    if xvals is not None:
        bars = ax.bar(short_names, xvals, color=bar_colors, alpha=0.85,
                      edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, xvals):
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.4,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.set_title(f'{label}\nZ* = {zval:.4f}', fontsize=12, fontweight='bold', color=col)
        ax.set_ylabel('Phân bổ (nghìn tỷ VND)', fontsize=10)
        ax.set_ylim(0, max(xvals)*1.25)
        ax.grid(axis='y', alpha=0.3); ax.set_facecolor('#f8f9fa')
        for i, (bar, mn) in enumerate(zip(bars, mins)):
            ax.axhline(y=mn, xmin=(i+0.1)/4, xmax=(i+0.9)/4,
                       color='red', linestyle='--', alpha=0.7, linewidth=1.8,
                       label='_ràng buộc min' if i==0 else '_')
        ax.text(0.98, 0.98, 'Đường đỏ: mức tối thiểu', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='red', alpha=0.8)
    else:
        ax.text(0.5,0.5,'KHÔNG KHẢ THI', ha='center', va='center',
                transform=ax.transAxes, fontsize=16, color='red', fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/comparison_x3_constraint.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Đã lưu: comparison_x3_constraint.png")

# ============================================================
# TÓM TẮT
# ============================================================
print("\n" + "═"*65)
print("  TÓM TẮT KẾT QUẢ TOÀN BỘ BÀI TOÁN 2.4")
print("═"*65)
print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  Câu 2.4.1 (scipy linprog):                                  │
  │    Z* = {Z_base:.4f}                                            │
  │    x1={x_base[0]:.1f}, x2={x_base[1]:.1f}, x3={x_base[2]:.1f}, x4={x_base[3]:.1f}               │
  ├──────────────────────────────────────────────────────────────┤
  │  Câu 2.4.2 (dual values từ HiGHS marginals):                │
  │    Shadow price ngân sách R1 = {duals_base[0]:.4f}                   │
  │    → Tăng 1 nghìn tỷ ngân sách → Z* tăng {duals_base[0]:.4f}        │
  ├──────────────────────────────────────────────────────────────┤
  │  Câu 2.4.3 (phân tích độ nhạy):                             │
  │    B=100 → Z* = {results[100][1]:.4f}                               │
  │    B=120 → Z* = {results[120][1]:.4f}  (+{results[120][1]-results[100][1]:.4f})                 │
  │    B=140 → Z* = {results[140][1]:.4f}  (+{results[140][1]-results[100][1]:.4f})                 │
  ├──────────────────────────────────────────────────────────────┤
  │  Câu 2.4.4 (x3 ≥ 30): Vẫn KHẢ THI                          │
  │    Z* = {Z_30:.4f}  (giảm {abs(Z_30-Z_base):.4f} = {abs(Z_30-Z_base)/Z_base*100:.2f}%)              │
  └──────────────────────────────────────────────────────────────┘
""")
print("✅ Hoàn thành tất cả các câu bài toán 2.4!")