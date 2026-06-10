import numpy as np
from scipy.optimize import linprog
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# BAI 9: Tac dong AI toi thi truong lao dong VN
# ============================================================

N = 8
sectors = ['Nông-LT', 'CN chế biến', 'Xây dựng', 'Bán buôn-bán lẻ',
           'Tài chính-NH', 'Logistics', 'CNTT-TT', 'Giáo dục-ĐT']

L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100

a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
b1_arr = np.array([45, 28, 35, 32, 22, 30, 20, 55])
c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])

# ============================================================
# Cau 9.4.1: LP toi da hoa tong NetJob
# ============================================================
print("="*60)
print("BÀI 9: Tác động AI tới thị trường lao động")
print("="*60)
print("\nCAU 9.4.1: LP toi da hoa tong NetJob")
print("="*60)

# Bien: [x_AI_0..x_AI_7, x_H_0..x_H_7] = 16 bien
# NetJob_i = NewJob_i + UpgradeJob_i - Displaced_i
#   NewJob = a1 * x_AI  (a2*x_D trong mo hinh ly thuyet, nhung bai toan chi phan bo x_AI va x_H)
#   Displaced = c1 * risk * x_AI
#   UpgradeJob = b1 * x_H  (nang cap ky nang -> giu viec, KHONG phai viec lam moi)
# LUU Y: a2 (he so xD) duoc dinh nghia trong du lieu nhung khong su dung do bai toan
# chi co 2 hang muc dau tu (x_AI, x_H). De mo rong, them bien x_D.
coeff_AI = a1 - c1 * risk
coeff_H = b1_arr.copy()

print("\nHe so hieu qua net (a1 - c1*risk) va he so upgrade (b1):")
print("  LUU Y: Rang buoc x_AI >= 9,000 ty (30% ngan sach) de tranh trivial solution")
for i in range(N):
    retrain_ratio = c1[i]*risk[i] / d1[i]
    print(f"  {sectors[i]:<15}: net_AI={coeff_AI[i]:>6.1f}, b1={b1_arr[i]:>3}, "
          f"c1*risk={c1[i]*risk[i]:>5.1f}, d1={d1[i]:>3}, retrain_ratio={retrain_ratio:.3f}")

c_obj = np.concatenate([-coeff_AI, -coeff_H])

# R1: sum(x_AI + x_H) <= 30000
A1 = np.concatenate([np.ones(N), np.ones(N)]).reshape(1, -1)

# R1b: x_AI floor - tranh trivial solution (x_AI=0, x_H=30000)
# Bat buoc it nhat 30% ngan sach cho AI de dam bao chuyen doi so thuc su
A1b = np.concatenate([-np.ones(N), np.zeros(N)]).reshape(1, -1)

# R2: NetJob_i >= 0
A2 = np.zeros((N, 2*N))
for i in range(N):
    A2[i, i] = -coeff_AI[i]
    A2[i, N+i] = -b1_arr[i]

# R3: Displaced <= RetrainCap: c1_i*risk_i*x_AI_i <= d1_i*x_H_i
A3 = np.zeros((N, 2*N))
for i in range(N):
    A3[i, i] = c1[i] * risk[i]
    A3[i, N+i] = -d1[i]

A_ub = np.vstack([A1, A1b, A2, A3])
b_ub = np.concatenate([[30000], [-9000], np.zeros(N), np.zeros(N)])
bounds = [(0, None)] * (2*N)

res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

x_AI_opt = res.x[:N]
x_H_opt = res.x[N:]
NetJob = coeff_AI * x_AI_opt + b1_arr * x_H_opt
NewJob = a1 * x_AI_opt
Upgrade = b1_arr * x_H_opt
Displaced = c1 * risk * x_AI_opt
RetrainCap = d1 * x_H_opt

print(f"\nTrang thai: {res.message}")
print(f"Tong NetJob = {-res.fun:,.0f} viec lam")
print(f"\n{'Nganh':<15} {'x_AI':>8} {'x_H':>8} {'NewJob':>8} {'Upgrade':>8} "
      f"{'Displace':>8} {'RetrCap':>8} {'NetJob':>8}")
print("-" * 88)
for i in range(N):
    print(f"{sectors[i]:<15} {x_AI_opt[i]:>8.0f} {x_H_opt[i]:>8.0f} {NewJob[i]:>8.0f} "
          f"{Upgrade[i]:>8.0f} {Displaced[i]:>8.0f} {RetrainCap[i]:>8.0f} {NetJob[i]:>8.0f}")
print("-" * 88)
print(f"{'Tong':<15} {x_AI_opt.sum():>8.0f} {x_H_opt.sum():>8.0f} {NewJob.sum():>8.0f} "
      f"{Upgrade.sum():>8.0f} {Displaced.sum():>8.0f} {RetrainCap.sum():>8.0f} {NetJob.sum():>8.0f}")
print(f"\nNgan sach su dung: {x_AI_opt.sum() + x_H_opt.sum():,.0f} / 30,000 ty VND")

# ============================================================
# Cau 9.4.2: Ngung dao tao toi thieu CN che bien
# ============================================================
print("\n" + "="*60)
print("CAU 9.4.2: Ngung dao tao toi thieu CN che bien")
print("="*60)

i = 1  # CN che bien
net_coeff = a1[i] - c1[i]*risk[i]
retrain_ratio = c1[i]*risk[i] / d1[i]
print(f"\nNganh CN che bien:")
print(f"  a1={a1[i]}, c1*risk={c1[i]*risk[i]:.1f}, d1={d1[i]}")
print(f"  He so net AI = {net_coeff:.1f} (DUONG -> AI tao net job)")
print(f"  Ty le retrain: c1*risk/d1 = {retrain_ratio:.3f}")
print(f"  => Rang buoc Displaced <= RetrainCap: {c1[i]*risk[i]:.1f}*x_AI <= {d1[i]}*x_H")
print(f"  => x_H >= {retrain_ratio:.3f} * x_AI")
print(f"  => Moi 1 ty dau tu AI can {retrain_ratio:.3f} ty dao tao de dam bao retraining capacity")
print(f"\n  Dac biet: neu x_AI toi da (dung het ngan sach):")
x_AI_max = 30000
x_H_needed = retrain_ratio * x_AI_max
print(f"    x_AI={x_AI_max} -> x_H >= {x_H_needed:.0f} (retrain constraint)")
print(f"    nhung NetJob >= 0 -> x_H >= {-net_coeff/b1_arr[i]:.3f}*x_AI = {-net_coeff/b1_arr[i]*x_AI_max:.0f}")
print(f"    => Rang buoc retrain BINDING hon NetJob: can x_H >= {x_H_needed:.0f}")

# Ve minh hoa
x_AI_range = np.linspace(0, 30000, 100)
x_H_retrain = retrain_ratio * x_AI_range
x_H_netjob = np.maximum(0, -net_coeff / b1_arr[i] * x_AI_range)
x_H_min = np.maximum(x_H_retrain, x_H_netjob)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_AI_range, x_H_retrain, 'r--', linewidth=2, label=f'Retrain: x_H>={retrain_ratio:.3f}*x_AI')
ax.plot(x_AI_range, x_H_netjob, 'b--', linewidth=2, label=f'NetJob>=0: x_H>={-net_coeff/b1_arr[i]:.3f}*x_AI')
ax.fill_between(x_AI_range, x_H_min, 30000, alpha=0.2, color='green', label='Vung kha thi')
ax.set_xlabel('x_AI (ty VND)')
ax.set_ylabel('x_H toi thieu (ty VND)')
ax.set_title(f'Ngung dao tao toi thieu nganh {sectors[i]} (CN che bien)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30000)
ax.set_ylim(0, 30000)
plt.tight_layout()
plt.savefig('bai9_retraining_threshold.png', dpi=150)

# ============================================================
# Cau 9.4.3: Mo phong nhom de ton thuong
# ============================================================
print("\n" + "="*60)
print("CAU 9.4.3: Mo phong nhom de ton thuong (nganh 1,3,4)")
print("="*60)

vulnerable = [0, 2, 3]  # Nong-LT, Xay dung, Ban buon
print(f"\nNhom de ton thuong: {[sectors[i] for i in vulnerable]}")
print(f"\n{'Nganh':<15} {'LĐ(tr)':>8} {'Risk%':>6} {'x_AI':>8} {'x_H':>8} "
      f"{'Displace':>8} {'NetJob':>8} {'NetJob/LĐ%':>10}")
print("-" * 75)
total_disp_vuln = 0
total_l_vuln = 0
for i in vulnerable:
    pct = NetJob[i] / (L[i]*1e6) * 100 if L[i] > 0 else 0
    print(f"{sectors[i]:<15} {L[i]:>8.2f} {risk[i]*100:>5.0f}% {x_AI_opt[i]:>8.0f} "
          f"{x_H_opt[i]:>8.0f} {Displaced[i]:>8.0f} {NetJob[i]:>8.0f} {pct:>9.2f}%")
    total_disp_vuln += Displaced[i]
    total_l_vuln += L[i]*1e6

print(f"\nTong lao dong de ton thuong: {total_l_vuln:,.0f}")
print(f"Tong displaced: {total_disp_vuln:,.0f}")
print(f"Ty le displaced/LĐ: {total_disp_vuln/total_l_vuln*100:.2f}%")

# Ve Sankey luong dich chuyen lao dong (matplotlib-based)
fig, ax = plt.subplots(figsize=(12, 6))

vuln_names = [sectors[i] for i in vulnerable]
y_pos = np.arange(len(vulnerable))

# Tinh luong
kept_arr = []; retrain_arr = []; lost_arr = []
for i in vulnerable:
    displaced = Displaced[i]
    retrain = min(displaced, RetrainCap[i])
    lost = max(0, displaced - retrain)
    kept_arr.append(L[i]*1e6 - displaced)
    retrain_arr.append(retrain)
    lost_arr.append(lost)

width = 0.6
# Stack bar: kept + retrain + lost
p1 = ax.bar(y_pos, kept_arr, width, label='Giu viec', color='#2ecc71')
p2 = ax.bar(y_pos, retrain_arr, width, bottom=kept_arr, label='Displaced → Retrained', color='#f39c12')
bottoms2 = [k+r for k,r in zip(kept_arr, retrain_arr)]
p3 = ax.bar(y_pos, lost_arr, width, bottom=bottoms2, label='Displaced → Mat viec', color='#e74c3c')

ax.set_xticks(y_pos)
ax.set_xticklabels(vuln_names, fontsize=9)
ax.set_ylabel('So lao dong')
ax.set_title('Swimming lane: Luong dich chuyen lao dong nhom de ton thuong')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Them text
for i in range(len(vulnerable)):
    idx = vulnerable[i]
    if Displaced[idx] > 0:
        ax.text(i, kept_arr[i]+retrain_arr[i]/2, f'Retrain:{retrain_arr[i]:.0f}',
                ha='center', va='center', fontsize=8, fontweight='bold')
        if lost_arr[i] > 0:
            ax.text(i, bottoms2[i]+lost_arr[i]/2, f'Lost:{lost_arr[i]:.0f}',
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('bai9_sankey.png', dpi=150)
print("Da luu bieu do swimming lane: bai9_sankey.png")

# ============================================================
# Cau 9.4.4: Rang buoc Displaced <= 5% L
# ============================================================
print("\n" + "="*60)
print("CAU 9.4.4: Them rang buoc Displaced <= 5% L")
print("="*60)

A4 = np.zeros((N, 2*N))
for i in range(N):
    A4[i, i] = c1[i] * risk[i]
b4 = 0.05 * L * 1e6

A_ub4 = np.vstack([A_ub, A4])
b_ub4 = np.concatenate([b_ub, b4])

res4 = linprog(c_obj, A_ub=A_ub4, b_ub=b_ub4, bounds=bounds, method='highs')

if res4.success:
    x_AI_4 = res4.x[:N]
    x_H_4 = res4.x[N:]
    NetJob4 = coeff_AI * x_AI_4 + b1_arr * x_H_4
    Displaced4 = c1 * risk * x_AI_4
    print(f"Tong NetJob (co rang buoc 5%) = {-res4.fun:,.0f}")
    print(f"Tong NetJob (khong rang buoc) = {-res.fun:,.0f}")
    diff = -res.fun - (-res4.fun)
    print(f"Giam: {diff:,.0f} viec lam ({diff/(-res.fun)*100:.1f}%)")
    print(f"\n{'Nganh':<15} {'Displace':>10} {'5% L':>10} {'Vi pham?':>10}")
    for i in range(N):
        disp = Displaced4[i]
        limit = 0.05 * L[i] * 1e6
        print(f"{sectors[i]:<15} {disp:>10.0f} {limit:>10.0f} {'CO' if disp > limit+1 else 'KHONG':>10}")
else:
    print(f"KHONG kha thi: {res4.message}")