import numpy as np
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# BAI 8: Toi uu dong phan bo lien thoi gian 2026-2035
# ============================================================

alpha, beta_cd, gamma_d, delta_ai, theta_h = 0.33, 0.42, 0.10, 0.08, 0.07
delta_K, delta_D, delta_AI = 0.05, 0.12, 0.15
theta_H_eff, mu = 0.8, 0.02
phi1, phi2, phi3 = 0.003, 0.002, 0.004
rho_disc = 0.97
gamma_cr = 1.5
T = 10
years = np.arange(2026, 2026+T+1)

K0, L0, D0, AI0, H0 = 27500.0, 53.9, 20.3, 86.0, 30.0
Y0 = 12847.6
A0 = Y0 / (K0**alpha * L0**beta_cd * D0**gamma_d * AI0**delta_ai * H0**theta_h)
L = np.array([L0 * 1.009**t for t in range(T+1)])

def compute_trajectory(u, shock_year=None, shock_pct=0.0):
    IK = u[0::4]; ID = u[1::4]; IAI = u[2::4]; IH = u[3::4]
    K = np.zeros(T+1); D = np.zeros(T+1); AI = np.zeros(T+1)
    H = np.zeros(T+1); A = np.zeros(T+1); Y = np.zeros(T+1); C = np.zeros(T)
    K[0]=K0; D[0]=D0; AI[0]=AI0; H[0]=H0; A[0]=A0

    for t in range(T):
        # Shock A_t (TFP) tai nam soc → anh huong keo dai qua dong hoc A[t+1]=A[t]*(1+growth)
        if shock_year is not None and t == shock_year:
            A[t] *= (1 - shock_pct)
        Y[t] = A[t]*K[t]**alpha*L[t]**beta_cd*D[t]**gamma_d*AI[t]**delta_ai*H[t]**theta_h
        C[t] = Y[t] - IK[t] - ID[t] - IAI[t] - IH[t]
        if C[t] <= 0: return None
        K[t+1] = (1-delta_K)*K[t] + IK[t]
        D[t+1] = (1-delta_D)*D[t] + ID[t]
        AI[t+1] = (1-delta_AI)*AI[t] + IAI[t]
        H[t+1] = H[t] + theta_H_eff*IH[t] - mu*H[t]
        A[t+1] = A[t]*(1 + phi1*(D[t]/100) + phi2*(AI[t]/100) + phi3*(H[t]/100))
    Y[T] = A[T]*K[T]**alpha*L[T]**beta_cd*D[T]**gamma_d*AI[T]**delta_ai*H[T]**theta_h
    return K, D, AI, H, Y, C, A

def welfare(u, shock_year=None, shock_pct=0.0):
    result = compute_trajectory(u, shock_year, shock_pct)
    if result is None: return 1e15
    K, D, AI, H, Y, C, A = result
    if np.any(C <= 0): return 1e15
    W = sum(rho_disc**t * (C[t]**(1-gamma_cr)-1)/(1-gamma_cr) for t in range(T))
    return -W

# Khoi tao
Y_est = 14000; invest_frac = 0.15; total_invest = Y_est * invest_frac
u0 = np.zeros(T*4)
for t in range(T):
    u0[t*4+0] = total_invest*0.40
    u0[t*4+1] = total_invest*0.25
    u0[t*4+2] = total_invest*0.20
    u0[t*4+3] = total_invest*0.15
bounds = [(0, None)]*(T*4)

def constraint_budget(u, **kwargs):
    result = compute_trajectory(u, **kwargs)
    if result is None: return -1e10
    return min(result[5]) - 1

constraints = [{'type':'ineq','fun':constraint_budget}]

# ============================================================
# Cau 8.3.1 + 8.3.2: Toi uu co so
# ============================================================
print("="*60)
print("BÀI 8: Tối ưu động 2026-2035")
print("="*60)

res = minimize(welfare, u0, method='SLSQP', bounds=bounds, constraints=constraints,
               options={'maxiter':1000, 'ftol':1e-8, 'disp':False})
K_opt, D_opt, AI_opt, H_opt, Y_opt, C_opt, A_opt = compute_trajectory(res.x)

print(f"\nPhuc loi W* = {-res.fun:.2f}")
print(f"\n{'Nam':<6} {'K':>10} {'D':>8} {'AI':>8} {'H':>8} {'TFP':>10} {'Y':>10} {'C':>10}")
print("-"*72)
for t in range(T+1):
    c_val = C_opt[t] if t < T else float('nan')
    print(f"{2026+t:<6} {K_opt[t]:>10.0f} {D_opt[t]:>8.1f} {AI_opt[t]:>8.1f} {H_opt[t]:>8.1f} "
          f"{A_opt[t]:>10.2f} {Y_opt[t]:>10.0f} {c_val:>10.0f}")

print(f"\nTy le dau tu / GDP:")
print(f"{'Nam':<6} {'IK/Y':>8} {'ID/Y':>8} {'IAI/Y':>8} {'IH/Y':>8} {'Tong':>8}")
for t in range(T):
    u_t = res.x[t*4:(t+1)*4]
    print(f"{2026+t:<6} {u_t[0]/Y_opt[t]*100:>8.1f} {u_t[1]/Y_opt[t]*100:>8.1f} "
          f"{u_t[2]/Y_opt[t]*100:>8.1f} {u_t[3]/Y_opt[t]*100:>8.1f} {u_t.sum()/Y_opt[t]*100:>8.1f}")

# ============================================================
# Cau 8.3.3: Soc nam 2028 (TFP giam 8% - persistent qua dong hoc A_t)
# ============================================================
print("\n" + "="*60)
print("CAU 8.3.3: Soc nam 2028 - Y giam 8%")
print("="*60)

SHOCK_T = 2  # nam 2028 = t index 2
SHOCK_PCT = 0.08

# Kich ban A: ke hoach goc, khong soc
K_n, D_n, AI_n, H_n, Y_n, C_n, A_n = compute_trajectory(res.x)
W_base = -welfare(res.x)

# Kich ban B: ke hoach goc, CO soc (khong doi lai ke hoach)
_, _, _, _, Y_sh, C_sh, _ = compute_trajectory(res.x, SHOCK_T, SHOCK_PCT)
W_plan_shock = -welfare(res.x, SHOCK_T, SHOCK_PCT)

# Kich ban C: toi uu lai sau soc
res_shock = minimize(lambda u: welfare(u, shock_year=SHOCK_T, shock_pct=SHOCK_PCT),
                     res.x, method='SLSQP', bounds=bounds, constraints=constraints,
                     options={'maxiter':1000, 'ftol':1e-8, 'disp':False})
K_s, D_s, AI_s, H_s, Y_s, C_s, A_s = compute_trajectory(res_shock.x, SHOCK_T, SHOCK_PCT)
W_reopt = -res_shock.fun

print(f"\n3 kich ban so sanh (soc TFP persistent):")
print(f"  (A) Khong soc:          W = {W_base:.6f}")
print(f"  (B) Co soc, giu ke hoach: W = {W_plan_shock:.6f}")
print(f"  (C) Co soc, tai toi uu:   W = {W_reopt:.6f}")
print(f"  Mat welfare do soc:       {W_base - W_plan_shock:.6f} ({(W_base - W_plan_shock)/W_base*100:.4f}%)")
print(f"  Phuc hoi khi tai toi uu:  {W_reopt - W_plan_shock:.6f}")

# So sanh dau tu
print(f"\nDau tu nam 2028 (t=2) va nam 2029 (t=3):")
u_orig_2 = res.x[2*4:3*4]; u_orig_3 = res.x[3*4:4*4]
u_sh_2 = res_shock.x[2*4:3*4]; u_sh_3 = res_shock.x[3*4:4*4]
print(f"  {'':>8} {'I_K':>8} {'I_D':>8} {'I_AI':>8} {'I_H':>8} {'Tong':>8}")
print(f"  {'Goc t2':<8} {u_orig_2[0]:>8.0f} {u_orig_2[1]:>8.0f} {u_orig_2[2]:>8.0f} {u_orig_2[3]:>8.0f} {u_orig_2.sum():>8.0f}")
print(f"  {'Soc t2':<8} {u_sh_2[0]:>8.0f} {u_sh_2[1]:>8.0f} {u_sh_2[2]:>8.0f} {u_sh_2[3]:>8.0f} {u_sh_2.sum():>8.0f}")
print(f"  {'Goc t3':<8} {u_orig_3[0]:>8.0f} {u_orig_3[1]:>8.0f} {u_orig_3[2]:>8.0f} {u_orig_3[3]:>8.0f} {u_orig_3.sum():>8.0f}")
print(f"  {'Soc t3':<8} {u_sh_3[0]:>8.0f} {u_sh_3[1]:>8.0f} {u_sh_3[2]:>8.0f} {u_sh_3[3]:>8.0f} {u_sh_3.sum():>8.0f}")

print(f"\nQuy dao Y:")
print(f"  {'Nam':<6} {'Y_khong_soc':>12} {'Y_co_soc':>12} {'Y_sai%':>8} {'C_khong':>10} {'C_co':>10}")
for t in range(T):
    ys = Y_sh[t]; yn = Y_n[t]
    y_diff = (ys-yn)/yn*100 if yn>0 else 0
    print(f"  {2026+t:<6} {yn:>12.0f} {ys:>12.0f} {y_diff:>+7.1f}% {C_n[t]:>10.0f} {C_sh[t]:>10.0f}")

print(f"\nPhan tich: Soc TFP persistent giam A_t tu 2028 tro di.")
print(f"  Anh huong keo dai qua dong hoc A[t+1]=A[t]*(1+growth).")
print(f"  Welfare % thay doi nho KHONG phai do risk-neutrality (CRRA gamma=1.5 VAN risk-averse).")
print(f"  Ly do: (1) consumption smoothing - giam dau tu de C khong sut manh,")
print(f"  (2) chiet khau rho=0.97 lam giam trong so cua welfare bi anh huong tu 2028 tro di,")
print(f"  (3) marginal utility U'(C)=C^(-gamma) tai C~30000 rat nho (~1.9e-7).")

# ============================================================
# Cau 8.3.4: So sanh 2 chien luoc
# ============================================================
print("\n" + "="*60)
print("CAU 8.3.4: So sanh chien luoc")
print("="*60)

u_even = np.zeros(T*4)
for t in range(T):
    u_even[t*4+0] = total_invest*0.40
    u_even[t*4+1] = total_invest*0.25
    u_even[t*4+2] = total_invest*0.20
    u_even[t*4+3] = total_invest*0.15
W_even = -welfare(u_even)

u_front = np.zeros(T*4)
for t in range(T):
    f = 1.5 if t < 3 else 0.7
    u_front[t*4+0] = total_invest*0.40*f
    u_front[t*4+1] = total_invest*0.25*f
    u_front[t*4+2] = total_invest*0.20*f
    u_front[t*4+3] = total_invest*0.15*f
W_front = -welfare(u_front)
W_opt = -res.fun

_, _, _, _, Y_e, _, _ = compute_trajectory(u_even)
_, _, _, _, Y_f, _, _ = compute_trajectory(u_front)

print(f"{'Chien luoc':<20} {'Phuc loi':>10} {'GDP 2035':>12}")
print("-"*45)
print(f"{'Toi uu (SLSQP)':<20} {W_opt:>10.2f} {Y_opt[T]:>12,.0f}")
print(f"{'Dau tu deu':<20} {W_even:>10.2f} {Y_e[T]:>12,.0f}")
print(f"{'Front-load':<20} {W_front:>10.2f} {Y_f[T]:>12,.0f}")

# Ve do thi
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, data, title, ylabel in [
    (axes[0,0], K_opt, 'K (von vat chat)', 'nghin ty'),
    (axes[0,1], D_opt, 'D (ha tang so)', '% GDP'),
    (axes[0,2], AI_opt, 'AI (nghin DN)', ''),
    (axes[1,0], H_opt, 'H (nhan luc)', '%'),
    (axes[1,2], A_opt, 'A (TFP)', ''),
]:
    ax.plot(years, data, 'b-o', markersize=4)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.plot(years, Y_opt, 'k-o', markersize=4, label='Y (GDP)')
ax.plot(years[:T], C_opt, 'c-o', markersize=4, label='C (tieu dung)')
ax.set_title('Y va C'); ax.set_ylabel('nghin ty'); ax.legend(); ax.grid(True, alpha=0.3)
plt.suptitle('Quy dao toi uu 2026-2035', fontsize=14)
plt.tight_layout(); plt.savefig('bai8_trajectory.png', dpi=150)