import pyomo.environ as pyo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# BAI 10: Quy hoach ngau nhien hai giai doan (Pyomo)
# ============================================================

J = ['I', 'D', 'AI', 'H']
S = ['s1', 's2', 's3', 's4']
p_s = {'s1': 0.30, 's2': 0.45, 's3': 0.20, 's4': 0.05}

beta_base = {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95}
beta_s = {
    ('s1','I'):1.25,('s1','D'):1.35,('s1','AI'):1.55,('s1','H'):1.05,
    ('s2','I'):1.00,('s2','D'):1.10,('s2','AI'):1.25,('s2','H'):0.95,
    ('s3','I'):0.75,('s3','D'):0.85,('s3','AI'):0.90,('s3','H'):1.00,
    ('s4','I'):0.40,('s4','D'):0.50,('s4','AI'):0.55,('s4','H'):1.10
}


def build_model(include_second_stage=True, fixed_scenario=None, fixed_x=None):
    """
    Xay dung mo hinh Pyomo.

    Parameters
    ----------
    include_second_stage : bool - co giai doan 2 khong
    fixed_scenario : str or None - neu chi dinh, chi dung 1 kich ban
    fixed_x : dict or None - neu chi dinh, co dinh first-stage
    """
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=J)
    m.S = pyo.Set(initialize=S if fixed_scenario is None else [fixed_scenario])

    m.beta = pyo.Param(m.J, initialize=beta_base)
    beta_s_filtered = {(s,j): beta_s[s,j] for s in (S if fixed_scenario is None else [fixed_scenario]) for j in J}
    m.beta_s = pyo.Param(m.S, m.J, initialize=beta_s_filtered)
    m.p = pyo.Param(m.S, initialize=p_s if fixed_scenario is None else {fixed_scenario: 1.0})

    if fixed_x is None:
        m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    else:
        m.x = pyo.Param(m.J, initialize=fixed_x)

    if include_second_stage:
        m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)

    # Rang buoc first-stage
    if fixed_x is None:
        m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65000)

    if include_second_stage:
        def budget2_rule(m, s):
            return sum(m.y[s,j] for j in m.J) <= 15000
        m.budget2 = pyo.Constraint(m.S, rule=budget2_rule)

        def ai_cap_rule(m, s):
            return m.y[s,'AI'] <= 0.5 * (m.x['H'] if fixed_x is None else fixed_x['H'])
        m.ai_cap = pyo.Constraint(m.S, rule=ai_cap_rule)

    # Ham muc tieu
    # LUU Y: De bai de cap Penalty(y_s - reserve) nhung khong dinh nghia cong thuc cu the.
    # O day Penalty duoc mo hinh hoa gian tiep qua rang buoc ngan sach y_s <= 15000.
    # De mo rong, co the them: penalty_coeff * max(0, sum(y_s) - reserve)
    def obj_rule(m):
        first = sum(m.beta[j] * m.x[j] for j in m.J)
        if include_second_stage:
            second = sum(m.p[s] * sum(m.beta_s[s,j] * m.y[s,j] for j in m.J) for s in m.S)
            return first + second
        return first
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    return m


def solve(m):
    solver = pyo.SolverFactory('appsi_highs')
    if not solver.available():
        solver = pyo.SolverFactory('glpk')
    if not solver.available():
        solver = pyo.SolverFactory('cbc')
    result = solver.solve(m, tee=False)
    return result


def extract_solution(m, has_y=True):
    x_val = {j: pyo.value(m.x[j]) for j in J}
    y_val = {}
    if has_y:
        for s in list(m.S):
            y_val[s] = {j: pyo.value(m.y[s,j]) for j in J}
    return x_val, y_val, pyo.value(m.obj)


# ============================================================
# Cau 10.5.1: Mo hinh SP (Pyomo + GLPK)
# ============================================================
print("="*60)
print("BÀI 10: Quy hoạch ngẫu nhiên 2 giai đoạn (Pyomo)")
print("="*60)
print("\nCAU 10.5.1: Mo hinh SP")
print("="*60)

m_sp = build_model()
solve(m_sp)
x_sp, y_sp, Z_SP = extract_solution(m_sp)

print(f"Z*_SP = {Z_SP:,.2f}")
print(f"\nFirst-stage:")
for j in J:
    print(f"  x_{j} = {x_sp[j]:>10.0f}")
print(f"  Tong = {sum(x_sp.values()):>10.0f}")

print(f"\nSecond-stage:")
for s in S:
    total = sum(y_sp[s].values())
    print(f"  {s} (p={p_s[s]}): y_I={y_sp[s]['I']:.0f}, y_D={y_sp[s]['D']:.0f}, "
          f"y_AI={y_sp[s]['AI']:.0f}, y_H={y_sp[s]['H']:.0f} | tong={total:.0f}")

# ============================================================
# Cau 10.5.2: Giai xac dinh tung kich ban
# ============================================================
print("\n" + "="*60)
print("CAU 10.5.2: Giai xac dinh tung kich ban rieng le")
print("="*60)

det_results = {}
for s in S:
    m_det = build_model(fixed_scenario=s)
    solve(m_det)
    x_d, y_d, Z_d = extract_solution(m_det)
    det_results[s] = {'x': x_d, 'y': y_d, 'Z': Z_d}
    print(f"\n{s}: Z* = {Z_d:,.2f}")
    print(f"  x: " + ", ".join(f"{j}={x_d[j]:.0f}" for j in J))
    print(f"  y: " + ", ".join(f"{j}={y_d[s][j]:.0f}" for j in J))

# EV solution: dung beta trung binh (expected value scenario)
beta_avg = {j: sum(p_s[s]*beta_s[s,j] for s in S) for j in J}
# Xay dung model voi beta_avg cho ca 2 stages
m_ev = pyo.ConcreteModel()
m_ev.J = pyo.Set(initialize=J)
m_ev.x = pyo.Var(m_ev.J, within=pyo.NonNegativeReals)
m_ev.budget1 = pyo.Constraint(expr=sum(m_ev.x[j] for j in m_ev.J) <= 65000)
m_ev.obj = pyo.Objective(expr=sum(beta_avg[j]*m_ev.x[j] for j in m_ev.J), sense=pyo.maximize)
solve(m_ev)
x_ev = {j: pyo.value(m_ev.x[j]) for j in J}
print(f"\nEV first-stage: {x_ev}")

# Z_EV: dung x_ev, optimize y tung scenario
Z_EV = sum(beta_base[j]*x_ev[j] for j in J)
for s in S:
    m_temp = build_model(fixed_x=x_ev, fixed_scenario=s)
    solve(m_temp)
    _, y_temp, _ = extract_solution(m_temp, has_y=True)
    Z_EV += p_s[s] * sum(beta_s[s,j]*y_temp[s][j] for j in J)

# Z_WS: perfect info -> tung scenario rieng
Z_WS = sum(p_s[s] * det_results[s]['Z'] for s in S)

# ============================================================
# Cau 10.5.3: VSS va EVPI
# ============================================================
print("\n" + "="*60)
print("CAU 10.5.3: VSS va EVPI")
print("="*60)

VSS = Z_SP - Z_EV
EVPI = Z_WS - Z_SP

print(f"Z_SP  (Stochastic)  = {Z_SP:,.2f}")
print(f"Z_EV  (EV solution) = {Z_EV:,.2f}")
print(f"Z_WS  (Wait&See)    = {Z_WS:,.2f}")
print(f"VSS = Z_SP - Z_EV = {VSS:,.2f} ({VSS/Z_SP*100:.2f}%)")
print(f"EVPI = Z_WS - Z_SP = {EVPI:,.2f} ({EVPI/Z_SP*100:.2f}%)")

print(f"\nY nghia:")
print(f"  VSS > 0: xem xet bat dinh khi quyet dinh co gia tri {VSS:,.0f}")
print(f"  EVPI: gia tri thong tin hoan hao la {EVPI:,.0f}")

# ============================================================
# Cau 10.5.4: Robust optimization (minimax regret)
# ============================================================
print("\n" + "="*60)
print("CAU 10.5.4: Robust optimization (minimax regret)")
print("="*60)

# Z*[s] = optimal cho tung scenario (da tinh o tren)
# Tim x toi uu min max_s {Z*[s] - Z[x,s]}
# Dung Pyomo: min w s.t. Z*[s] - Z[x,s] <= w for all s

m_rob = pyo.ConcreteModel()
m_rob.J = pyo.Set(initialize=J)
m_rob.S = pyo.Set(initialize=S)
m_rob.x = pyo.Var(m_rob.J, within=pyo.NonNegativeReals)
m_rob.y = pyo.Var(m_rob.S, m_rob.J, within=pyo.NonNegativeReals)
m_rob.w = pyo.Var(within=pyo.Reals)  # max regret

beta_s_filtered = {(s,j): beta_s[s,j] for s in S for j in J}
m_rob.beta_s = pyo.Param(m_rob.S, m_rob.J, initialize=beta_s_filtered)
m_rob.beta = pyo.Param(m_rob.J, initialize=beta_base)
m_rob.p = pyo.Param(m_rob.S, initialize=p_s)
m_rob.Zstar = pyo.Param(m_rob.S, initialize={s: det_results[s]['Z'] for s in S})

m_rob.budget1 = pyo.Constraint(expr=sum(m_rob.x[j] for j in m_rob.J) <= 65000)
def budget2_rule(m, s):
    return sum(m.y[s,j] for j in m.J) <= 15000
m_rob.budget2 = pyo.Constraint(m_rob.S, rule=budget2_rule)
def ai_cap_rule(m, s):
    return m.y[s,'AI'] <= 0.5 * m.x['H']
m_rob.ai_cap = pyo.Constraint(m_rob.S, rule=ai_cap_rule)

# Regret constraints: Z*[s] - (beta*x + beta_s*y_s) <= w
def regret_rule(m, s):
    z_here = sum(m.beta[j]*m.x[j] for j in m.J) + sum(m.beta_s[s,j]*m.y[s,j] for j in m.J)
    return m.Zstar[s] - z_here <= m.w
m_rob.regret = pyo.Constraint(m_rob.S, rule=regret_rule)

m_rob.obj = pyo.Objective(expr=m_rob.w, sense=pyo.minimize)
solve(m_rob)

x_rob = {j: pyo.value(m_rob.x[j]) for j in J}
w_rob = pyo.value(m_rob.w)
print(f"Robust first-stage: {x_rob}")
print(f"Minimax regret = {w_rob:.0f}")

# So sanh regret cua SP, EV, Robust
def compute_regret_dict(x_fixed):
    reg = {}
    for s in S:
        z_here = sum(beta_base[j]*x_fixed[j] for j in J)
        m_temp = build_model(fixed_x=x_fixed, fixed_scenario=s)
        solve(m_temp)
        _, y_t, _ = extract_solution(m_temp, has_y=True)
        z_here += sum(beta_s[s,j]*y_t[s][j] for j in J)
        reg[s] = det_results[s]['Z'] - z_here
    return reg

reg_sp = compute_regret_dict(x_sp)
reg_ev = compute_regret_dict(x_ev)
reg_rob = compute_regret_dict(x_rob)

print(f"\n{'Kich ban':<8} {'Z*[s]':>10} {'Reg_SP':>8} {'Reg_EV':>8} {'Reg_Rob':>8}")
for s in S:
    print(f"{s:<8} {det_results[s]['Z']:>10.0f} {reg_sp[s]:>8.0f} {reg_ev[s]:>8.0f} {reg_rob[s]:>8.0f}")
print(f"{'Max':<8} {'':>10} {max(reg_sp.values()):>8.0f} {max(reg_ev.values()):>8.0f} {max(reg_rob.values()):>8.0f}")

# ============================================================
# Ve do thi so sanh
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x_pos = np.arange(len(J))
width = 0.35
sp_vals = [x_sp[j] for j in J]
ev_vals = [x_ev[j] for j in J]
ax.bar(x_pos-width/2, sp_vals, width, label='SP', color='steelblue')
ax.bar(x_pos+width/2, ev_vals, width, label='EV', color='coral')
ax.set_xticks(x_pos); ax.set_xticklabels(J)
ax.set_ylabel('Nghin ty VND'); ax.set_title('First-stage: SP vs EV'); ax.legend(); ax.grid(axis='y', alpha=0.3)

ax = axes[1]
for s_idx, s in enumerate(S):
    y_vals = [y_sp[s][j] for j in J]
    ax.bar(x_pos+s_idx*0.2-0.3, y_vals, 0.2, label=s)
ax.set_xticks(x_pos); ax.set_xticklabels(J)
ax.set_ylabel('Nghin ty VND'); ax.set_title('Second-stage theo kich ban'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

plt.suptitle('Bài 10: Stochastic Programming (Pyomo)', fontsize=14)
plt.tight_layout(); plt.savefig('bai10_stochastic.png', dpi=150)