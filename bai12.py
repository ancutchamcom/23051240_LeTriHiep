
"""AIDEOM-VN: Nguyen mau tich hop 6 module."""
import numpy as np
import pandas as pd
import pulp
from scipy.optimize import linprog, minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as moo_minimize
from pymoo.termination import get_termination
import matplotlib
import matplotlib.pyplot as plt
from io import StringIO

print("="*60)
print("BÀI 12: AIDEOM-VN - Nguyên mẫu tích hợp 6 module")
print("="*60)

# ============================================================
# M1: Du bao kinh te (Cobb-Douglas) 2026-2030
# ============================================================
print("\n" + "="*60)
print("M1: Du bao kinh te Cobb-Douglas 2026-2030")
print("="*60)

a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
K0, L0, D0_v, AI0, H0, A0 = 27500, 53.9, 20.3, 86, 30, 33.70
T = 4; years = list(range(2026, 2026+T+1)); budget_annual = 3000

def forecast(K0, D0, AI0, H0, A0, L0, alloc, T=T):
    K,D,AI,H,A = K0,D0,AI0,H0,A0
    traj = [A*K**a*L0**b*D**g*AI**d*H**th]
    for t in range(T):
        K=(1-0.05)*K+alloc['K']*budget_annual
        D=(1-0.12)*D+alloc['D']*budget_annual*0.01
        AI=(1-0.15)*AI+alloc['AI']*budget_annual*0.05
        H=H+0.8*alloc['H']*budget_annual*0.01-0.02*H
        A=A*(1+0.003*(D/100)+0.002*(AI/100)+0.004*(H/100))
        L=L0*1.009**(t+1)
        traj.append(A*K**a*L**b*D**g*AI**d*H**th)
    return traj

scenarios = {
    'S1_Truyen thong': {'K':0.70,'D':0.10,'AI':0.10,'H':0.10},
    'S2_So hoa nhanh': {'K':0.25,'D':0.45,'AI':0.15,'H':0.15},
    'S3_AI dan dat':   {'K':0.20,'D':0.20,'AI':0.45,'H':0.15},
    'S4_Bao trum':     {'K':0.30,'D':0.20,'AI':0.10,'H':0.40},
    'S5_Toi uu':       {'K':0.25,'D':0.25,'AI':0.30,'H':0.20},
}
gdp_fc = {n: forecast(K0,D0_v,AI0,H0,A0,L0,al) for n,al in scenarios.items()}

header = f"{'Kich ban':<22}" + "".join(f"{y:>8}" for y in years)
print(header); print("-"*62)
for name, traj in gdp_fc.items():
    print(f"{name:<22}" + "".join(f"{v:>8.0f}" for v in traj))

# ============================================================
# M2: TOPSIS + Entropy
# ============================================================
print("\n" + "="*60)
print("M2: TOPSIS + Entropy weight")
print("="*60)

RAW_REGIONS = """region_id,region_name_en,region_name_short,grdp_per_capita_million_VND,fdi_registered_billion_USD,digital_index_0_100,ai_readiness_0_100,trained_labor_pct,rd_intensity_pct,internet_penetration_pct,gini_coef
1,"N. Midlands\n& Mountains",NMM,57.0,3.5,38,22,21.5,0.18,72,0.405
2,"Red River\nDelta",RRD,152.3,20.0,78,68,36.8,0.85,92,0.358
3,"N-S Central\nCoast",NSCC,87.5,8.2,55,40,27.5,0.32,84,0.372
4,"Central\nHighlands",CH,68.9,0.8,32,18,18.2,0.15,68,0.412
5,Southeast,SE,158.9,18.5,82,75,42.5,0.78,94,0.385
6,"Mekong\nDelta",MD,80.5,2.1,48,30,16.8,0.22,78,0.392"""
df_reg = pd.read_csv(StringIO(RAW_REGIONS))

criteria = ['grdp_per_capita_million_VND','fdi_registered_billion_USD',
            'digital_index_0_100','ai_readiness_0_100',
            'trained_labor_pct','rd_intensity_pct',
            'internet_penetration_pct','gini_coef']
is_ben = [True,True,True,True,True,True,True,False]
w_expert = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])

def topsis(X, w, is_ben):
    R = X/np.sqrt((X**2).sum(0)); V = R*w
    A_s = np.where(is_ben,V.max(0),V.min(0))
    A_n = np.where(is_ben,V.min(0),V.max(0))
    S_s = np.sqrt(((V-A_s)**2).sum(1))
    S_n = np.sqrt(((V-A_n)**2).sum(1))
    return S_n/(S_s+S_n)

def entropy_weights(X):
    P = X/X.sum(0); k = 1.0/np.log(len(X))
    E = -k*np.nansum(P*np.log(P+1e-12),0); d = 1-E
    return d/d.sum()

X = df_reg[criteria].values.astype(float)

# Expert weights
C_exp = topsis(X, w_expert, is_ben)
# Entropy weights
w_ent = entropy_weights(X)
C_ent = topsis(X, w_ent, is_ben)

print(f"\nTrong so Entropy: {dict(zip([c[:8] for c in criteria], np.round(w_ent,4)))}")
print(f"\n{'Vung':<30} {'C*_expert':>10} {'C*_entropy':>10} {'Rank_E':>6} {'Rank_En':>6}")
for i, rn in enumerate(df_reg['region_name_en']):
    re = np.argsort(-C_exp)[i] if False else None
    print(f"{rn:<30} {C_exp[i]:>10.4f} {C_ent[i]:>10.4f}")

rank_exp = np.argsort(-C_exp) + 1
rank_ent = np.argsort(-C_ent) + 1
for i, rn in enumerate(df_reg['region_name_en']):
    print(f"{rn:<30} {C_exp[i]:>10.4f} {C_ent[i]:>10.4f} {list(np.argsort(-C_exp)).index(i)+1:>6} {list(np.argsort(-C_ent)).index(i)+1:>6}")

# ============================================================
# M3: Phan bo ngan sach LP + Dynamic (Bai 8)
# ============================================================
print("\n" + "="*60)
print("M3: Phan bo LP (C1-C5) + Dynamic optimization")
print("="*60)

regions = ['NMM','RRD','NCC','CH','SE','MD']
items = ['I','D','AI','H']
beta = {('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
        ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
        ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
        ('CH','I'):1.20,('CH','D'):0.75,('CH','AI'):0.45,('CH','H'):1.35,
        ('SE','I'):0.90,('SE','D'):1.30,('SE','AI'):1.55,('SE','H'):1.00,
        ('MD','I'):1.10,('MD','D'):0.85,('MD','AI'):0.65,('MD','H'):1.25}

D0_dict = dict(zip(
    df_reg['region_name_en'].map({'N. Midlands\n& Mountains':'NMM',
        'Red River\nDelta':'RRD','N-S Central\nCoast':'NCC',
        'Central\nHighlands':'CH','Southeast':'SE','Mekong\nDelta':'MD'}),
    df_reg['digital_index_0_100']))

gamma_val, lam_val = 0.002, 0.6
m = pulp.LpProblem('M3',pulp.LpMaximize)
x = pulp.LpVariable.dicts('x',(regions,items),lowBound=0)
m += pulp.lpSum(beta[(r,j)]*x[r][j] for r in regions for j in items)
m += pulp.lpSum(x[r][j] for r in regions for j in items) <= 50000
for r in regions:
    m += pulp.lpSum(x[r][j] for j in items) >= 5000
    m += pulp.lpSum(x[r][j] for j in items) <= 12000
m += pulp.lpSum(x[r]['H'] for r in regions) >= 12000
Mvar = pulp.LpVariable('Dmax')
for r in regions:
    m += D0_dict[r]+gamma_val*x[r]['D'] <= Mvar
    m += D0_dict[r]+gamma_val*x[r]['D'] >= lam_val*Mvar
m.solve(pulp.PULP_CBC_CMD(msg=False))

alloc_mat = np.zeros((6,4))
for i,r in enumerate(regions):
    for j_idx,j in enumerate(items):
        alloc_mat[i,j_idx] = x[r][j].value()
Z_lp = pulp.value(m.objective)
print(f"LP Z* = {Z_lp:.0f} ty VND (co C5)")

# Dynamic: tinh TFP growth tu allocation (dung gia tri chi so, khong phai tien dau tu)
# D, AI, H la chi so, dau tu tang chung theo ty le rieng
# D_total (ty VND) -> chi so D tang = D_total * 0.01 / 50000 * 50  (scale)
# Don gian hon: tinh % ngan sach cho tung loai va uoc tinh TFP boost
total_budget = alloc_mat.sum()
pct_D = alloc_mat[:,1].sum() / total_budget  # ty le D
pct_AI = alloc_mat[:,2].sum() / total_budget  # ty le AI
pct_H = alloc_mat[:,3].sum() / total_budget   # ty le H
# TFP boost = phi * delta_index, voi delta_index ~ pct * budget * scale
# Dung scale tu Bai 8: D ~ 20% GDP, tang ~1% GDP/nam voi 25% ngan sach cho D
dD = pct_D * 50000 * 0.01   # ~ delta D (% GDP)
dAI = pct_AI * 50000 * 0.05  # ~ delta AI (nghin DN)
dH = pct_H * 50000 * 0.01    # ~ delta H (%)
tfp_boost = 0.003*(dD/100) + 0.002*(dAI/100) + 0.004*(dH/100)
print(f"Dynamic TFP boost uoc tinh: {tfp_boost*100:.3f}%/nam")
print(f"  dD={dD:.1f}% GDP, dAI={dAI:.0f} nghin DN, dH={dH:.1f}%")

# ============================================================
# M4: Thi truong lao dong NetJob
# ============================================================
print("\n" + "="*60)
print("M4: Mo phong thi truong lao dong")
print("="*60)

a1 = np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
b1 = np.array([45,28,35,32,22,30,20,55])
c1 = np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
d1 = np.array([50,32,42,38,26,36,24,62])
risk = np.array([18,42,25,38,52,35,28,22])/100
sec = ['NongLT','CNchebien','Xaydung','Banbuon','Taichinh','Logistics','CNTT','Giaoduc']
N = 8; coeff = a1-c1*risk

c_obj = np.concatenate([-coeff,-b1])
A1_l = np.concatenate([np.ones(N),np.ones(N)]).reshape(1,-1)
A2 = np.zeros((N,2*N))
for i in range(N): A2[i,i]=-coeff[i]; A2[i,N+i]=-b1[i]
A3 = np.zeros((N,2*N))
for i in range(N): A3[i,i]=c1[i]*risk[i]; A3[i,N+i]=-d1[i]
res = linprog(c_obj,A_ub=np.vstack([A1_l,A2,A3]),
              b_ub=np.concatenate([[30000],np.zeros(N),np.zeros(N)]),
              bounds=[(0,None)]*(2*N),method='highs')
xA=res.x[:N]; xH=res.x[N:]
NJ=coeff*xA+b1*xH
print(f"Tong NetJob = {-res.fun:,.0f}")
