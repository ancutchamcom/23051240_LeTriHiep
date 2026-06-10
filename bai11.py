import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# BAI 11: Q-learning cho chinh sach kinh te thich nghi
# ============================================================

class VietnamEconomyEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        # State: [gdp_level, d_level, ai_level, h_level]
        # LUU Y: Khong dung U (that nghiep) trong state vi action anh huong H truc tiep,
        # nhung state phai co H de agent hoc quan he action -> H -> Y -> reward.
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.T = 10

        self.allocation = {
            0: np.array([0.70, 0.10, 0.10, 0.10]),  # Truyen thong
            1: np.array([0.40, 0.25, 0.15, 0.20]),  # Can bang
            2: np.array([0.25, 0.45, 0.15, 0.15]),  # So hoa nhanh
            3: np.array([0.20, 0.20, 0.45, 0.15]),  # AI dan dat
            4: np.array([0.30, 0.20, 0.10, 0.40]),  # Bao trum
        }
        self.action_names = ['Truyen thong', 'Can bang', 'So hoa nhanh', 'AI dan dat', 'Bao trum']
        self.w = np.array([0.40, 0.25, 0.20, 0.15])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'state' in options:
            self.state = np.array(options['state'])
        else:
            self.state = self.np_random.integers(0, 3, size=4)
        self.t = 0
        self.K = 27500.0
        self.D = 20.3
        self.AI = 86.0
        self.H = 30.0
        self.Y_prev = 12847.6
        return self.state.copy(), {}

    def step(self, action):
        a = self.allocation[action]
        budget = 2100.0  # ~15% GDP nam 2026

        dK = a[0] * budget
        dD = a[1] * budget * 0.01
        dAI = a[2] * budget * 0.05
        dH = a[3] * budget * 0.01

        self.K = (1 - 0.05)*self.K + dK
        self.D = (1 - 0.12)*self.D + dD
        self.AI = (1 - 0.15)*self.AI + dAI
        self.H = self.H + 0.8*dH - 0.02*self.H

        A = 33.70 * (1 + 0.003*(self.D/100) + 0.002*(self.AI/100) + 0.004*(self.H/100))**self.t
        L = 53.9 * 1.009**self.t
        Y = A * self.K**0.33 * L**0.42 * self.D**0.10 * self.AI**0.08 * self.H**0.07

        delta_gdp = (Y - self.Y_prev) / self.Y_prev
        delta_unemploy = max(0, -delta_gdp * 0.5)  # proxy: GDP giam -> that nghiep tang
        cyber_risk = (self.AI / (self.H + 1)) * 0.01  # AI cao, H thap -> rui ro an ninh
        emission = (self.K + self.AI) * 0.0001  # phat thai tu K va AI

        # Reward = w1*dGDP - w2*dUnemploy - w3*CyberRisk - w4*Emission
        # LUU Y: delta_unemploy, cyber_risk, emission la proxy don gian,
        # khong phai mo hinh kinh te chinh thuc.
        reward = (self.w[0] * delta_gdp * 100
                  - self.w[1] * delta_unemploy * 100
                  - self.w[2] * cyber_risk
                  - self.w[3] * emission)

        self.Y_prev = Y
        self.t += 1

        # State discretization thresholds:
        # GDP growth: <3%=low, 3-6%=med, >6%=high
        # D (% GDP): <25=low, 25-35=med, >35=high
        # AI (nghin DN): <100=low, 100-200=med, >200=high
        # H (% LĐ): <35=low, 35-50=med, >50=high
        gdp_level = 0 if delta_gdp < 0.03 else (1 if delta_gdp < 0.06 else 2)
        d_level = 0 if self.D < 25 else (1 if self.D < 35 else 2)
        ai_level = 0 if self.AI < 100 else (1 if self.AI < 200 else 2)
        h_level = 0 if self.H < 35 else (1 if self.H < 50 else 2)
        self.state = np.array([gdp_level, d_level, ai_level, h_level])

        done = self.t >= self.T
        return self.state.copy(), reward, done, False, {}

# ============================================================
# Q-learning Training
# ============================================================
print("="*60)
print("BÀI 11: Q-learning cho chính sách kinh tế")
print("="*60)

Q = np.zeros((3, 3, 3, 3, 5))
n_episodes = 10000
gamma = 0.95
alpha = 0.1
reward_history = []

env = VietnamEconomyEnv()

for ep in range(n_episodes):
    s, _ = env.reset()
    total_reward = 0
    eps = max(0.05, 1.0 - ep / 5000)

    while True:
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            a = int(np.argmax(Q[tuple(s)]))

        s2, r, done, _, _ = env.step(a)
        best_next = np.max(Q[tuple(s2)])
        Q[tuple(s) + (a,)] += alpha * (r + gamma * best_next * (1 - done) - Q[tuple(s) + (a,)])
        total_reward += r
        s = s2
        if done:
            break

    reward_history.append(total_reward)

# ============================================================
# Cau 11.3.3: Chinh sach pi*
# ============================================================
print("\n" + "="*60)
print("CAU 11.3.3: Chính sách π*(s) = argmax Q(s,a)")
print("="*60)

print("\nChinh sach toi uu cho 5 trang thai khoi dau:")
test_states = [
    ([1, 1, 0, 1], "VN 2026 thuc te (GDP_med, D_med, AI_low, H_med)"),
    ([0, 0, 0, 2], "Kich ban te (GDP_low, D_low, AI_low, H_high)"),
    ([2, 2, 2, 2], "Kich ban tot (GDP_high, D_high, AI_high, H_high)"),
    ([0, 1, 0, 0], "Sau khung hoang (GDP_low, D_med, AI_low, H_low)"),
    ([1, 0, 2, 1], "AI manh, D yeu (GDP_med, D_low, AI_high, H_med)"),
]

for state, desc in test_states:
    a = int(np.argmax(Q[tuple(state)]))
    q_vals = Q[tuple(state)]
    print(f"  {desc}")
    print(f"    -> π* = {env.action_names[a]} (action {a})")
    print(f"    Q-values: {', '.join(f'{env.action_names[i]}={q_vals[i]:.3f}' for i in range(5))}")

# ============================================================
# Cau 11.3.4: So sanh voi chinh sach rule-based
# ============================================================
print("\n" + "="*60)
print("CAU 11.3.4: So sanh voi rule-based")
print("="*60)

def evaluate_policy(policy_fn, n_eval=500):
    rewards = []
    for _ in range(n_eval):
        s, _ = env.reset()
        total = 0
        while True:
            a = policy_fn(s)
            s, r, done, _, _ = env.step(a)
            total += r
            if done:
                break
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)

def policy_optimal(s): return int(np.argmax(Q[tuple(s)]))
def policy_always1(s): return 1  # Can bang
def policy_always3(s): return 3  # AI dan dat
def policy_random(s): return np.random.randint(5)

policies = [
    ('π* (Q-learning)', policy_optimal),
    ('Luon Can bang (a1)', policy_always1),
    ('Luon AI dan dat (a3)', policy_always3),
    ('Random', policy_random),
]

print(f"\n{'Chinh sach':<25} {'Mean reward':>12} {'Std':>8}")
print("-" * 48)
results = {}
for name, fn in policies:
    mean_r, std_r = evaluate_policy(fn)
    results[name] = (mean_r, std_r)
    print(f"{name:<25} {mean_r:>12.2f} {std_r:>8.2f}")

# Learning curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
window = 200
smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
ax.plot(smoothed, 'b-', alpha=0.8)
ax.set_xlabel('Episode')
ax.set_ylabel('Tong phuc loi')
ax.set_title('Learning curve (Q-learning)')
ax.grid(True, alpha=0.3)

ax = axes[1]
names = [p[0] for p in policies]
means = [results[n][0] for n in names]
stds = [results[n][1] for n in names]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']
bars = ax.bar(range(len(names)), means, yerr=stds, color=colors, capsize=5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=15, ha='right')
ax.set_ylabel('Tong phuc loi binh quan')
ax.set_title('So sanh chinh sach')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('bai11_qlearning.png', dpi=150)

print(f"\nπ* hon Random: {results['π* (Q-learning)'][0] - results['Random'][0]:.2f}")
print(f"π* hon Luon Can bang: {results['π* (Q-learning)'][0] - results['Luon Can bang (a1)'][0]:.2f}")