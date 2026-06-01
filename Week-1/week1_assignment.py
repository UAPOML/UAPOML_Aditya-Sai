"""
Portfolio Theory & Optimisation
UAPOML Summer Project - Week 1 Assignment
Part 2: Coding Questions (Q10 - Q15)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Q10: NumPy Slicing and Basic Stats
# ==========================================
print("--- Question 10 ---")
prices = np.array([
    [100, 108, 103, 115, 110, 119, 125, 121, 130, 127, 135, 140],
    [200, 195, 210, 205, 220, 215, 225, 230, 222, 235, 240, 238]
])

# (a) Simple returns using slicing
returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
print(f"Shape of returns array: {returns.shape}")

# (b) Annualised mean and standard deviation
mean_monthly = returns.mean(axis=1)
std_monthly = returns.std(axis=1, ddof=1)
mean_ann = mean_monthly * 12
std_ann = std_monthly * np.sqrt(12)
print(f"Annualised Means: {mean_ann}")
print(f"Annualised Stds: {std_ann}")

# (c) 2x2 sample covariance matrix
cov_matrix = np.cov(returns)
print("Sample Covariance Matrix:")
print(cov_matrix)
print()


# ==========================================
# Q11: N-Asset Matrix Math
# ==========================================
print("--- Question 11 ---")
mu = np.array([0.15, 0.08, 0.05])
cov = np.array([
    [0.0625, 0.0120, 0.00100],
    [0.0120, 0.0144, 0.00096],
    [0.00100, 0.00096, 0.00160]
])

# (a) Equal weight portfolio
w_eq = np.array([1/3, 1/3, 1/3])
E_Rp_eq = w_eq @ mu
var_Rp_eq = w_eq @ cov @ w_eq
print(f"Equal-Weight Expected Return: {E_Rp_eq:.4f}")
print(f"Equal-Weight Variance: {var_Rp_eq:.6f}")

# (b) Monte Carlo 10,000 Portfolios
np.random.seed(42)
weights = np.random.dirichlet(np.ones(3), size=10000)
port_rets = weights @ mu
# Vectorized variance calculation
port_vars = np.sum(weights * (weights @ cov), axis=1)
port_vols = np.sqrt(port_vars)

# (c) Max Sharpe
rf = 0.04
sharpes = (port_rets - rf) / port_vols
best_idx = np.argmax(sharpes)
max_sharpe = sharpes[best_idx]
best_weights = weights[best_idx]
print(f"Max Sharpe Ratio: {max_sharpe:.4f}")
print(f"Max Sharpe Weights: {best_weights}")
print()


# ==========================================
# Q12: Correlation Sweep
# ==========================================
print("--- Question 12 ---")
mu1, sig1 = 0.12, 0.20
mu2, sig2 = 0.06, 0.10
w1, w2 = 0.6, 0.4

# (a) Vectorized volatility sweep
rhos = np.linspace(-1, 1, 200)
vols_sweep = np.sqrt(w1**2 * sig1**2 + w2**2 * sig2**2 + 2*w1*w2*rhos*sig1*sig2)
print(f"Shape of vols_sweep: {vols_sweep.shape}")

# (b) Minimum volatility
min_idx = np.argmin(vols_sweep)
min_rho = rhos[min_idx]
min_vol = vols_sweep[min_idx]
print(f"Minimum volatility {min_vol:.4f} occurs at rho = {min_rho:.4f}")
print()


# ==========================================
# Q13: Pandas Data Manipulation
# ==========================================
print("--- Question 13 ---")
np.random.seed(0)
dates = pd.date_range('2023-01-02', periods=52, freq='W-MON')
mu_weekly = np.array([0.003, 0.002, 0.001, 0.0015])
sig_weekly = np.array([0.04, 0.03, 0.02, 0.025])
returns_sim = np.random.normal(mu_weekly, sig_weekly, (52, 4))
prices_sim = 100 * np.cumprod(1 + returns_sim, axis=0)
df = pd.DataFrame(prices_sim, index=dates, columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN'])

# (a) Compute returns
df_returns = df.pct_change().dropna()
print("First 3 rows of returns:")
print(df_returns.head(3))
print(f"Shape of returns DataFrame: {df_returns.shape}")

# (b) Describe statistics
desc = df_returns.describe()
print("\nDescriptive statistics:")
print(desc)
# From describe, we can see the asset with the highest mean and highest std
highest_mean_asset = desc.loc['mean'].idxmax()
highest_std_asset = desc.loc['std'].idxmax()
print(f"Highest Mean Return: {highest_mean_asset}")
print(f"Highest Standard Deviation: {highest_std_asset}")

# (c) Annualised Sharpe Ratios
ann_sharpes_pd = (df_returns.mean() * 52 - 0.02) / (df_returns.std() * np.sqrt(52))
print("\nAnnualised Sharpe Ratios:")
print(ann_sharpes_pd)
print()


# ==========================================
# Q14: Pandas Portfolio Tracking
# ==========================================
print("--- Question 14 ---")
# (a) Correlation matrix
corr_matrix = df_returns.corr()
print("Correlation Matrix:")
print(corr_matrix)

# To dynamically find the lowest correlation pair:
corr_unstacked = corr_matrix.unstack()
corr_unstacked = corr_unstacked[corr_unstacked < 1.0] # drop diagonals
lowest_corr_pair = corr_unstacked.idxmin()
print(f"Lowest correlation pair: {lowest_corr_pair} ({corr_unstacked.min():.4f})")

# (b) Equal weight portfolio tracking
w_pandas = pd.Series([0.25, 0.25, 0.25, 0.25], index=df_returns.columns)
port_ret_series = df_returns.dot(w_pandas)

# (c) Resample to monthly
# Note: Using 'ME' (Month End) instead of deprecated 'M' for resampling rule
monthly_port = port_ret_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
mean_monthly_port = monthly_port.mean()
std_monthly_port = monthly_port.std()
print(f"\nMonthly Resampled Portfolio Mean Return: {mean_monthly_port:.6f}")
print(f"Monthly Resampled Portfolio Std Dev: {std_monthly_port:.6f}")
print()


# ==========================================
# Q15: Matplotlib
# ==========================================
print("--- Question 15 ---")
print("Generating 'week1_plots.png'...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Portfolio Theory --- Week 1 Visualisations', fontsize=16, fontweight='bold')

# --- Subplot 1: Efficient Frontier ---
sc = ax1.scatter(port_vols, port_rets, c=sharpes, cmap='viridis', alpha=0.5, s=10)
# Max Sharpe Star
ax1.scatter(port_vols[best_idx], port_rets[best_idx], color='gold', marker='*', s=250, edgecolor='black', label='Max Sharpe')
# Individual Assets
ax1.scatter(np.sqrt(np.diag(cov)), mu, color='#1A1A1A', s=100, zorder=5)
tickers = ['Asset 1', 'Asset 2', 'Asset 3']
for i, txt in enumerate(tickers):
    ax1.annotate(txt, (np.sqrt(cov[i,i]) + 0.005, mu[i]), fontsize=10, fontweight='bold')

cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label('Sharpe Ratio')
ax1.set_xlabel('Portfolio Volatility ($\sigma_p$)')
ax1.set_ylabel('Expected Return ($\mathbb{E}[R_p]$)')
ax1.set_title('Efficient Frontier (3 Assets)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# --- Subplot 2: Correlation Sensitivity ---
ax2.plot(rhos, vols_sweep, color='teal', lw=2.5)
weighted_avg_risk = w1 * sig1 + w2 * sig2
ax2.axhline(weighted_avg_risk, color='red', linestyle='--', label='Weighted Avg. Risk')
ax2.fill_between(rhos, vols_sweep, weighted_avg_risk, where=(vols_sweep <= weighted_avg_risk), color='lightgreen', alpha=0.4, label='Diversification Benefit')

ax2.set_xlabel('Correlation ($\\rho$)')
ax2.set_ylabel('Portfolio Volatility ($\sigma_p$)')
ax2.set_title('Risk Sensitivity to Correlation')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
plt.savefig('week1_plots.png', dpi=150)
print("Plots successfully saved as 'week1_plots.png'!")
