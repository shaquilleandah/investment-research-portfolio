"""
Dynamic Stock Portfolio Tracker
================================
Author: Shaquille-Benjamin Andah
College of Wooster | Business Economics '27
Market Risk & Financial Analysis | Investment Research

A Python-based portfolio tracker that pulls live market data,
calculates performance metrics, and benchmarks returns against
major indices. Built to demonstrate quantitative finance skills
in equity analysis, risk measurement, and data visualization.

Portfolio: A diversified 10-stock portfolio across Financials,
Technology, Healthcare, Energy, and Consumer sectors.

Requirements:
    pip install yfinance pandas numpy matplotlib

Usage:
    python portfolio_tracker.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── 1. PORTFOLIO HOLDINGS ─────────────────────────────────────
"""
A diversified portfolio of 10 large and mid-cap U.S. equities
spanning five sectors. Purchase prices reflect January 2025
entry points. Portfolio is equally weighted at inception.
"""

holdings = {
    # Ticker : (shares, purchase_price, sector)
    "JPM":   (50,   193.50,  "Financials"),
    "GS":    (25,   502.00,  "Financials"),
    "AAPL":  (60,   226.00,  "Technology"),
    "MSFT":  (35,   415.00,  "Technology"),
    "V":     (45,   310.00,  "Financials"),
    "UNH":   (20,   510.00,  "Healthcare"),
    "XOM":   (75,   110.00,  "Energy"),
    "AMZN":  (40,   218.00,  "Technology"),
    "BRK-B": (55,   460.00,  "Financials"),
    "JNJ":   (60,   155.00,  "Healthcare"),
}

CASH        = 50_000.00
BENCHMARK   = "SPY"       # S&P 500 ETF
BENCHMARK2  = "QQQ"       # Nasdaq 100 ETF
START_DATE  = "2025-01-01"

# ── 2. FETCH LIVE DATA ────────────────────────────────────────

print("=" * 65)
print("  DYNAMIC STOCK PORTFOLIO TRACKER")
print("  Shaquille-Benjamin Andah | College of Wooster '27")
print(f"  Live as of: {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
print("=" * 65)

tickers = list(holdings.keys()) + [BENCHMARK, BENCHMARK2]
print(f"\nFetching live data for {len(holdings)} holdings + benchmarks...")

raw    = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)
prices = raw["Close"]

print("Data fetched successfully.\n")

# ── 3. PORTFOLIO METRICS ──────────────────────────────────────

rows = []
total_cost    = 0
total_mkt_val = 0

for ticker, (shares, purchase_price, sector) in holdings.items():
    try:
        current_price = float(prices[ticker].dropna().iloc[-1])
    except:
        current_price = purchase_price

    cost_basis   = shares * purchase_price
    market_value = shares * current_price
    pnl          = market_value - cost_basis
    pnl_pct      = (pnl / cost_basis) * 100

    total_cost    += cost_basis
    total_mkt_val += market_value

    rows.append({
        "Ticker":          ticker,
        "Sector":          sector,
        "Shares":          shares,
        "Purchase Price":  purchase_price,
        "Current Price":   round(current_price, 2),
        "Cost Basis ($)":  round(cost_basis, 2),
        "Mkt Value ($)":   round(market_value, 2),
        "P&L ($)":         round(pnl, 2),
        "P&L (%)":         round(pnl_pct, 2),
    })

df = pd.DataFrame(rows).sort_values("Mkt Value ($)", ascending=False)
df["Weight (%)"] = (df["Mkt Value ($)"] / total_mkt_val * 100).round(2)
grand_total = total_mkt_val + CASH

print("── HOLDINGS SNAPSHOT ──────────────────────────────────────\n")
display_cols = ["Ticker", "Sector", "Shares", "Current Price",
                "Mkt Value ($)", "P&L ($)", "P&L (%)", "Weight (%)"]
print(df[display_cols].to_string(index=False))

total_pnl     = total_mkt_val - total_cost
total_pnl_pct = (total_pnl / total_cost) * 100

print(f"\n{'─'*55}")
print(f"  Equity Market Value:     ${total_mkt_val:>12,.2f}")
print(f"  Cash Position:           ${CASH:>12,.2f}")
print(f"  Total Portfolio Value:   ${grand_total:>12,.2f}")
print(f"  Total Cost Basis:        ${total_cost:>12,.2f}")
print(f"  Total Unrealized P&L:    ${total_pnl:>12,.2f}  ({total_pnl_pct:+.2f}%)")

# ── 4. RISK METRICS ───────────────────────────────────────────

print("\n── RISK METRICS ───────────────────────────────────────────\n")

port_returns = pd.Series(0.0, index=prices.index)

for ticker, (shares, purchase_price, sector) in holdings.items():
    try:
        ticker_prices  = prices[ticker].dropna()
        ticker_returns = ticker_prices.pct_change().dropna()
        mkt_val        = shares * float(ticker_prices.iloc[-1])
        weight         = mkt_val / total_mkt_val
        aligned        = ticker_returns.reindex(port_returns.index).fillna(0)
        port_returns  += aligned * weight
    except:
        pass

port_returns = port_returns.replace(0, np.nan).dropna()

bench_returns  = prices[BENCHMARK].pct_change().dropna()
bench2_returns = prices[BENCHMARK2].pct_change().dropna()

aligned_idx  = port_returns.index.intersection(bench_returns.index)
port_r       = port_returns.loc[aligned_idx]
bench_r      = bench_returns.loc[aligned_idx]
bench2_r     = bench2_returns.loc[aligned_idx]

ann_vol      = port_r.std() * np.sqrt(252) * 100
sharpe       = (port_r.mean() * 252) / (port_r.std() * np.sqrt(252))
max_dd       = ((1 + port_r).cumprod() / (1 + port_r).cumprod().cummax() - 1).min() * 100
cum_port     = (1 + port_r).cumprod() - 1
cum_bench    = (1 + bench_r).cumprod() - 1
cum_bench2   = (1 + bench2_r).cumprod() - 1

total_return = cum_port.iloc[-1] * 100
bench_return = cum_bench.iloc[-1] * 100
alpha        = total_return - bench_return

# Beta calculation
cov_matrix   = np.cov(port_r, bench_r)
beta         = cov_matrix[0, 1] / cov_matrix[1, 1]

print(f"  Annualized Volatility:   {ann_vol:.2f}%")
print(f"  Sharpe Ratio:            {sharpe:.3f}")
print(f"  Beta vs S&P 500:         {beta:.3f}")
print(f"  Max Drawdown:            {max_dd:.2f}%")
print(f"  Portfolio Return (YTD):  {total_return:+.2f}%")
print(f"  S&P 500 (SPY) YTD:       {bench_return:+.2f}%")
print(f"  Alpha vs S&P 500:        {alpha:+.2f}%")

# ── 5. SECTOR BREAKDOWN ────────────────────────────────────────

print("\n── SECTOR BREAKDOWN ───────────────────────────────────────\n")
sector_df = df.groupby("Sector").agg(
    Holdings=("Ticker", "count"),
    Market_Value=("Mkt Value ($)", "sum"),
    Weight=("Weight (%)", "sum")
).sort_values("Weight", ascending=False)
sector_df["Weight"]       = sector_df["Weight"].map(lambda x: f"{x:.1f}%")
sector_df["Market_Value"] = sector_df["Market_Value"].map(lambda x: f"${x:,.0f}")
print(sector_df.to_string())

# ── 6. VISUALIZATIONS ─────────────────────────────────────────

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#0D1B2A")
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

NAVY  = "#1F4E79"
BLUE  = "#2E75B6"
LBLUE = "#9DC3E6"
WHITE = "#FFFFFF"
GREEN = "#70AD47"
RED   = "#FF5252"
GOLD  = "#FFC000"
TEAL  = "#00BCD4"

def style_ax(ax, title):
    ax.set_facecolor("#132030")
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BLUE)

# Chart 1: Cumulative returns vs benchmarks
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(cum_port.index,   cum_port.values   * 100, color=GOLD,  linewidth=2,   label="Portfolio")
ax1.plot(cum_bench.index,  cum_bench.values  * 100, color=LBLUE, linewidth=1.5, label="S&P 500 (SPY)", linestyle="--")
ax1.plot(cum_bench2.index, cum_bench2.values * 100, color=GREEN, linewidth=1.5, label="Nasdaq 100 (QQQ)", linestyle=":")
ax1.axhline(0, color=WHITE, linewidth=0.6, alpha=0.3)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_xlabel("Date", fontsize=9)
ax1.set_ylabel("Cumulative Return (%)", fontsize=9)
ax1.legend(fontsize=8, facecolor="#132030", labelcolor=WHITE, loc="upper left")
style_ax(ax1, "Portfolio Performance vs Benchmarks (YTD)")

# Chart 2: Portfolio weights
ax2 = fig.add_subplot(gs[0, 2])
top  = df.nlargest(10, "Weight (%)")
cols = [GOLD, BLUE, LBLUE, GREEN, TEAL, "#FF9800", "#E91E63", "#9C27B0", "#F44336", "#4CAF50"]
bars = ax2.barh(top["Ticker"], top["Weight (%)"], color=cols[:len(top)], edgecolor="#0D1B2A")
ax2.set_xlabel("Portfolio Weight (%)", fontsize=9)
ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
for bar, val in zip(bars, top["Weight (%)"]):
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", fontsize=7, color=WHITE)
style_ax(ax2, "Holdings by Weight")

# Chart 3: P&L by holding
ax3 = fig.add_subplot(gs[1, :2])
df_s   = df.sort_values("P&L ($)")
colors = [GREEN if x > 0 else RED for x in df_s["P&L ($)"]]
ax3.barh(df_s["Ticker"], df_s["P&L ($)"], color=colors, edgecolor="#0D1B2A", linewidth=0.5)
ax3.axvline(0, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.5)
ax3.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:,.0f}K"))
ax3.set_xlabel("Unrealized P&L ($)", fontsize=9)
style_ax(ax3, "Unrealized P&L by Holding")

# Chart 4: Sector allocation
ax4 = fig.add_subplot(gs[1, 2])
sector_vals = df.groupby("Sector")["Mkt Value ($)"].sum()
s_colors    = [GOLD, BLUE, LBLUE, GREEN, TEAL]
wedges, texts, autotexts = ax4.pie(
    sector_vals.values, labels=sector_vals.index,
    autopct="%1.1f%%", colors=s_colors[:len(sector_vals)],
    pctdistance=0.8, startangle=90)
for t in texts:
    t.set_color(WHITE); t.set_fontsize(7)
for at in autotexts:
    at.set_color(WHITE); at.set_fontsize(7)
style_ax(ax4, "Sector Allocation")

fig.suptitle(
    "Dynamic Stock Portfolio Tracker  |  Shaquille-Benjamin Andah  |  Wooster '27",
    fontsize=12, fontweight="bold", color=WHITE, y=0.98)

plt.savefig("project2_portfolio_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"\n  Dashboard saved to: project2_portfolio_dashboard.png")
print("\n" + "=" * 65)
print("  Run complete. Upload both files to GitHub.")
print("=" * 65)
