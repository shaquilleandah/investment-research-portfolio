"""
Interest Rate Sensitivity Model
================================
Author: Shaquille-Benjamin Andah
College of Wooster | Business Economics '27
GitHub: github.com/shaquille-andah

This model prices bonds, calculates duration and convexity,
and measures how a portfolio of bonds responds to interest
rate shocks. Directly relevant to treasury and market risk
analysis in banking.

Key concepts covered:
- Bond pricing (present value of cash flows)
- Macaulay Duration (weighted average time to receive cash flows)
- Modified Duration (price sensitivity to rate changes)
- Convexity (curvature correction for large rate moves)
- Parallel rate shift scenarios (+/- 25, 50, 100, 200 bps)
- Portfolio-level P&L and % change under each scenario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec

# ── 1. BOND PRICING FUNCTIONS ─────────────────────────────────

def bond_price(face_value, coupon_rate, ytm, years, freq=2):
    """
    Calculate the present value (price) of a bond.

    Parameters
    ----------
    face_value  : float  — par value of the bond (e.g. 1000)
    coupon_rate : float  — annual coupon rate (e.g. 0.05 for 5%)
    ytm         : float  — yield to maturity (e.g. 0.04 for 4%)
    years       : float  — years to maturity
    freq        : int    — coupon payments per year (2 = semi-annual)

    Returns
    -------
    float — clean price of the bond
    """
    periods = int(years * freq)
    coupon = (coupon_rate * face_value) / freq
    rate_per_period = ytm / freq

    # Present value of coupon payments
    pv_coupons = sum(
        coupon / (1 + rate_per_period) ** t
        for t in range(1, periods + 1)
    )

    # Present value of face value at maturity
    pv_face = face_value / (1 + rate_per_period) ** periods

    return pv_coupons + pv_face


def macaulay_duration(face_value, coupon_rate, ytm, years, freq=2):
    """
    Macaulay Duration — weighted average time to receive cash flows.
    Measured in years. Tells you the 'average life' of the bond.
    """
    periods = int(years * freq)
    coupon = (coupon_rate * face_value) / freq
    rate_per_period = ytm / freq
    price = bond_price(face_value, coupon_rate, ytm, years, freq)

    weighted_times = sum(
        (t / freq) * (coupon / (1 + rate_per_period) ** t)
        for t in range(1, periods + 1)
    )
    weighted_times += (years) * (face_value / (1 + rate_per_period) ** periods)

    return weighted_times / price


def modified_duration(face_value, coupon_rate, ytm, years, freq=2):
    """
    Modified Duration — direct measure of price sensitivity.
    A modified duration of 5 means a 1% rise in rates
    causes approximately a 5% drop in bond price.
    """
    mac_dur = macaulay_duration(face_value, coupon_rate, ytm, years, freq)
    return mac_dur / (1 + ytm / freq)


def convexity(face_value, coupon_rate, ytm, years, freq=2):
    """
    Convexity — captures the curvature in the price/yield relationship.
    Duration alone underestimates price gains and overestimates price
    losses for large rate moves. Convexity corrects for this.
    """
    periods = int(years * freq)
    coupon = (coupon_rate * face_value) / freq
    rate_per_period = ytm / freq
    price = bond_price(face_value, coupon_rate, ytm, years, freq)

    conv_sum = sum(
        coupon * t * (t + 1) / (1 + rate_per_period) ** (t + 2)
        for t in range(1, periods + 1)
    )
    conv_sum += face_value * periods * (periods + 1) / (1 + rate_per_period) ** (periods + 2)

    return conv_sum / (price * freq ** 2)


def price_change_estimate(mod_dur, conv, delta_ytm, price):
    """
    Estimate price change using Duration + Convexity approximation.

    ΔP ≈ -ModDur × ΔY × P  +  0.5 × Convexity × (ΔY)² × P

    This is the formula risk managers use daily to estimate
    how much a bond or portfolio will gain/lose when rates move.
    """
    duration_effect = -mod_dur * delta_ytm * price
    convexity_effect = 0.5 * conv * (delta_ytm ** 2) * price
    return duration_effect + convexity_effect


# ── 2. DEFINE PORTFOLIO ───────────────────────────────────────

"""
A representative bank treasury portfolio with five bonds
spanning different maturities and credit profiles.
This mirrors what you'd see in a real bank's held-to-maturity
or available-for-sale portfolio.
"""

portfolio = pd.DataFrame({
    "Bond":         ["US Treasury 2Y", "US Treasury 5Y", "US Treasury 10Y",
                     "Corp Bond (IG) 5Y", "Corp Bond (IG) 7Y"],
    "Face_Value":   [1_000_000, 1_000_000, 1_000_000, 750_000, 750_000],
    "Coupon_Rate":  [0.0425,    0.0450,    0.0475,    0.0550,  0.0575],
    "YTM":          [0.0430,    0.0460,    0.0490,    0.0570,  0.0600],
    "Years":        [2,         5,         10,        5,       7],
    "Freq":         [2,         2,         2,         2,       2],
})

# ── 3. CALCULATE BASE METRICS ─────────────────────────────────

print("=" * 65)
print("  INTEREST RATE SENSITIVITY MODEL")
print("  Shaquille-Benjamin Andah | College of Wooster '27")
print("=" * 65)

results = []
for _, bond in portfolio.iterrows():
    fv  = bond["Face_Value"]
    cr  = bond["Coupon_Rate"]
    ytm = bond["YTM"]
    yr  = bond["Years"]
    fr  = bond["Freq"]

    price    = bond_price(fv, cr, ytm, yr, fr)
    mac_dur  = macaulay_duration(fv, cr, ytm, yr, fr)
    mod_dur  = modified_duration(fv, cr, ytm, yr, fr)
    conv     = convexity(fv, cr, ytm, yr, fr)
    mkt_val  = price  # price already accounts for face value scaling

    results.append({
        "Bond":              bond["Bond"],
        "Market Value ($)":  round(mkt_val, 2),
        "Coupon (%)":        f"{cr*100:.2f}%",
        "YTM (%)":           f"{ytm*100:.2f}%",
        "Mac Duration":      round(mac_dur, 3),
        "Mod Duration":      round(mod_dur, 3),
        "Convexity":         round(conv, 3),
    })

results_df = pd.DataFrame(results)
print("\n── PORTFOLIO METRICS ──────────────────────────────────────\n")
print(results_df.to_string(index=False))

total_mkt_val = sum(r["Market Value ($)"] for r in results)
print(f"\n  Total Portfolio Market Value: ${total_mkt_val:,.2f}")

# Portfolio-weighted modified duration
port_mod_dur = sum(
    r["Mod Duration"] * r["Market Value ($)"] / total_mkt_val
    for r in results
)
port_conv = sum(
    r["Convexity"] * r["Market Value ($)"] / total_mkt_val
    for r in results
)
print(f"  Portfolio Modified Duration:  {port_mod_dur:.3f}")
print(f"  Portfolio Convexity:          {port_conv:.3f}")

# ── 4. RATE SHOCK SCENARIOS ───────────────────────────────────

print("\n── RATE SHOCK SCENARIOS (Parallel Shifts) ─────────────────\n")

shocks_bps = [-200, -100, -50, -25, +25, +50, +100, +200]
scenario_results = []

for shock_bps in shocks_bps:
    delta_ytm = shock_bps / 10_000  # convert bps to decimal

    bond_changes = []
    total_new_val = 0

    for _, bond in portfolio.iterrows():
        fv  = bond["Face_Value"]
        cr  = bond["Coupon_Rate"]
        ytm = bond["YTM"]
        yr  = bond["Years"]
        fr  = bond["Freq"]

        base_price = bond_price(fv, cr, ytm, yr, fr)
        new_ytm    = ytm + delta_ytm
        new_price  = bond_price(fv, cr, new_ytm, yr, fr)
        pnl        = new_price - base_price
        bond_changes.append(pnl)
        total_new_val += new_price

    total_pnl    = total_new_val - total_mkt_val
    pct_change   = (total_pnl / total_mkt_val) * 100

    scenario_results.append({
        "Rate Shock (bps)": f"{'+' if shock_bps > 0 else ''}{shock_bps}",
        "New Portfolio Value ($)": round(total_new_val, 2),
        "P&L ($)":                 round(total_pnl, 2),
        "P&L (%)":                 round(pct_change, 4),
    })

scenarios_df = pd.DataFrame(scenario_results)
print(scenarios_df.to_string(index=False))

# Duration approximation vs actual (for +100bps)
shock_check = 0.01
approx_pnl = price_change_estimate(port_mod_dur, port_conv, shock_check, total_mkt_val)
actual_row  = next(r for r in scenario_results if r["Rate Shock (bps)"] == "+100")
actual_pnl  = actual_row["P&L ($)"]

print(f"\n  Duration+Convexity estimate (+100bps): ${approx_pnl:,.2f}")
print(f"  Actual repriced P&L        (+100bps): ${actual_pnl:,.2f}")
print(f"  Estimation error:                     ${abs(approx_pnl - actual_pnl):,.2f}")

# ── 5. VISUALIZATIONS ─────────────────────────────────────────

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0D1B2A")
gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

NAVY   = "#1F4E79"
BLUE   = "#2E75B6"
LBLUE  = "#9DC3E6"
WHITE  = "#FFFFFF"
GRAY   = "#AAAAAA"
GREEN  = "#70AD47"
RED    = "#FF5252"
GOLD   = "#FFC000"

ax_style = dict(facecolor="#132030", labelcolor=WHITE,
                tick_params=dict(colors=WHITE))

def style_ax(ax, title):
    ax.set_facecolor("#132030")
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E75B6")

# ── Chart 1: P&L by scenario ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
shocks  = [int(r["Rate Shock (bps)"].replace("+", "")) for r in scenario_results]
pnls    = [r["P&L ($)"] for r in scenario_results]
colors  = [GREEN if p > 0 else RED for p in pnls]
bars    = ax1.bar(range(len(shocks)), pnls, color=colors, width=0.6, edgecolor="#0D1B2A", linewidth=0.5)
ax1.set_xticks(range(len(shocks)))
ax1.set_xticklabels([f"{'+' if s > 0 else ''}{s}bps" for s in shocks], fontsize=9)
ax1.axhline(0, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.5)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:,.0f}K"))
ax1.set_xlabel("Rate Shock (Parallel Shift)", fontsize=10)
ax1.set_ylabel("Portfolio P&L ($)", fontsize=10)
for bar, pnl in zip(bars, pnls):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (5000 if pnl >= 0 else -12000),
             f"${pnl/1000:,.1f}K", ha="center", va="bottom", fontsize=8, color=WHITE)
style_ax(ax1, "Portfolio P&L Under Parallel Rate Shocks")

# ── Chart 2: Price/Yield curve for 10Y Treasury ───────────────
ax2 = fig.add_subplot(gs[1, 0])
ytm_range = np.linspace(0.01, 0.10, 200)
fv, cr, yr, fr = 1_000_000, 0.0475, 10, 2
prices_curve = [bond_price(fv, cr, y, yr, fr) for y in ytm_range]
ax2.plot(ytm_range * 100, prices_curve, color=LBLUE, linewidth=2)
ax2.axvline(0.0490 * 100, color=GOLD, linewidth=1, linestyle="--", alpha=0.8, label="Current YTM")
ax2.set_xlabel("Yield to Maturity (%)", fontsize=10)
ax2.set_ylabel("Bond Price ($)", fontsize=10)
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
ax2.legend(fontsize=8, facecolor="#132030", labelcolor=WHITE)
style_ax(ax2, "Price / Yield Curve (10Y Treasury)")

# ── Chart 3: Modified Duration by bond ────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
bond_names  = [r["Bond"].replace(" ", "\n") for r in results]
mod_durs    = [r["Mod Duration"] for r in results]
bar_colors  = [LBLUE, BLUE, NAVY, GREEN, GOLD]
ax3.barh(range(len(bond_names)), mod_durs, color=bar_colors, edgecolor="#0D1B2A")
ax3.set_yticks(range(len(bond_names)))
ax3.set_yticklabels(bond_names, fontsize=8)
ax3.set_xlabel("Modified Duration (years)", fontsize=10)
ax3.axvline(port_mod_dur, color=RED, linewidth=1.5, linestyle="--", label=f"Portfolio Avg: {port_mod_dur:.2f}")
ax3.legend(fontsize=8, facecolor="#132030", labelcolor=WHITE)
style_ax(ax3, "Modified Duration by Bond")

# ── Title ─────────────────────────────────────────────────────
fig.suptitle("Interest Rate Sensitivity Model  |  Shaquille-Benjamin Andah  |  Wooster '27",
             fontsize=13, fontweight="bold", color=WHITE, y=0.98)

plt.savefig("/mnt/user-data/outputs/interest_rate_sensitivity.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\n  Charts saved.")
