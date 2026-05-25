# Investment Research Portfolio
### Shaquille-Benjamin Andah | Business Economics, The College of Wooster '27
#### Market Risk | Financial Analysis | Quantitative Research

---

## Project 1 — Interest Rate Sensitivity Model

A Python-based model that prices a five-bond treasury portfolio, calculates 
risk metrics, and stress tests portfolio value across eight parallel rate shock 
scenarios (-200bps to +200bps).

### Concepts Covered
- Bond pricing (present value of cash flows)
- Macaulay Duration and Modified Duration
- Convexity (curvature correction for large rate moves)
- Duration + Convexity approximation vs. full repricing
- Parallel rate shift scenario analysis

### Portfolio
| Bond | Face Value | Coupon | YTM |
|---|---|---|---|
| US Treasury 2Y | $1,000,000 | 4.25% | 4.30% |
| US Treasury 5Y | $1,000,000 | 4.50% | 4.60% |
| US Treasury 10Y | $1,000,000 | 4.75% | 4.90% |
| Corp Bond (IG) 5Y | $750,000 | 5.50% | 5.70% |
| Corp Bond (IG) 7Y | $750,000 | 5.75% | 6.00% |

### Key Results
- Total Portfolio Value: $4,465,845
- Portfolio Modified Duration: 4.81
- +100bps shock: -$207,765 (-4.65%)
- -100bps shock: +$222,415 (+4.98%)

### Tools
Python | NumPy | Pandas | Matplotlib

---

## Coming Soon
- Dynamic Stock Portfolio Tracker (Python + yfinance)
- Ghana Macroeconomic Dashboard (Tableau + World Bank data)

---

*Built as part of an independent finance research portfolio 
targeting market risk and treasury analyst roles.*
