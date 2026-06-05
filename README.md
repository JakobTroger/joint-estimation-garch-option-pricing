# Joint Estimation of GARCH Option Pricing Model

**Author:** Jakob Troger | Financial Econometrics, University of Vienna | January 2026

---

## What This Project Does

This project estimates the Heston-Nandi GARCH model using S&P 500 data from 2007 to 2025. The model is used to price options. The goal is not to build a better model. The goal is to understand exactly why this model fails to match real option prices.

### Key Findings

- The model identifies a negative variance risk premium (λ = 0.006). This means investors pay extra to protect against volatility spikes.
- The model fits return data well but produces a nearly flat implied volatility smile. The overall error is 39%.
- The failure is structural. The leverage effect is too weak to generate realistic skew.
- Even with a very large leverage parameter, the return-variance correlation only reaches 0.09. It needs to be at least 0.3 to match market prices.
- Short-maturity options are the hardest to fit. Their steep skew reflects crash risk, which this type of model cannot capture.
- This shows why jump-diffusion models are needed for realistic option pricing.

---

## Repository Structure

```
joint-estimation-garch-option-pricing/
├── README.md
├── .gitignore
├── requirements.txt
├── Tools.py                              # All functions and classes
├── notebooks/
│   └── GARCH_Analysis_Troger.ipynb       # Runs the analysis and shows results
└── paper/
    └── Final_Project_Troger.pdf          # Full written paper
```

---

## How to Run

**Step 1: Clone the repository**
```bash
git clone https://github.com/JakobTroger/joint-estimation-garch-option-pricing.git
cd joint-estimation-garch-option-pricing
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Add your data**

S&P 500 returns are downloaded automatically via yfinance. You need to provide an Excel file with the implied volatility surface. The columns should follow this format:
```
SPX XX% MONEYNESS - IMPLIED VOL X MTH
```
Update the file path in the notebook before running.

**Step 4: Run the notebook**
```bash
jupyter notebook notebooks/GARCH_Analysis_Troger.ipynb
```

---

## Model

The Heston-Nandi GARCH(1,1) model works as follows:

```
r_t = μ + λh_t + √h_t · z_t
h_{t+1} = ω + βh_t + α(z_t − γ√h_t)²
```

Option prices are computed using the characteristic function and the Carr-Madan FFT method. Estimation uses both return data and implied volatility data at the same time.

---

## Experiments

| Figure | What It Shows |
|--------|--------------|
| 1 | How the return-variance correlation changes with the leverage parameter |
| 2 | Why both alpha and gamma need to be large at the same time |
| 3 | How symmetric variance is after positive vs negative returns |
| 4 | Why the implied volatility smile stays flat regardless of the leverage parameter |
| 5 | Why short-maturity options are hardest to fit |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scipy` | Optimization and statistics |
| `numba` | Fast computation |
| `matplotlib` | Plotting |
| `yfinance` | S&P 500 data download |
| `openpyxl` | Reading Excel files |

---

## References

- Heston, S. L. & Nandi, S. (2000). *A Closed-Form GARCH Option Valuation Model.* Review of Financial Studies, 13(3).
- Carr, P. & Madan, D. (1999). *Option Valuation Using the Fast Fourier Transform.* Journal of Computational Finance, 2(4).
- Bates, D. S. (2000). *Post-87 Crash Fears in the S&P 500 Futures Option Market.* Journal of Econometrics, 94.
- Bakshi, G., Cao, C. & Chen, Z. (1997). *Empirical Performance of Alternative Option Pricing Models.* Journal of Finance, 52(5).
