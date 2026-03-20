# Joint Estimation of GARCH Option Pricing Model

A forensic analysis of the Heston-Nandi GARCH model applied to S&P 500 option pricing.  
**Author:** Jakob Troger | Financial Econometrics, University of Vienna | January 2026
---

*A personal note: This is my first coding project and my first time using 
GitHub. I'm an Economics and Statistics student, not a computer scientist — 
so if you stumble across something that looks clunky or inefficient, that's 
probably why. Everything you see here was built from scratch alongside my 
Financial Econometrics course, learning as I went. I'm still very much on 
the journey of learning to code, and this project represents an early 
milestone in that process. I'm proud of how it turned out given where I 
started, but please be gentle 😄 — feedback and suggestions are genuinely 
welcome.*

---

## Overview

This project implements and critically evaluates the **Heston-Nandi (2000) GARCH option pricing model** using joint estimation on S&P 500 return and implied volatility data spanning 2007–2025.

Rather than proposing a new model, the goal is to understand **mechanically why** the Heston-Nandi GARCH model fails to replicate the steep implied volatility skew observed in equity index options. The analysis combines analytical derivations, counterfactual simulations, and empirical estimation.

### Key Findings

- Joint estimation identifies a **negative variance risk premium** (λ = −0.006), confirming investors pay for volatility protection.
- Despite fitting return dynamics well, the model produces a **nearly flat IV smile** (overall RMSE: 39%).
- The failure is **structural**: the leverage effect operates through the product αγ = 3.66 × 10⁻⁵, which is too small by an order of magnitude to generate realistic skew.
- Even quadrupling γ to 20 yields a return-variance correlation of only −0.09 — far below the −0.3 to −0.5 required for realistic skew.
- **Short-maturity options** are hardest to fit because their steep skew reflects sudden crash risk, which a diffusion model cannot capture.
- This motivates **jump-diffusion extensions**, which generate skew through an independent mechanism compatible with return dynamics.

---

## Repository Structure

```
joint-estimation-garch-option-pricing/
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── GARCH_Analysis_Troger.ipynb   # Main notebook: estimation + forensic analysis
└── paper/
    └── Final_Project_Troger.pdf       # Full written paper
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/joint-estimation-garch-option-pricing.git
cd joint-estimation-garch-option-pricing
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Provide data

- **S&P 500 returns** are downloaded automatically via `yfinance`.
- **Implied volatility surface data** must be provided as an Excel file with columns following the format:
  ```
  SPX XX% MONEYNESS - IMPLIED VOL X MTH
  ```
  Update the file path in the notebook's data loading section accordingly.

### 4. Run the notebook

```bash
jupyter notebook notebooks/GARCH_Analysis_Troger.ipynb
```

Run the cells in order: **Part 1** (core model) → **Part 2** (forensic functions) → **Part 3** (execute analysis).

---

## Model Specification

The **Heston-Nandi GARCH(1,1)** model specifies log-returns under the physical measure P as:

```
r_t = μ + λh_t + √h_t · z_t,    z_t ~ N(0,1)
h_{t+1} = ω + βh_t + α(z_t − γ√h_t)²
```

Under the risk-neutral measure Q, the leverage parameter shifts to γ* = γ + λ + ½, enabling closed-form option pricing via the characteristic function and Carr-Madan FFT.

**Estimation** combines a return log-likelihood with an implied volatility RMSE objective, weighted to balance the two components.

---

## Forensic Analysis Experiments

| Figure | Experiment | Key Result |
|--------|-----------|------------|
| 1 | Corr(r_t, h_{t+1}) vs. γ | Even γ = 20 yields only −0.09 correlation |
| 2 | α-γ interaction contour plot | Both parameters must be large simultaneously |
| 3 | Variance ratio (neg. vs. pos. returns) | Ratio = 1.02 at γ = 5; target is 1.5–2.0 |
| 4 | IV smile across γ values | Model smile is flat at all γ |
| 5 | Model vs. market IV by maturity | Worst fit at short maturities (jump-dominated) |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scipy` | Optimization (L-BFGS-B) and statistics |
| `numba` | JIT compilation for variance path and FFT |
| `matplotlib` | Plotting |
| `yfinance` | S&P 500 price download |
| `openpyxl` | Excel file reading for IV surface data |

---

## References

- Heston, S. L. & Nandi, S. (2000). *A Closed-Form GARCH Option Valuation Model.* Review of Financial Studies, 13(3).
- Carr, P. & Madan, D. (1999). *Option Valuation Using the Fast Fourier Transform.* Journal of Computational Finance, 2(4).
- Bates, D. S. (2000). *Post-'87 Crash Fears in the S&P 500 Futures Option Market.* Journal of Econometrics, 94.
- Bakshi, G., Cao, C. & Chen, Z. (1997). *Empirical Performance of Alternative Option Pricing Models.* Journal of Finance, 52(5).
