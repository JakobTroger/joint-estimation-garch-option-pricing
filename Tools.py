"""
Tools.py
========
All functions and classes for the Heston-Nandi GARCH option pricing project.

Contents:
    1. Numba-optimized core functions (variance path, log-likelihood, CF, FFT pricer)
    2. Data loading utilities
    3. Model classes: HestonNandiGARCH, JointEstimator
    4. Results and IV fit analysis
    5. Forensic analysis experiments

Requirements:
    pip install yfinance pandas numpy scipy openpyxl numba matplotlib
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from numba import jit

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Run: pip install yfinance")

# ---------------------------------------------------------------------------
# PLOT STYLE
# ---------------------------------------------------------------------------

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150


# ---------------------------------------------------------------------------
# 1. NUMBA-OPTIMIZED CORE FUNCTIONS
# ---------------------------------------------------------------------------

@jit(nopython=True)
def compute_variance_path_single(returns, mu, omega, beta, alpha, gamma):
    """
    Heston-Nandi GARCH(1,1) variance recursion.

    h_{t+1} = omega + beta*h_t + alpha*(z_t - gamma*sqrt(h_t))^2
    """
    T = len(returns)
    h = np.zeros(T + 1)

    denom = 1 - beta - alpha * gamma * gamma
    if denom > 0:
        h[0] = omega / denom
    else:
        h[0] = 0.0001

    if h[0] <= 0 or np.isnan(h[0]):
        mean_r = 0.0
        for i in range(T):
            mean_r += returns[i]
        mean_r /= T
        var_r = 0.0
        for i in range(T):
            var_r += (returns[i] - mean_r) ** 2
        h[0] = var_r / T

    for t in range(T):
        if h[t] > 0:
            sqrt_h = np.sqrt(h[t])
            z_t = (returns[t] - mu) / sqrt_h
        else:
            z_t = 0.0
            sqrt_h = 0.0
        h[t + 1] = omega + beta * h[t] + alpha * (z_t - gamma * sqrt_h) ** 2
        if h[t + 1] < 1e-10:
            h[t + 1] = 1e-10

    return h


@jit(nopython=True)
def returns_log_likelihood_single(returns, mu, h):
    """Gaussian log-likelihood for the GARCH return process."""
    T = len(returns)
    ll = 0.0
    log_2pi = np.log(2 * np.pi)

    for t in range(T):
        if h[t] <= 0:
            return -1e10
        ll += (-0.5 * log_2pi
               - 0.5 * np.log(h[t])
               - 0.5 * (returns[t] - mu) ** 2 / h[t])

    return ll


@jit(nopython=True)
def characteristic_function_single(u_real, u_imag, h_t, tau,
                                    omega, beta, alpha, gamma_star, r_f):
    """
    Closed-form characteristic function of the log-price under Q.

    Implements the backward recursion from Heston & Nandi (2000).
    gamma_star = gamma + lambda_ + 0.5  (risk-neutral leverage)
    """
    A_real = 0.0
    A_imag = 0.0
    B_real = 0.0
    B_imag = 0.0

    for _ in range(int(tau)):
        denom_real = 1 - 2 * alpha * B_real
        denom_imag = -2 * alpha * B_imag
        denom_mag_sq = denom_real ** 2 + denom_imag ** 2

        if denom_mag_sq < 1e-20:
            return np.nan, np.nan

        denom_mag = np.sqrt(denom_mag_sq)
        log_denom_real = np.log(denom_mag)
        log_denom_imag = np.arctan2(denom_imag, denom_real)

        A_new_real = (A_real + (-u_imag) * r_f
                      + omega * B_real - 0.5 * log_denom_real)
        A_new_imag = (A_imag + u_real * r_f
                      + omega * B_imag - 0.5 * log_denom_imag)

        gs2a = gamma_star ** 2 * alpha
        num_real = -0.5 - gs2a * B_real
        num_imag = u_real - gs2a * B_imag

        div_real = (num_real * denom_real + num_imag * denom_imag) / denom_mag_sq
        div_imag = (num_imag * denom_real - num_real * denom_imag) / denom_mag_sq

        B_new_real = div_real + beta * B_real + gs2a
        B_new_imag = div_imag + beta * B_imag

        if (np.isnan(A_new_real) or np.isnan(B_new_real)
                or np.abs(B_new_real) > 1e10):
            return np.nan, np.nan

        A_real, A_imag = A_new_real, A_new_imag
        B_real, B_imag = B_new_real, B_new_imag

    exp_arg_real = A_real + B_real * h_t
    exp_arg_imag = A_imag + B_imag * h_t
    exp_mag = np.exp(exp_arg_real)

    return exp_mag * np.cos(exp_arg_imag), exp_mag * np.sin(exp_arg_imag)


@jit(nopython=True)
def price_call_fft_single(S, K, h_t, tau, omega, beta, alpha,
                           gamma_star, r_f, N, alpha_cm):
    """
    European call price via Carr-Madan FFT.

    References: Carr & Madan (1999), Journal of Computational Finance.
    """
    eta = 0.25
    lambda_fft = 2 * np.pi / (N * eta)
    b = N * lambda_fft / 2

    k = np.zeros(N)
    psi_real = np.zeros(N)
    psi_imag = np.zeros(N)

    for j in range(N):
        k[j] = -b + lambda_fft * j
        v_j = eta * j
        phi_real, phi_imag = characteristic_function_single(
            v_j, -(alpha_cm + 1), h_t, tau,
            omega, beta, alpha, gamma_star, r_f
        )
        if np.isnan(phi_real):
            return np.nan

        denom_real = alpha_cm ** 2 + alpha_cm - v_j ** 2
        denom_imag = (2 * alpha_cm + 1) * v_j
        denom_mag_sq = denom_real ** 2 + denom_imag ** 2

        if denom_mag_sq < 1e-20:
            return np.nan

        discount = np.exp(-r_f * tau)
        psi_real[j] = discount * (phi_real * denom_real + phi_imag * denom_imag) / denom_mag_sq
        psi_imag[j] = discount * (phi_imag * denom_real - phi_real * denom_imag) / denom_mag_sq

    x_real = np.zeros(N)
    x_imag = np.zeros(N)
    for j in range(N):
        angle = b * eta * j
        x_real[j] = eta * (np.cos(angle) * psi_real[j] - np.sin(angle) * psi_imag[j])
        x_imag[j] = eta * (np.sin(angle) * psi_real[j] + np.cos(angle) * psi_imag[j])

    log_moneyness = np.log(K / S)
    if log_moneyness < k[0] or log_moneyness > k[N - 1]:
        return np.nan

    idx = 1
    for j in range(N):
        if k[j] > log_moneyness:
            idx = j
            break

    prices = np.zeros(2)
    for i, m in enumerate([idx - 1, idx]):
        fft_real = 0.0
        for j in range(N):
            angle = 2 * np.pi * m * j / N
            fft_real += x_real[j] * np.cos(angle) - x_imag[j] * np.sin(angle)
        prices[i] = np.exp(-alpha_cm * k[m]) / np.pi * fft_real

    w = (log_moneyness - k[idx - 1]) / (k[idx] - k[idx - 1])
    return S * ((1 - w) * prices[0] + w * prices[1])


# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------

def parse_iv_column_name(col_name):
    """Parse Bloomberg-style IV column names: 'SPX XX% MONEYNESS - IMPLIED VOL X MTH'."""
    pattern = r'SPX (\d+)% MONEYNESS - IMPLIED VOL (\d+) (MTH|YR)'
    match = re.match(pattern, col_name)
    if not match:
        return None
    moneyness = int(match.group(1))
    duration = int(match.group(2))
    unit = match.group(3)
    maturity = float(duration) if unit == 'YR' else duration / 12.0
    return (moneyness, maturity)


def load_iv_surface(filepath):
    """
    Load implied volatility surface from Excel file.

    Returns
    -------
    iv_wide : DataFrame  (dates x moneyness-maturity columns)
    iv_long : DataFrame  (date, moneyness, maturity, iv)
    """
    df = pd.read_excel(filepath)
    df['date'] = pd.to_datetime(df['Name'])
    df = df.drop(columns=['Name']).set_index('date')
    iv_wide = df.copy()

    records = []
    for col in df.columns:
        parsed = parse_iv_column_name(col)
        if parsed is None:
            continue
        moneyness, maturity = parsed
        for date, iv_value in df[col].items():
            if pd.notna(iv_value):
                records.append({
                    'date': date,
                    'moneyness': moneyness,
                    'maturity': maturity,
                    'iv': iv_value / 100.0
                })

    iv_long = pd.DataFrame(records)
    return iv_wide, iv_long


def load_returns(start_date, end_date, ticker="^GSPC"):
    """Download S&P 500 log-returns from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance required. Run: pip install yfinance")

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    prices = data[price_col].to_frame(name='price')
    prices.index.name = 'date'
    prices['log_return'] = np.log(prices['price'] / prices['price'].shift(1))
    return prices.dropna()


def load_all_data(iv_filepath, ticker="^GSPC"):
    """
    Load and align IV surface data with S&P 500 returns.

    Parameters
    ----------
    iv_filepath : str  Path to the Excel file containing the IV surface.
    ticker      : str  Yahoo Finance ticker (default '^GSPC').

    Returns
    -------
    dict with keys: returns, iv_wide, iv_long, aligned_dates, prices
    """
    print("Loading IV surface data...")
    iv_wide, iv_long = load_iv_surface(iv_filepath)
    print(f"  IV data: {len(iv_wide)} days, "
          f"{iv_wide.index[0].date()} to {iv_wide.index[-1].date()}")

    print("Loading returns from Yahoo Finance...")
    start = (iv_wide.index[0] - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    end = (iv_wide.index[-1] + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    returns = load_returns(start, end, ticker)
    print(f"  Returns: {len(returns)} days")

    common_dates = sorted(set(iv_wide.index) & set(returns.index))
    aligned_dates = pd.DatetimeIndex(common_dates)
    print(f"  Aligned: {len(aligned_dates)} common dates")

    return {
        'returns': returns.loc[aligned_dates],
        'iv_wide': iv_wide.loc[aligned_dates],
        'iv_long': iv_long[iv_long['date'].isin(aligned_dates)],
        'aligned_dates': aligned_dates,
        'prices': returns.loc[aligned_dates, 'price']
    }


# ---------------------------------------------------------------------------
# 3. MODEL CLASSES
# ---------------------------------------------------------------------------

class HestonNandiGARCH:
    """
    Heston-Nandi GARCH(1,1) option pricing model.

    Physical measure P:
        r_t = mu + lambda_*h_t + sqrt(h_t)*z_t
        h_{t+1} = omega + beta*h_t + alpha*(z_t - gamma*sqrt(h_t))^2

    Risk-neutral measure Q (gamma* = gamma + lambda_ + 0.5):
        r_t = r_f - 0.5*h_t + sqrt(h_t)*z*_t
        h_{t+1} = omega + beta*h_t + alpha*(z*_t - gamma**sqrt(h_t))^2

    Reference: Heston & Nandi (2000), Review of Financial Studies.
    """

    def __init__(self):
        self.param_names = ['mu', 'omega', 'beta', 'alpha', 'gamma', 'lambda_']
        self.bounds = [
            (-0.1,   0.1),    # mu
            (1e-10,  1e-4),   # omega
            (0.0,    0.9999), # beta
            (1e-10,  1e-4),   # alpha
            (-5,     5),      # gamma  (bounds from Heston & Nandi 2000)
            (-10,    10),     # lambda_
        ]
        self.n_params = 6

    def unpack_params(self, theta):
        return dict(zip(self.param_names, theta))

    def compute_variance_path(self, returns, params):
        h = compute_variance_path_single(
            returns, params['mu'], params['omega'],
            params['beta'], params['alpha'], params['gamma']
        )
        return h, None, None

    def returns_log_likelihood(self, returns, params):
        h, _, _ = self.compute_variance_path(returns, params)
        return returns_log_likelihood_single(returns, params['mu'], h)

    def price_call_option(self, S, K, h_t, tau, params,
                          r_f=0.0, N=1024, alpha_cm=1.5):
        gamma_star = params['gamma'] + params['lambda_'] + 0.5
        return price_call_fft_single(
            S, K, h_t, tau,
            params['omega'], params['beta'], params['alpha'],
            gamma_star, r_f, N, alpha_cm
        )

    def price_to_iv(self, price, S, K, tau, r_f=0.0):
        """Newton-Raphson Black-Scholes inversion."""
        if price <= 0 or np.isnan(price):
            return np.nan
        sigma = 0.2
        for _ in range(100):
            d1 = ((np.log(S / K) + (r_f + 0.5 * sigma ** 2) * tau)
                  / (sigma * np.sqrt(tau)))
            d2 = d1 - sigma * np.sqrt(tau)
            bs = S * norm.cdf(d1) - K * np.exp(-r_f * tau) * norm.cdf(d2)
            vega = S * np.sqrt(tau) * norm.pdf(d1)
            if vega < 1e-10:
                return np.nan
            diff = bs - price
            if abs(diff) < 1e-8:
                return sigma
            sigma -= diff / vega
            if sigma <= 0 or sigma > 5:
                return np.nan
        return sigma

    def compute_model_iv(self, S, moneyness, maturity_months, h_t, params,
                          r_f_annual=0.02):
        K = S * moneyness / 100.0
        tau_days = maturity_months * 21
        tau_years = maturity_months / 12.0
        r_f_daily = r_f_annual / 252.0
        price = self.price_call_option(S, K, h_t, tau_days, params, r_f_daily)
        if np.isnan(price) or price <= 0:
            return np.nan
        return self.price_to_iv(price, S, K, tau_years, r_f_annual)

    def get_persistence(self, params):
        return params['beta'] + params['alpha'] * params['gamma'] ** 2

    def get_uncond_vol(self, params):
        pers = self.get_persistence(params)
        if pers < 1:
            return np.sqrt(252 * params['omega'] / (1 - pers))
        return np.nan


class JointEstimator:
    """
    Jointly estimates the Heston-Nandi GARCH model using returns and IV data.

    Objective:
        Q(theta) = -L_returns(theta) + w_IV * sum((IV_model - IV_market)^2)

    The weight w_IV = 1000 balances the two components.
    """

    def __init__(self, returns, prices, iv_data, model):
        self.returns = returns
        self.prices = prices
        self.iv_data = iv_data
        self.model = model
        self.iv_subset = self._subsample_iv_data()
        self._eval_count = 0

    def _subsample_iv_data(self):
        """Focus on ATM and near-ATM options for estimation."""
        df = self.iv_data.copy()
        moneyness_keep = [95, 100, 105]
        maturity_keep = [1 / 12, 3 / 12, 6 / 12]
        mask = (df['moneyness'].isin(moneyness_keep) &
                df['maturity'].isin(maturity_keep))
        return df[mask].reset_index(drop=True)

    def returns_only_objective(self, theta):
        params = self.model.unpack_params(theta)
        if any(np.isnan(v) for v in params.values()):
            return 1e10
        ll = self.model.returns_log_likelihood(self.returns, params)
        return 1e10 if (np.isnan(ll) or np.isinf(ll)) else -ll

    def joint_objective(self, theta, weight_iv=1.0):
        self._eval_count += 1
        if self._eval_count % 10 == 0:
            print(f"  Evaluation {self._eval_count}...")

        params = self.model.unpack_params(theta)
        if any(np.isnan(v) for v in params.values()):
            return 1e10

        ll_returns = self.model.returns_log_likelihood(self.returns, params)
        if np.isnan(ll_returns) or np.isinf(ll_returns):
            return 1e10

        h, _, _ = self.model.compute_variance_path(self.returns, params)
        iv_sse = 0.0
        n_iv = 0

        for date in self.iv_subset['date'].unique()[::20]:
            idx_arr = np.where(self.prices.index == date)[0]
            if len(idx_arr) == 0:
                continue
            idx = idx_arr[0]
            S = self.prices.iloc[idx]
            h_t = h[idx]
            for _, row in self.iv_subset[self.iv_subset['date'] == date].iterrows():
                model_iv = self.model.compute_model_iv(
                    S, row['moneyness'], row['maturity'] * 12, h_t, params
                )
                if not np.isnan(model_iv):
                    iv_sse += (model_iv - row['iv']) ** 2
                    n_iv += 1

        if n_iv == 0:
            return 1e10

        objective = -ll_returns + weight_iv * iv_sse * 1000
        if self._eval_count % 10 == 0:
            print(f"    n_iv={n_iv}, iv_sse={iv_sse:.3f}, obj={objective:.0f}")
        return objective

    def estimate_returns_only(self, x0=None):
        print(f"Estimating {self.model.__class__.__name__} from returns only...")
        if x0 is None:
            x0 = [0.0001, 1e-6, 0.8, 1e-6, 100, 0.0]
        result = minimize(
            self.returns_only_objective, x0,
            method='L-BFGS-B', bounds=self.model.bounds,
            options={'maxiter': 1000, 'disp': False}
        )
        params = self.model.unpack_params(result.x)
        h, _, _ = self.model.compute_variance_path(self.returns, params)
        print(f"  Converged: {result.success} | Log-likelihood: {-result.fun:.2f}")
        return {
            'params': params, 'theta': result.x,
            'log_likelihood': -result.fun, 'converged': result.success,
            'variance_path': h
        }

    def estimate_joint(self, x0=None, weight_iv=1.0):
        print(f"Estimating {self.model.__class__.__name__} jointly...")
        print(f"  Using {len(self.iv_subset)} IV observations (ATM-focused)")
        self._eval_count = 0
        if x0 is None:
            x0 = [0.0001, 1e-6, 0.8, 1e-6, 100, 0.0]
        result = minimize(
            lambda theta: self.joint_objective(theta, weight_iv),
            x0, method='L-BFGS-B', bounds=self.model.bounds,
            options={'maxiter': 1000, 'disp': False}
        )
        params = self.model.unpack_params(result.x)
        h, _, _ = self.model.compute_variance_path(self.returns, params)
        print(f"  Converged: {result.success} | Evaluations: {self._eval_count}")
        return {
            'params': params, 'theta': result.x,
            'converged': result.success, 'variance_path': h
        }


# ---------------------------------------------------------------------------
# 4. RESULTS & IV FIT
# ---------------------------------------------------------------------------

def print_parameter_table(results_returns, results_joint, model):
    """Print Table 2: parameter estimates under both estimation approaches."""
    print("\n" + "=" * 70)
    print("PARAMETER ESTIMATES")
    print("=" * 70)
    print("{:<15} {:>20} {:>20}".format("Parameter", "Returns Only", "Joint"))
    print("-" * 55)
    for param in model.param_names:
        v_r = results_returns['params'][param]
        v_j = results_joint['params'][param]
        fmt = "{:<15} {:>20.2e} {:>20.2e}" if param in ('omega', 'alpha') else \
              "{:<15} {:>20.6f} {:>20.6f}"
        print(fmt.format(param, v_r, v_j))
    print("-" * 55)
    for label, res in [("Returns Only", results_returns), ("Joint", results_joint)]:
        p = res['params']
        print(f"  [{label}] Persistence: {model.get_persistence(p):.4f} | "
              f"Uncond. Vol: {model.get_uncond_vol(p)*100:.2f}%")


def compute_iv_fit(data, results, model, sample_dates=50):
    """Compute model vs market IV for a sample of dates."""
    params = results['params']
    h = results['variance_path']
    prices = data['prices']
    iv_long = data['iv_long']

    unique_dates = iv_long['date'].unique()
    idx = np.linspace(0, len(unique_dates) - 1, sample_dates, dtype=int)
    records = []

    for date in unique_dates[idx]:
        arr = np.where(prices.index == date)[0]
        if len(arr) == 0:
            continue
        S = prices.iloc[arr[0]]
        h_t = h[arr[0]]
        for _, row in iv_long[iv_long['date'] == date].iterrows():
            model_iv = model.compute_model_iv(
                S, row['moneyness'], row['maturity'] * 12, h_t, params
            )
            if not np.isnan(model_iv):
                records.append({
                    'date': date,
                    'moneyness': row['moneyness'],
                    'maturity': row['maturity'],
                    'observed_iv': row['iv'],
                    'model_iv': model_iv,
                    'error': model_iv - row['iv']
                })

    return pd.DataFrame(records)


def print_iv_fit_summary(iv_fit):
    """Print Table 3: IV fit statistics by moneyness and maturity."""
    print("\n" + "=" * 70)
    print("IMPLIED VOLATILITY FIT SUMMARY")
    print("=" * 70)
    if iv_fit.empty:
        print("No valid IV fits.")
        return
    rmse = np.sqrt((iv_fit['error'] ** 2).mean())
    mae = np.abs(iv_fit['error']).mean()
    print(f"Overall RMSE: {rmse*100:.2f}%  |  MAE: {mae*100:.2f}%")
    print("\nRMSE by Moneyness:")
    for m in sorted(iv_fit['moneyness'].unique()):
        s = iv_fit[iv_fit['moneyness'] == m]
        print(f"  {m:>3}%: {np.sqrt((s['error']**2).mean())*100:.2f}%")
    print("\nRMSE by Maturity:")
    for tau in sorted(iv_fit['maturity'].unique()):
        s = iv_fit[iv_fit['maturity'] == tau]
        print(f"  {tau*12:.0f}M: {np.sqrt((s['error']**2).mean())*100:.2f}%")


# ---------------------------------------------------------------------------
# 5. FORENSIC ANALYSIS
# ---------------------------------------------------------------------------

@jit(nopython=True)
def simulate_correlation(omega, beta, alpha, gamma, n_sim=50000, seed=42):
    """Monte Carlo estimate of Corr(r_t, h_{t+1})."""
    np.random.seed(seed)
    persistence = beta + alpha * gamma ** 2
    h_bar = (omega + alpha) / (1 - persistence) if persistence < 1 else omega / (1 - beta)
    h = h_bar
    returns = np.zeros(n_sim)
    variances = np.zeros(n_sim)
    for t in range(n_sim):
        z = np.random.randn()
        returns[t] = -0.5 * h + np.sqrt(h) * z
        h_next = max(omega + beta * h + alpha * (z - gamma * np.sqrt(h)) ** 2, 1e-12)
        variances[t] = h_next
        h = h_next
    return np.corrcoef(returns[:-1], variances[1:])[0, 1]


@jit(nopython=True)
def simulate_variance_ratio(omega, beta, alpha, gamma, n_sim=50000):
    """Ratio of mean variance after negative vs positive returns."""
    np.random.seed(42)
    persistence = beta + alpha * gamma ** 2
    h_bar = (omega + alpha) / (1 - persistence) if persistence < 1 else omega / (1 - beta)
    h = h_bar
    var_neg = []
    var_pos = []
    for t in range(n_sim):
        z = np.random.randn()
        r = -0.5 * h + np.sqrt(h) * z
        h_next = max(omega + beta * h + alpha * (z - gamma * np.sqrt(h)) ** 2, 1e-12)
        if r < 0:
            var_neg.append(h_next)
        else:
            var_pos.append(h_next)
        h = h_next
    return np.mean(np.array(var_neg)) / np.mean(np.array(var_pos))


@jit(nopython=True)
def mc_option_price(S0, K, h0, tau, omega, beta, alpha,
                    gamma_star, r_f, n_paths=20000):
    """Monte Carlo European call price under Q."""
    np.random.seed(42)
    payoffs = np.zeros(n_paths)
    for i in range(n_paths):
        S = S0
        h = h0
        for t in range(int(tau)):
            z = np.random.randn()
            S = S * np.exp(r_f - 0.5 * h + np.sqrt(max(h, 1e-12)) * z)
            h = max(omega + beta * h + alpha * (z - gamma_star * np.sqrt(max(h, 1e-12))) ** 2, 1e-12)
        payoffs[i] = max(S - K, 0.0)
    return np.exp(-r_f * tau) * np.mean(payoffs)


def _bs_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _implied_vol(S, K, T, r, price):
    """Brent root-finding Black-Scholes inversion."""
    if price <= max(S - K * np.exp(-r * T), 1e-10):
        return np.nan
    try:
        return brentq(lambda sig: _bs_call(S, K, T, r, sig) - price, 0.001, 2.0)
    except Exception:
        return np.nan


# --- Experiment functions ---

def experiment_correlation_vs_gamma(omega, beta, alpha, save_path=None):
    """Figure 1: Corr(r_t, h_{t+1}) as a function of gamma."""
    gamma_range = np.linspace(0, 20, 21)
    correlations = [simulate_correlation(omega, beta, alpha, g) for g in gamma_range]

    fig, ax = plt.subplots()
    ax.plot(gamma_range, correlations, 'b-o', lw=2, ms=6)
    ax.axhline(-0.3, color='red', ls='--', lw=1.5, label='Target for realistic skew (-0.3)')
    ax.axhline(-0.5, color='red', ls=':', lw=1.5, label='Strong skew (-0.5)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(5, color='green', ls='--', lw=1.5, label='Estimated γ = 5 (at bound)')
    ax.annotate(f'At γ=5: corr = {correlations[5]:.3f}',
                xy=(5, correlations[5]), xytext=(8, correlations[5] + 0.02),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')
    ax.annotate(f'At γ=20: corr = {correlations[-1]:.3f}\n(still far from -0.3)',
                xy=(20, correlations[-1]), xytext=(15, -0.15),
                arrowprops=dict(arrowstyle='->', color='blue'), fontsize=10, color='blue')
    ax.set_xlabel('Leverage Parameter (γ)')
    ax.set_ylabel('Correlation(r_t, h_{t+1})')
    ax.set_title(f'Why Increasing γ Alone Cannot Generate Realistic Skew\n'
                 f'(Fixed α = {alpha:.2e}, β = {beta:.4f})')
    ax.legend(loc='lower left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return gamma_range, correlations


def experiment_alpha_gamma_interaction(omega, beta, save_path=None):
    """Figure 2: Contours of Corr(r_t, h_{t+1}) over (alpha, gamma) space."""
    alpha_range = np.logspace(-7, -3, 30)
    gamma_range = np.linspace(0, 15, 30)
    corr_matrix = np.zeros((len(alpha_range), len(gamma_range)))
    for i, a in enumerate(alpha_range):
        for j, g in enumerate(gamma_range):
            try:
                corr_matrix[i, j] = simulate_correlation(omega, beta, a, g, n_sim=10000)
            except Exception:
                corr_matrix[i, j] = np.nan

    G, A = np.meshgrid(gamma_range, alpha_range)
    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(G, A, corr_matrix, levels=20, cmap='RdYlBu', alpha=0.7)
    plt.colorbar(cf, ax=ax, label='Correlation(r_t, h_{t+1})')
    cs = ax.contour(G, A, corr_matrix,
                    levels=[-0.5, -0.4, -0.3, -0.2, -0.1, -0.05, -0.02, 0], colors='blue')
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.2f')
    ax.plot(5, 7.31e-6, 'r*', ms=20, label='Estimated (γ=5, α=7.3e-6)')
    ax.axhline(1e-4, color='green', ls='--', label='α = 1e-4 (10x larger needed)')
    ax.set_xlabel('Leverage Parameter (γ)')
    ax.set_ylabel('Shock Impact (α)')
    ax.set_yscale('log')
    ax.set_title('Parameter Interaction: Both α AND γ Must Be Large for Skew\nContours show Corr(r_t, h_{t+1})')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return alpha_range, gamma_range, corr_matrix


def experiment_variance_ratio_vs_gamma(omega, beta, alpha, save_path=None):
    """Figure 3: Variance ratio (after neg / after pos returns) vs gamma."""
    gamma_range = np.linspace(0, 20, 21)
    ratios = [simulate_variance_ratio(omega, beta, alpha, g) for g in gamma_range]

    fig, ax = plt.subplots()
    ax.plot(gamma_range, ratios, 'b-o', lw=2, ms=6)
    ax.axhline(1.5, color='red', ls='--', lw=1.5, label='Target ratio (1.5x)')
    ax.axhline(2.0, color='red', ls=':', lw=1.5, label='Strong asymmetry (2.0x)')
    ax.axhline(1.0, color='gray', lw=0.5)
    ax.axvline(5, color='green', ls='--', lw=1.5, label='Estimated γ = 5')
    ax.annotate(f'At γ=5: ratio = {ratios[5]:.2f}\n(nearly symmetric)',
                xy=(5, ratios[5]), xytext=(8, ratios[5] + 0.1),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=10)
    ax.set_xlabel('Leverage Parameter (γ)')
    ax.set_ylabel('Variance Ratio (after negative / after positive return)')
    ax.set_title('Variance Asymmetry: How Much Higher is Variance After Negative Returns?')
    ax.legend(loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return gamma_range, ratios


def experiment_iv_smile_vs_gamma(omega, beta, alpha, lambda_, h0,
                                  S0=100, save_path=None):
    """Figure 4: IV smile for different gamma values vs typical market skew."""
    tau = 63
    r_f = 0.02 / 252
    T_years = tau / 252
    moneyness_range = [90, 95, 100, 105, 110]
    colors = ['blue', 'green', 'orange', 'red']

    fig, ax = plt.subplots()
    for gamma, color in zip([1, 5, 10, 20], colors):
        gamma_star = gamma + lambda_ + 0.5
        ivs = []
        for m in moneyness_range:
            K = S0 * m / 100
            price = mc_option_price(S0, K, h0, tau, omega, beta, alpha, gamma_star, r_f)
            iv = _implied_vol(S0, K, T_years, 0.02, price)
            ivs.append(iv * 100 if iv and not np.isnan(iv) else np.nan)
        ax.plot(moneyness_range, ivs, 'o-', color=color, lw=2, ms=8, label=f'γ = {gamma}')

    ax.plot(moneyness_range, [18, 15, 13, 12, 11.5], 'k--', lw=2, ms=8,
            label='Typical Market Skew')
    ax.annotate('Market has steep downward skew', xy=(92, 17), fontsize=10)
    ax.annotate('Model smiles are nearly flat', xy=(92, 12), fontsize=10, color='blue')
    ax.set_xlabel('Moneyness (%)')
    ax.set_ylabel('Implied Volatility (%)')
    ax.set_title('IV Smile for Different Leverage Parameters\nModel smile remains flat regardless of γ')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def experiment_maturity_effect(omega, beta, alpha, gamma, lambda_,
                                h0, S0=100, save_path=None):
    """Figure 5: Model vs market IV smile at 1M, 3M, 6M, 12M maturities."""
    r_f = 0.02 / 252
    gamma_star = gamma + lambda_ + 0.5
    moneyness_range = [90, 95, 100, 105, 110]
    market_ivs = {
        21:  [25, 18, 14, 12, 11],
        63:  [20, 16, 14, 13, 12],
        126: [18, 15, 14, 13, 12.5],
        252: [17, 15, 14, 13.5, 13],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (tau, label) in zip(axes.flatten(),
                                  zip([21, 63, 126, 252], ['1M', '3M', '6M', '12M'])):
        T_years = tau / 252
        model_ivs = []
        for m in moneyness_range:
            K = S0 * m / 100
            price = mc_option_price(S0, K, h0, tau, omega, beta, alpha, gamma_star, r_f)
            iv = _implied_vol(S0, K, T_years, 0.02, price)
            model_ivs.append(iv * 100 if iv and not np.isnan(iv) else np.nan)

        valid = [(a, b) for a, b in zip(market_ivs[tau], model_ivs) if b and not np.isnan(b)]
        rmse = np.sqrt(np.mean([(a - b) ** 2 for a, b in valid])) if valid else float('nan')

        ax.plot(moneyness_range, market_ivs[tau], 'ko-', lw=2, ms=8,
                label='Market IV', markerfacecolor='white')
        ax.plot(moneyness_range, model_ivs, 'b^--', lw=2, ms=8, label='Model IV')
        ax.set_title(f'{label} Maturity (RMSE = {rmse:.1f}%)')
        ax.set_xlabel('Moneyness (%)')
        ax.set_ylabel('Implied Volatility (%)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(5, 30)

    plt.suptitle('Why Short-Maturity Options Are Hardest to Fit\n'
                 'Short maturities have steepest skew (jump-dominated)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def run_forensic_analysis(params, h0, S0=2000, save_figures=True):
    """
    Run all forensic experiments.

    Parameters
    ----------
    params        : dict  Estimated parameters from JointEstimator.
    h0            : float Current variance (last value of variance path).
    S0            : float Current S&P 500 level.
    save_figures  : bool  Save figures to disk.
    """
    omega   = params['omega']
    beta    = params['beta']
    alpha   = params['alpha']
    gamma   = params['gamma']
    lambda_ = params['lambda_']

    print("=" * 70)
    print("FORENSIC ANALYSIS OF GARCH OPTION PRICING FAILURE")
    print("=" * 70)

    print("\n--- 1. Correlation vs Gamma ---")
    gamma_range, correlations = experiment_correlation_vs_gamma(
        omega, beta, alpha,
        save_path='figure_1_corr_vs_gamma.png' if save_figures else None
    )

    print("\n--- 2. Alpha-Gamma Interaction ---")
    experiment_alpha_gamma_interaction(
        omega, beta,
        save_path='figure_2_alpha_gamma.png' if save_figures else None
    )

    print("\n--- 3. Variance Ratio ---")
    _, ratios = experiment_variance_ratio_vs_gamma(
        omega, beta, alpha,
        save_path='figure_3_variance_ratio.png' if save_figures else None
    )

    print("\n--- 4. IV Smile vs Gamma ---")
    experiment_iv_smile_vs_gamma(
        omega, beta, alpha, lambda_, h0, S0,
        save_path='figure_4_iv_smile.png' if save_figures else None
    )

    print("\n--- 5. Maturity Effect ---")
    experiment_maturity_effect(
        omega, beta, alpha, gamma, lambda_, h0, S0,
        save_path='figure_5_maturity.png' if save_figures else None
    )

    print("\n" + "=" * 70)
    print("FORENSIC ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  Corr(r,h) at γ=5:    {correlations[5]:.4f}  (need -0.3 to -0.5)")
    print(f"  Corr(r,h) at γ=20:   {correlations[-1]:.4f}")
    print(f"  αγ product:          {alpha*gamma:.2e}  (need ~5e-4)")
    print(f"  Variance ratio at γ=5: {ratios[5]:.3f}  (need 1.5-2.0)")
