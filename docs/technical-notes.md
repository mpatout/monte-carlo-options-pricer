> **Note on line references.** These notes were written against the original
> single-file version of the pricer. The code has since been split into the
> `mc_pricer/` package, so the line numbers below no longer line up — the
> function names do. See the module table in the README for where each lives.

# Monte Carlo Options Pricing - Detailed Technical Answers

## Lines 119-130: Implied Volatility Calculation

**Equation Used:** Black-Scholes formula with Newton-Raphson iteration

```python
# Line 120: d1 formula from Black-Scholes
d1 = (ln(S/K) + (r + 0.5*σ²)*T) / (σ√T)

# Line 121: d2 formula
d2 = d1 - σ√T

# Lines 123-126: Black-Scholes pricing formula
Call Price = S*N(d1) - K*e^(-rT)*N(d2)
Put Price = K*e^(-rT)*N(-d2) - S*N(-d1)

# Line 127: Vega for Newton-Raphson
Vega = S*φ(d1)*√T  (where φ is normal PDF)

# Line 133: Newton-Raphson update
σ_new = σ_old + (Market_Price - BS_Price) / Vega
```

**Purpose:** Inverts the Black-Scholes formula to back out implied volatility from market option prices.

---

## calculate_iv_from_price vs build_implied_vol_surface

### `calculate_iv_from_price` (Lines 95-138)
- **Purpose:** Calculates IV for a **single option** at one strike/expiry
- **Method:** Newton-Raphson root finding
- **Input:** Market price, spot, strike, expiry, rate, option type
- **Output:** Single IV value (or 0.0 if fails)

### `build_implied_vol_surface` (Lines 140-187)
- **Purpose:** Builds **entire IV surface** across all strikes for an expiration
- **Method:** Calls `calculate_iv_from_price` for each strike in the option chain
- **Input:** Ticker, expiry date, option type, spot
- **Output:** Dict with arrays of strikes, IVs, moneyness, volumes
- **Filters:** Removes options with zero volume, invalid bid/ask, or failed IV calculations

**Relationship:** `build_implied_vol_surface` is the wrapper that fetches market data and calls `calculate_iv_from_price` repeatedly to build a complete surface.

---

## fetch_risk_free_rate: Manual Entry for 1Y/2Y

**Current Rates Fetched (Lines 273-328):**
- 3M Treasury (^IRX): 0.25Y
- 5Y Treasury (^FVX): 5.0Y
- 10Y Treasury (^TNX): 10.0Y
- 30Y Treasury (^TYX): 30.0Y

**Issue:** Missing 1Y and 2Y rates. The function interpolates, but you want manual entry.

**Recommendation to add:**
```python
# After line 302, add:
manual_1y = input("Enter 1Y Treasury rate (or press Enter to interpolate): ").strip()
manual_2y = input("Enter 2Y Treasury rate (or press Enter to interpolate): ").strip()

if manual_1y:
    tenors.append(1.0)
    rates.append(float(manual_1y))
if manual_2y:
    tenors.append(2.0)
    rates.append(float(manual_2y))

# Then sort tenors and rates together
sorted_pairs = sorted(zip(tenors, rates))
tenors = [t for t, r in sorted_pairs]
rates = [r for t, r in sorted_pairs]
```

---

## fetch_all_market_data - Why Fetch Again?

**Answer:** It's **not** fetching again - it's an **aggregation function**.

**What it does (Lines 330-363):**
1. Calls `fetch_ticker_data()` → gets spot price
2. Calls `build_implied_vol_surface()` → gets IV surface
3. Calls `fetch_dividends()` → gets dividend schedule
4. Calls `fetch_risk_free_rate()` → gets rate curve
5. Returns **one unified dict** with all market data

**Purpose:** Centralized data collection function to avoid passing 10 separate arguments. It's the "master fetcher" that coordinates all other fetchers.

---

## prepare_iv_data - Do We Already Compute/Filter IV?

**Answer:** Yes and no. Here's the difference:

### `build_implied_vol_surface` (Line 140)
- **Purpose:** Initial IV calculation from option prices
- **Filters:** Volume > 0, valid bid/ask, IV > 0
- **Usage:** Fetches raw IV surface from market

### `prepare_iv_data` (Line 365)
- **Purpose:** **Additional filtering** for calibration quality
- **Filters applied:**
  - Volume/open interest thresholds
  - Bid-ask spread < 50% (removes wide markets)
  - **Moneyness range [0.8, 1.2]** (focuses on ATM ±20%)
  - Assigns ATM-weighted weights
- **Usage:** Prepares data specifically for Heston/GBM calibration

**Moneyness [0.8, 1.2] Appropriate?**
- **Yes** for calibration - focuses on liquid ATM options
- **No** if you want to capture full volatility smile/skew
- For equity options, consider [0.7, 1.3] to capture skew better
- For deep OTM analysis, use [0.6, 1.5]

**Recommendation:** Keep [0.8, 1.2] for standard calibration, but add a parameter to adjust if needed.

---

## Moneyness

**Definition:** 
```
Moneyness = Strike / Spot
```

**Examples (Spot = $100):**
- Strike $80 → Moneyness = 0.80 (20% OTM put / ITM call)
- Strike $100 → Moneyness = 1.00 (ATM)
- Strike $120 → Moneyness = 1.20 (20% OTM call / ITM put)

**Why Use It:**
- Normalizes strikes across different stock prices
- Makes it easy to compare option surfaces across tickers
- [0.9, 1.1] = ±10% around ATM

---

## calibrate_gbm_vol - Is It Used for Advanced Greeks?

**Answer:** No, these are separate.

### `calibrate_gbm_vol` (Lines 633-653)
- **Purpose:** Extracts a single volatility value from the IV surface for GBM simulation
- **Method:** Interpolates IV surface at the target strike
- **Used:** Before simulating paths (GBM needs one vol parameter)

### Advanced Greeks (Lines 1936-2106)
- **Purpose:** Computes Greeks with adaptive Richardson extrapolation
- **When Asked:** Terminal prompt "Run Greeks bump sensitivity test? (y/N)"
- **NOT related to GBM vol calibration**

**The "advanced greeks" terminal prompt refers to:**
- `compute_adaptive_fd_greek()` - Richardson extrapolation for Greeks
- `test_greek_bump_sensitivity()` - Tests stability across bump sizes

**GBM vol calibration happens automatically for all simulations.**

---

## check_feller_condition - What Is It?

**Feller Condition (Lines 655-669):**
```
2*κ*θ > σ²
```

**What it means:**
- **κ** (kappa): Mean reversion speed
- **θ** (theta): Long-run variance
- **σ** (sigma): Volatility of volatility

**Purpose:** Ensures variance in Heston model **never goes negative**

**If Satisfied:** Variance stays strictly positive
**If Violated:** Variance can hit zero (QE scheme handles this with absorption/reflection)

**Example:**
- κ = 2.0, θ = 0.04, σ = 0.3
- LHS: 2 * 2.0 * 0.04 = 0.16
- RHS: 0.3² = 0.09
- 0.16 > 0.09 ✓ PASS

---

## price_european_heston_characteristic - What Does It Do?

**Purpose (Lines 673-728):** Prices **European** options under Heston model using the **characteristic function** (Heston's 1993 closed-form solution).

**Why it exists:**
1. **For calibration** - prices options analytically (no MC noise)
2. **More accurate than MC** for finding Heston parameters
3. **Not used for final American pricing** - that's done via LSM

**Workflow:**
1. Computes characteristic function Φ(φ)
2. Integrates: P_j = 0.5 + (1/π) ∫ [e^(-iφ ln K) * Φ(φ) / (iφ)] dφ
3. Call price = S*P_1 - K*e^(-rT)*P_2

**This is NOT where we adjust MC simulation results.** It's purely for Heston parameter calibration by matching to market prices.

---

## heston_iv_from_price - What Does Inversion Mean?

**Purpose (Lines 731-745):** Converts a Heston option **price** back to **implied volatility**.

**Method:**
1. Takes Heston model price (from `price_european_heston_characteristic`)
2. Finds σ such that Black-Scholes(σ) = Heston_Price
3. Uses `brentq` root-finder: BS(σ) - Heston_Price = 0

**Why:** For calibration, we want to compare model IVs to market IVs (not prices directly).

**Example:**
- Heston model → Call price = $5.23
- Invert → IV = 28.4%
- Compare to market IV = 30.1%
- Minimize: (28.4% - 30.1%)²

---

## calibrate_heston_params - Two Methods?

**Two Methods (Lines 747-929):**

### Method 1: Moment Matching (Lines 771-805)
- **Speed:** Fast (< 1 second)
- **Accuracy:** Approximate
- **How:** Matches ATM vol, skew, dispersion from IV surface
- **Always runs** to get initial guess

### Method 2: Smile Calibration (Lines 810-918)
- **Speed:** Slow (~10-60 seconds depending on strikes)
- **Accuracy:** Precise
- **How:** Prices at all strikes, minimizes IV errors
- **Only if:** `use_smile_calibration=True` (user choice via prompt)

**Are They Used Together?**
- Yes: Moment matching provides **initial guess**
- Then: Smile calibration refines using differential evolution
- Result: Best of both (speed + accuracy)

**User Control:** Prompt asks "Run market-consistent calibration? (y/N)"

**How Much Longer?**
- Moment matching: < 1 second
- Smile calibration: 10-60 seconds (depends on # strikes and path count)
- For 5 strikes @ 10k paths: ~15 seconds

---

## How Do the Three Heston Functions Work Together?

```
[Market Data: Option Prices at Multiple Strikes]
              ↓
[calibrate_heston_params] ← Main orchestrator
   ├─ Step 1: Moment matching (initial guess)
   ├─ Step 2: For each strike:
   │      ├─ [price_european_heston_characteristic] ← Price with trial params
   │      ├─ [heston_iv_from_price] ← Convert price → IV
   │      └─ Compute error: (Model IV - Market IV)²
   └─ Step 3: Minimize total error → Optimal params

[Output: HestonParams(v0, κ, θ, σ, ρ)]
              ↓
[simulate_paths_heston_qe] ← Use calibrated params
              ↓
[price_american_lsm] ← Price American option
```

**Heston Model Explained:**
- **Stochastic volatility:** Volatility itself follows a random process
- **Better than GBM:** Captures volatility smile/skew observed in markets
- **Five parameters:** v0 (initial var), κ (mean reversion), θ (long-run var), σ (vol-of-vol), ρ (spot-vol correlation)

---

## brownian_bridge - What Is It Used For?

**Purpose (Lines 931-957):** Constructs Brownian paths via **midpoint interpolation** instead of sequential stepping.

**Standard Method (Sequential):**
```
W(t+1) = W(t) + √dt * Z
```

**Brownian Bridge (Midpoint):**
```
1. Generate W(T) at endpoint
2. Fill in midpoints recursively: W(mid) = (W(left) + W(right))/2 + correction
```

**Where Used in Code:**
- Line 1033: `if use_bridge:` in `simulate_paths_gbm`
- **Default: OFF** (`use_bridge=False`)

**Why Not Used:**
- **Pro:** Lower variance for **European** path-independent options
- **Con:** Higher variance for **path-dependent** options (like American with dividends)
- American exercise depends on prices at **all time steps**, so bridge is suboptimal

**Recommendation:** Keep disabled for American options.

---

## How Does QSM (Quasi-Monte Carlo) Work?

**QSM = Quasi-Monte Carlo with Sobol Sequences**

### generate_sobol_normals (Lines 959-970)
- **Purpose:** Generate **low-discrepancy** random numbers
- **Method:** Sobol sequence → Inverse CDF → Normal variates
- **Where:** Used in all path simulations (GBM and Heston)

**How It Works:**
1. Generate Sobol points in [0,1]^d (uniform, low-discrepancy)
2. Apply inverse normal CDF: Z = Φ^(-1)(U)
3. Result: Normal random variables with better coverage than pseudo-random

**Benefit:**
- Converges at O(1/n) vs O(1/√n) for standard MC
- ~10-50% variance reduction in practice
- **Used automatically** in lines 1019, 1125, 1135

**Part of Simulation:**
- Replaces `np.random.randn()` calls
- Generates all random shocks (Z values) for price paths

---

## simulate_paths_gbm: Risk-Neutral Drift = r - q - σ²/2?

**Your Question (Line 1055):**
```python
drift = (r - 0.5 * vol ** 2) * dt
```

**Why not just r?**

### Full Risk-Neutral Drift Formula:
```
μ_Q = r - q - σ²/2
```

**Where:**
- **r** = risk-free rate
- **q** = dividend yield
- **σ²/2** = Itô correction (from Itô's lemma)

**The σ²/2 term comes from:**
- We simulate log prices: d(ln S) = (μ - σ²/2)dt + σdW
- Exponentiating: S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
- Without -σ²/2, the expected value would be e^(μt + σ²t/2), not e^(μt)

**In Your Code:**
- Dividends are handled **discretely** (line 1119-1126), not as continuous yield q
- So drift = r - σ²/2 (no q term because dividends are explicit jumps)

**This is correct!** The -σ²/2 is mandatory for lognormal processes.

---

## Paths Under Both Heston QE and GBM - Why Both?

**Answer:** They're **alternatives**, not used together.

### User chooses ONE model:
- **GBM** (Lines 972-1127): Constant volatility, simpler, faster
- **Heston** (Lines 1129-1249): Stochastic volatility, more realistic, slower

**Selection:**
- Line 2869: `model: str='gbm'` parameter in `run_pricer()`
- Line 2958-2969: If-else branching

**Which to Use:**
- **GBM:** Standard options, quick analysis, < 30 DTE
- **Heston:** Options with vol smile, longer DTE, earnings events

**They don't work together** - it's one or the other per pricing run.

---

## What Does the Laguerre Model Do?

**Purpose (Lines 1251-1334):** Generates **basis functions** for LSM (Longstaff-Schwartz Method) regression.

**Not a separate model** - it's part of the LSM pricing algorithm.

### Laguerre Polynomials (Lines 1306-1323):
```python
L_0(x) = 1
L_1(x) = 1 - x
L_2(x) = 1 - 2x + x²/2
L_3(x) = 1 - 3x + 3x²/2 - x³/6
```

**Why Laguerre (not simple polynomials)?**
- **Orthogonal** → Better regression conditioning
- **Weighted for (0, ∞)** → Perfect for positive stock prices
- **Standard in LSM papers** → Proven to work well

### How It Fits in Simulation:
```
[Simulated Paths S(t)]
         ↓
[build_lsm_features] ← Computes Laguerre basis at each time
         ↓
[Regression: Continuation Value = β₀L₀ + β₁L₁ + β₂L₂ + β₃L₃]
         ↓
[Compare: Exercise Now vs Hold]
```

**Degree = 3:** Uses L₀, L₁, L₂, L₃ (4 basis functions)

---

## payoff_function - What Does It Do?

**Purpose (Lines 1325-1333):** Computes option **intrinsic value** (not cost difference).

```python
Call: max(S - K, 0)
Put: max(K - S, 0)
```

**Examples:**
- Call with S=105, K=100 → Payoff = $5
- Put with S=95, K=100 → Payoff = $5
- Call with S=95, K=100 → Payoff = $0

**This is NOT** "option cost vs calculated value" - that's the P&L calculation (different function).

---

## price_american_lsm - Is LSM for Optimal Exercise?

**Yes!** LSM = Longstaff-Schwartz Method for American option pricing.

**Purpose (Lines 1336-1487):** Determines **optimal early exercise policy** at each time step.

### LSM Algorithm:
1. **Start at expiry:** Payoff = max(S_T - K, 0)
2. **Step backward in time:** For each t from T-1 to 0:
   - For **ITM paths** only:
     - Regress continuation value on Laguerre basis
     - Compare: Exercise Now vs Expected Continuation
     - Exercise if Immediate > Continuation
3. **Result:** Exercise matrix (True/False for each path/time)

**How It Works:**
```
Time t: S = [102, 98, 105, 95]  (Strike = 100, Call)
Immediate Exercise = [2, 0, 5, 0]
Continuation Value (from regression) = [4, 0, 3, 0]

Decision:
- Path 1: 2 < 4 → HOLD
- Path 2: 0 = 0 → HOLD (OTM)
- Path 3: 5 > 3 → EXERCISE ✓
- Path 4: 0 = 0 → HOLD (OTM)
```

**Output:** Option price = Average of discounted cashflows across all paths.

---

## black_scholes_price Used for Control Variate?

**Yes!** Lines 1489-1510 and Lines 1512-1573.

### Control Variate Method:
```
V_adjusted = V_American + β * (BS_Euro_Analytical - BS_Euro_MC)
```

**What it does:**
1. Price American option via MC → V_Am (has MC noise)
2. Price European option via MC → V_Eu_MC (same noise)
3. Price European via Black-Scholes formula → V_Eu_Exact (no noise)
4. **Correction:** Add β * (Exact - MC) to American price
5. **Result:** Reduced variance (often 50-90% reduction)

**This is NOT adjusting for market prices** - it's a variance reduction technique using a known analytical solution as a control.

---

## calibrate_model_to_market - How Does It Work?

**Purpose (Lines 1575-1758):** Iteratively adjusts model parameters to **match market option prices** (not just IVs).

### Workflow:
1. Select **5 strikes around ATM** (±2 strikes)
2. Fetch market mid-prices for these strikes
3. **Optimize:**
   - For each parameter guess
   - Simulate paths with those parameters
   - Price each option via LSM
   - Compute error: Σ(Model_Price - Market_Price)²
4. **Output:** Calibrated parameters that minimize pricing error

**Difference from IV Calibration:**
- **IV calibration:** Matches model IVs to market IVs
- **Price calibration:** Matches model prices to market prices
- Both are valid; price calibration is more direct

**Used when?** User selects "Run market-consistent calibration? (y/N)"

---

## compute_pathwise_delta - What Is It?

**Purpose (Lines 1760-1791):** Computes Delta using **pathwise derivative** method (not finite differences).

**Method:**
```python
∂V/∂S = E[∂Payoff/∂S * dS/dS₀]

For Call: ∂Payoff/∂S = 1 if S > K, else 0
For Put: ∂Payoff/∂S = -1 if S < K, else 0

dS/dS₀ = S_exercised / S₀
```

**Advantages:**
- Lower variance than finite differences
- Uses single simulation (no re-pricing needed)
- Unbiased estimator

**Is this the "advanced Greek"?**
- **No** - this is an alternative method (pathwise vs FD)
- Advanced Greeks = `compute_adaptive_fd_greek` with Richardson extrapolation
- Both can be used, but FD with Richardson is more common

---

## compute_likelihood_ratio_vega - What's This?

**Purpose (Lines 1793-1819):** Computes Vega using **likelihood ratio** (score function) method.

**Method:**
```python
Vega = E[V * Score]

Score = Σ [(Z_t² - 1)/σ - Z_t√dt]
```

**Where Z_t are the random shocks in the simulation.**

**Advantages:**
- Unbiased
- Single simulation (no vol bump needed)
- Lower variance than finite differences for certain payoffs

**Is this used?**
- Optional feature (flag: `use_likelihood_ratio=True`)
- Default: Uses finite differences
- Generally **FD is more stable** for American options

---

## compute_adaptive_fd_greek - Is This the Long Accurate Greek?

**Yes!** Lines 1821-1900.

**Purpose:** Computes Greeks via **adaptive finite differences with Richardson extrapolation**.

### Richardson Extrapolation:
```
R(h) = [4*FD(h) - FD(2h)] / 3

Achieves O(h⁴) accuracy (vs O(h²) for standard FD)
```

**Adaptive Bump Sizing:**
- **ATM:** Smaller bumps (more precise)
- **OTM/ITM:** Larger bumps (avoid MC noise)
- **Long DTE:** Larger bumps
- **Short DTE:** Smaller bumps

**Is Richardson extrapolation the IV inversion method?**
- **No** - Richardson is for numerical derivatives (Greeks)
- IV inversion uses Newton-Raphson (different technique)

**Used for ALL Greeks?**
- **Yes** (if user selects advanced calibration)
- Delta, Gamma, Vega, Rho all use Richardson
- Theta uses discrete 1-day shift (Richardson fails near expiry)

---

## All T0 Greeks Computed with Adaptive Richardson?

**Answer:** Yes (except Theta). Lines 1902-2106.

### Delta (∂V/∂S):
```
V(S + h) - V(S - h)
─────────────────── → Richardson extrapolation
        2h
```

### Gamma (∂²V/∂S²):
```
V(S + h) - 2V(S) + V(S - h)
─────────────────────────── → Richardson extrapolation
          h²
```

### Vega (∂V/∂σ):
```
V(σ + h) - V(σ)
───────────────── → Richardson extrapolation
       h
```

### Rho (∂V/∂r):
```
V(r + h) - V(r)
───────────────── → Richardson extrapolation
       h
```

### Theta (∂V/∂t):
```
V(t + 1 day) - V(t)
─────────────────── × 365
        1 day
```

**Why no Richardson for Theta?**
- Time is discrete (trading days)
- Richardson assumes continuous parameter
- 1-day shift is most realistic for trading

---

## Should I Update to Add More Paths Until CV < Thresholds?

**Your Observation:** `test_greek_bump_sensitivity` suggests CV < 5% for Delta, < 10% for Gamma.

### Current Default: 50,000 paths

**Adaptive Path Algorithm:**
```python
# Pseudo-code for adaptive paths
target_delta_cv = 0.05
target_gamma_cv = 0.10
n_paths = 50000

while delta_cv > target_delta_cv or gamma_cv > target_gamma_cv:
    # Double paths
    n_paths *= 2
    
    # Re-compute Greeks
    greeks = compute_greeks_t0(n_paths=n_paths)
    
    # Check convergence
    if n_paths > 1_000_000:
        break  # Max out at 1M paths
```

**Path Counts for Higher Accuracy:**
- **Current (50k):** CV ~5-10% (5-10 minutes)
- **100k paths:** CV ~3-7% (10-20 minutes)
- **250k paths:** CV ~2-5% (30-60 minutes)
- **1M paths:** CV ~1-2% (2-4 hours)

**For 99% Confidence:**
- Need CV < 1%
- Requires ~1M paths
- Runtime: 2-4 hours (depends on model/computer)

**Recommendation:**
- Add adaptive path algorithm as **optional feature**
- Let user specify target CV thresholds
- Default: Keep 50k (good balance speed/accuracy)
- Power users: Enable adaptive mode

---

## Q-Measure vs P-Measure - Are Paths All Q-Measure?

**Correct!** All current pricing is Q-measure (risk-neutral).

### Q-Measure (Risk-Neutral):
- **Drift:** r (risk-free rate)
- **Used for:** Pricing (what's the fair value?)
- **Lines:** All `simulate_paths_gbm` calls without `drift_override`

### P-Measure (Real-World):
- **Drift:** μ (expected stock return, e.g., 10%)
- **Used for:** Forecasting P&L (what profit will I make?)
- **Lines:** 2017-2148 in `compute_real_world_pnl`

### Is Q to P Drift Logic Sufficient?

**Current Logic (Line 2003):**
```python
p_drift = expected_return  # Just sets μ = 10%
```

**This IS correct!** 

**P-measure drift:**
```
Under P: dS/S = μ dt + σ dW_P
```

**What about risk premium?**
- Risk premium = μ - r (e.g., 10% - 4.5% = 5.5%)
- Already embedded in μ
- No need for explicit premium calculation

**Is This Sufficient for P-Measure Option?**
- **Yes** for P&L forecasting
- **No** for hedging ratios (Delta/Gamma under P ≠ under Q)
- Consider adding P-measure Greeks if doing real-world hedging

**Code Already Has P-Measure:**
- `compute_real_world_pnl` (Line 2017) uses `drift_override=expected_return`
- This switches from Q to P

---

## apply_exercise_policy - Are Paths Re-Run?

**Purpose (Lines 2217-2243):** Apply **Q-measure exercise policy** to **new P-measure paths**.

**What It Does:**
1. **Q-measure pricing** computes optimal exercise boundary
2. **P-measure simulation** generates new paths with μ = 10% (not r)
3. **Apply Q-policy to P-paths:** Exercise when P-path crosses Q-boundary

**Are Paths Re-Run?**
- **Yes** - new P-measure paths are simulated
- **No** - exercise decisions come from original Q-measure LSM regression

**Why?**
- Realistic: Traders exercise based on market prices (Q), not forecasts (P)
- Avoids re-solving LSM under P (which would be theoretically wrong)

**Used Where:** Line 2042 in `compute_real_world_pnl`

---

## convergence_sweep - What Does It Do?

**Purpose (Lines 2272-2290):** Tests how price **converges** as path count increases.

### Runs Pricing with Different Path Counts:
```python
n_paths = [10000, 25000, 50000, 100000, 250000]

For each n:
    Price option with n paths
    Record: price, std_error, confidence interval
    
Output: DataFrame showing convergence
```

**What You See:**
```
n_paths  |  price    | std_error | ci_lower | ci_upper | price_change
---------|-----------|-----------|----------|----------|-------------
10,000   |  $5.234   | $0.087    | $5.063   | $5.405   |    -
25,000   |  $5.189   | $0.055    | $5.082   | $5.296   | $0.045
50,000   |  $5.201   | $0.039    | $5.125   | $5.277   | $0.012
100,000  |  $5.198   | $0.028    | $5.143   | $5.253   | $0.003
250,000  |  $5.200   | $0.018    | $5.165   | $5.235   | $0.002
```

**Purpose:** Verify MC has converged before trusting results.

**When to Use:** If you suspect price is unstable, run this to find adequate path count.

---

## price_option - Just to Get Input?

**Purpose (Lines 2764-3074):** **Interactive CLI** for end-users.

### What It Does:
1. **Prompts user** for ticker, option type, expiry, strike
2. **Fetches market data** automatically
3. **Runs pricer** with all features
4. **Displays results** with plots
5. **Optional:** Calibration, bump sensitivity tests

**Not just input** - it's the complete user interface that orchestrates:
- Data fetching
- Model selection
- Pricing
- Greeks
- Forecasts
- Visualizations

**Two Modes:**
```bash
# Interactive
python "Monte Carlo Options Simulation.py"

# Command line
python "Monte Carlo Options Simulation.py" AAPL C
```

---

## Summary Recommendations

### High Priority Updates:

1. **Add 1Y/2Y manual rate entry** (lines 302-328)
2. **Make moneyness range configurable** (line 365) - add parameter
3. **Add adaptive path algorithm** (optional feature)
4. **Document Q vs P measure** more clearly in docstrings
5. **Add P-measure Greeks** (optional) for hedging under real-world measure

### Parameter Optimization:

**For Most Accurate Results (99% confidence):**
- Paths: 500k - 1M
- Runtime: 1-4 hours
- CV: < 1%

**For Production Trading (balanced):**
- Paths: 100k - 250k
- Runtime: 15-45 minutes
- CV: 2-5%

**For Quick Analysis:**
- Paths: 50k (current default)
- Runtime: 5-10 minutes
- CV: 5-10%

### Moneyness Ranges by Use Case:

- **Standard calibration:** [0.8, 1.2]
- **Capture volatility skew:** [0.7, 1.3]
- **Full smile analysis:** [0.6, 1.5]
- **Deep OTM analysis:** [0.5, 2.0]

---

## Quick Reference Table

| Function | Purpose | Used When |
|----------|---------|-----------|
| `calculate_iv_from_price` | Single IV from price | Building IV surface |
| `build_implied_vol_surface` | Full IV surface | Fetching market data |
| `prepare_iv_data` | Filter/weight for calibration | Before Heston calibration |
| `calibrate_gbm_vol` | Extract GBM vol | Before GBM simulation |
| `calibrate_heston_params` | Fit Heston to market | Before Heston simulation |
| `price_european_heston_characteristic` | Analytical Heston price | Inside Heston calibration |
| `heston_iv_from_price` | Price → IV conversion | Inside Heston calibration |
| `brownian_bridge` | Midpoint path construction | Optional (disabled default) |
| `generate_sobol_normals` | Low-discrepancy RNs | All simulations |
| `simulate_paths_gbm` | GBM price paths | GBM model pricing |
| `simulate_paths_heston_qe` | Heston price paths | Heston model pricing |
| `build_lsm_features` | Laguerre basis | Inside LSM regression |
| `price_american_lsm` | LSM pricing algorithm | Main pricing function |
| `apply_control_variate` | Variance reduction | After LSM pricing |
| `compute_greeks_t0` | All Greeks with Richardson | After pricing |
| `compute_adaptive_fd_greek` | Single Greek with Richardson | Inside Greeks computation |
| `calibrate_model_to_market` | Match market prices | Optional advanced feature |
| `compute_real_world_pnl` | P-measure forecast | Optional P&L analysis |
| `convergence_sweep` | Test path count convergence | Diagnostics |
| `price_option` | Interactive CLI | End-user interface |

---

## All Code is Q-Measure EXCEPT:

1. `compute_real_world_pnl` (Line 2017) - Uses P-measure with `drift_override`
2. `convert_Q_to_P_drift` (Line 2003) - Helper function for P-measure
3. `apply_exercise_policy` (Line 2217) - Applies Q-policy to P-paths

**Everything else is risk-neutral pricing (Q-measure).**
