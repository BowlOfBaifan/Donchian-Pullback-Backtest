# QUANTITATIVE STRATEGY REPORT: v1_classic

**Ticker:** SPY  
**Period:** 2023-02-01 to 2025-12-12  
**Status:** Stable / Defensive / Low Beta  

## 1. Strategy Hypothesis
The strategy relies on a "Trend-Following Mean Reversion" hypothesis: in a confirmed long-term uptrend, short-term drops below a significant recent low (20-hour support) represent temporary liquidity flushes rather than structural reversals. By buying these "oversold" dips, the strategy aims to capture the snap-back effect as the price reverts to its primary upward trajectory.

## 2. Strategy Logic Description
* **Universe:** SPY (S&P 500 ETF), Hourly Data (RTH).
* **Trend Filter (Regime):** Long positions are permitted only when the Price is **above** the 200-period Simple Moving Average (SMA). The SMA is calculated using data from $t_{-200}$ to $t_{-1}$ to strictly prevent look-ahead bias.
* **Entry Signal:** Buy when the Close Price drops **below** the Lower Donchian Band (the lowest low of the previous 20 hours).
* **Exit Signal (Take Profit):** Sell when the Close Price rallies **above** the Upper Donchian Band (the highest high of the previous 20 hours).
* **Execution:** All trades are simulated as "Market Orders" executed at the **Open** of the next bar ($t_{+1}$) to account for overnight gaps and realistic slippage.
* **Risk Management:**
    * **Hard Stop Loss:** 2.0% fixed percentage from entry price.
    * **Transaction Costs:** 0.1% per trade (covering commission and slippage).

## 3. Executive Summary
The "v1_classic" Donchian Pullback strategy exhibits a highly defensive, mean-reversion profile that prioritizes capital preservation over aggressive growth. Over the 2-year backtest period, the strategy delivered a Total Return of 12.87% with a Sharpe Ratio of 1.65, indicating a respectable risk-adjusted performance despite significantly lagging the benchmark's return of 68.10%. The system operates as a high-precision "sniper," executing only 42 trades with a high Win Rate of 78%, effectively avoiding market noise but failing to capitalize on the broader momentum of the bull market. While the logic successfully limits Max Drawdown to 8.37%, the restrictive entry criteria result in substantial opportunity costs.

## 4. Key Performance Indicators (KPIs)

| Metric | Value | Assessment |
| :--- | :--- | :--- |
| **Total Return** | 12.87% | Underperforms Benchmark (68.10%) |
| **Sharpe Ratio** | 1.65 | Good (Institutional Acceptable > 1.0) |
| **Win Rate** | 78.05% | High Precision |
| **Profit Factor** | 1.77 | Healthy Profitability |
| **Max Drawdown** | -8.37% | Defensive (Low Risk) |
| **Avg Win / Loss** | 0.51 | Negative Skew (Avg Win < Avg Loss) |

## 5. Strategic Analysis

### Strengths (The Edge)
The strategy’s primary edge lies in its disciplined trade selection and defensive capability. A Win Rate of 78% confirms the validity of buying dips at the 20-period Donchian Lower Band within a long-term uptrend. The system effectively filters out low-quality setups, resulting in a Max Drawdown of just 8.37%, which is significantly lower than typical equity market volatility. This stability makes the strategy an excellent candidate for leverage or as a defensive component in a larger portfolio, as it generates returns with minimal emotional or capital stress.

### Weaknesses (The Drag)
The most glaring weakness is the lack of opportunity capture. With only 42 trades in nearly three years, the strategy sits in cash for extended periods, missing the majority of the benchmark's upside moves. The 200-period SMA combined with the 20-period Donchian channel creates an entry filter that is too stringent for the hourly timeframe, filtering out valid shallow pullbacks. Furthermore, the Total Return of 12.87% is difficult to justify when the simple buy-and-hold benchmark returned over 68%, highlighting that the cost of safety in this specific configuration is too high.

### Risk Factors (The Danger)
The strategy exhibits a dangerous asymmetry in its payoff ratio, with the Average Loss (-1.67%) being roughly double the Average Win (0.86%). This "negative skew" means the system relies entirely on maintaining its high Win Rate (78%) to be profitable. If market dynamics shift, such as a transition to a high-volatility bear market where dips keep dipping and the win rate drops toward 50%, the mathematical expectancy of the system will immediately turn negative. The strategy currently lacks a mechanism to let winners run, capping upside potential while still exposing capital to gap risks on the downside.