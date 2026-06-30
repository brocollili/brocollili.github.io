---
layout: post
title: "Buying Value Without Buying Value Traps"
date: 2026-06-29
categories: [equities, fundamental]
tags: [value-investing, fundamental-data, vietnam-equities]
math: true
---

When a stock trades at a low P/E, low P/B, or low EV/EBITDA multiple, the natural reaction is to think that it is cheap. But the point is that cheap does not always mean attractive. A stock can look cheap because the market is overlooking it, but it can also look cheap because the underlying business is genuinely weak: earnings are declining, leverage is high, margins are deteriorating, or cash flow is poor.

In this notebook, I test a simple idea: instead of buying cheap stocks unconditionally, I buy cheap stocks with fundamental support. The additional condition is that the company should show reasonable signals in quality, growth, cash flow, and balance-sheet strength.

## 1. Main idea

The baseline signal is `value_score`, constructed from valuation metrics:

- earnings yield;
- book-to-price;
- EBITDA-to-EV.

Then I construct an additional `support_score` from other fundamental groups:

- `quality_score`: ROE, gross margin, and net margin;
- `growth_score`: growth in gross profit, revenue, EBT, and EPS;
- `cash_score`: FCF yield;
- `balance_score`: higher score for lower debt-to-assets.

The final signal is:

```python
conditioned_value_score = value_score * support_score
```

The intuition behind multiplying the two scores is that a stock needs to be both cheap and supported by reasonably healthy fundamentals. If a stock is extremely cheap but has weak support, its final score is pulled down. This is a simple way to reduce the risk of falling into a value trap.

## 2. Workflow

The project is divided into five steps.

**Step 1: Filter the universe**

I first remove banks, securities companies, insurers, and funds because financial firms have accounting structures that differ from non-financial companies. Comparing P/B, debt/assets, or FCF yield across financial and non-financial firms can distort the results.

```python
financial_mask = (
    raw_fund['ticker'].isin(FINANCIAL_TICKER_EXCLUSIONS)
    | raw_fund['ticker'].str.startswith(FINANCIAL_PREFIX_EXCLUSIONS)
)
fund = raw_fund[~financial_mask].copy()
```

**Step 2: Construct valuation variables**

```python
fund['earnings_yield'] = np.where(fund['pe'] > 0, 1 / fund['pe'], np.nan)
fund['book_to_price'] = np.where(fund['pb'] > 0, 1 / fund['pb'], np.nan)
fund['ebitda_to_ev'] = np.where(fund['ev_ebitda'] > 0, 1 / fund['ev_ebitda'], np.nan)
```

**Step 3: Rank stocks by month**

Each variable is winsorized at the 1st and 99th percentiles, then ranked within each `signal_date`.

```python
panel[clean_column] = panel.groupby('signal_date')[column].transform(
    lambda s: s.clip(s.quantile(0.01), s.quantile(0.99))
)
panel[rank_column] = panel.groupby('signal_date')[clean_column].rank(pct=True)
```

**Step 4: Apply a reporting lag**

The notebook assumes that financial statement data become tradable with a four-month lag.

```python
panel['signal_date'] = (
    panel['period_end'] + pd.DateOffset(months=4)
).dt.to_period('M').dt.to_timestamp('M')
```

This conservative lag is used to reduce look-ahead bias when the actual publication dates of financial statements are not available.

**Step 5: Backtest the strategy**

Each month, the strategy selects the top 20% of stocks by signal and assigns equal weights. I compare the baseline `value_score` with the final `conditioned_value_score`, using a transaction cost assumption of 20 bps.

```python
data['rank_pct'] = data[signal].rank(pct=True)
top = data[data['rank_pct'] >= TOP_QUANTILE].copy()
weight.loc[dt, top['ticker']] = 1.0 / len(top)
```

## 3. Results

The backtest runs from 2013-01-31 to 2025-12-31. After filtering the universe, 359 stocks with price data remain in the sample.

| Signal | Annualized Return | Sharpe | Max Drawdown | Ending Value |
|---|---:|---:|---:|---:|
| `value_score` | 22.48% | 1.1494 | -43.08% | 13.96x |
| `conditioned_value_score` | 24.71% | 1.3204 | -38.90% | 17.64x |

In this sample, conditioned value performs better than pure value: it delivers a higher return, a higher Sharpe ratio, and a lower maximum drawdown.

![Equity growth](./value_research_rewritten_assets/equity_growth.png)

From the equity curve, the two strategies move closely in the earlier years. After 2020, conditioned value starts to pull ahead more clearly. This suggests that the quality-support layer may help the portfolio avoid some cheap but fundamentally weak stocks.

![Drawdown](./value_research_rewritten_assets/drawdown.png)
![Monthly Returns of Value strat](image-3.png)
![Monthly Returns of Conditioned strat](image-2.png)

That said, the drawdown remains large. Conditioned value reduces maximum drawdown from around -43.08% to -38.90%, but this is still a high level of risk. A real implementation would need additional controls for liquidity, position sizing, and market regime.

## 4. Conclusion

The main idea of this notebook is simple: buy cheap stocks, but do not buy them blindly. Adding filters for quality, growth, cash flow, and leverage improves the value strategy in this sample.
