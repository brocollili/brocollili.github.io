---
layout: post
title: ""
date: 2026-06-29
categories: [quant, crypto, stat-arb]
tags: [pairs-trading, ornstein-uhlenbeck, validation, backtest]
math: true
---

# Value rẻ chưa đủ: lọc thêm chất lượng để tránh value trap

Khi nhìn một cổ phiếu có P/E thấp, P/B thấp hoặc EV/EBITDA thấp, phản ứng tự nhiên là nghĩ nó đang "rẻ". Nhưng rẻ không luôn đồng nghĩa với hấp dẫn. Một cổ phiếu có thể rẻ vì thị trường đang bỏ sót, nhưng cũng có thể rẻ vì doanh nghiệp thật sự yếu: lợi nhuận giảm, nợ cao, biên lợi nhuận xấu hoặc dòng tiền kém.

Vì vậy, trong notebook này tôi thử một ý tưởng đơn giản: thay vì chỉ mua cổ phiếu rẻ, tôi mua cổ phiếu rẻ **có điều kiện**. Điều kiện ở đây là doanh nghiệp phải có thêm một số tín hiệu hỗ trợ về chất lượng, tăng trưởng, dòng tiền và bảng cân đối.

## 1. Ý tưởng chính

Baseline là `value_score`, được tạo từ các thước đo định giá:

- earnings yield;
- book-to-price;
- EBITDA-to-EV.

Sau đó `support_score` được tạo thêm từ các nhóm cơ bản khác:

- `quality_score`: ROE, gross margin, net margin;
- `growth_score`: tăng trưởng lợi nhuận gộp, doanh thu, EBT, EPS;
- `cash_score`: FCF yield;
- `balance_score`: nợ/tài sản thấp thì điểm cao hơn.

Signal cuối cùng:

```python
conditioned_value_score = value_score * support_score
```

Lý do nhân hai điểm này với nhau là để cổ phiếu phải vừa rẻ, vừa có chất lượng cơ bản tương đối ổn. Nếu một mã rất rẻ nhưng support yếu, điểm cuối cùng sẽ bị kéo xuống. Đây là cách đơn giản để giảm rủi ro rơi vào value trap.

## 2. Các bước làm

Bài toán được chia thành 5 bước.

**Bước 1: Lọc universe**

Trước tiên loại ngân hàng, chứng khoán, bảo hiểm và các quỹ vì nhóm tài chính có cấu trúc kế toán khác doanh nghiệp phi tài chính. Nếu so P/B, debt/assets hoặc FCF yield giữa hai nhóm này thì kết quả dễ bị lệch.

```python
financial_mask = (
    raw_fund['ticker'].isin(FINANCIAL_TICKER_EXCLUSIONS)
    | raw_fund['ticker'].str.startswith(FINANCIAL_PREFIX_EXCLUSIONS)
)
fund = raw_fund[~financial_mask].copy()
```

**Bước 2: Tạo biến định giá**

Một điểm quan trọng là tôi không xem multiple âm là rẻ. Ví dụ P/E âm thường do EPS âm, nên đưa vào nhóm "cheap" sẽ sai về mặt kinh tế.

```python
fund['earnings_yield'] = np.where(fund['pe'] > 0, 1 / fund['pe'], np.nan)
fund['book_to_price'] = np.where(fund['pb'] > 0, 1 / fund['pb'], np.nan)
fund['ebitda_to_ev'] = np.where(fund['ev_ebitda'] > 0, 1 / fund['ev_ebitda'], np.nan)
```

**Bước 3: Rank theo từng tháng**

Các chỉ tiêu được winsorize ở 1% và 99%, rồi rank trong từng `signal_date`. Làm vậy để so sánh cổ phiếu trong cùng một thời điểm.

```python
panel[clean_column] = panel.groupby('signal_date')[column].transform(
    lambda s: s.clip(s.quantile(0.01), s.quantile(0.99))
)
panel[rank_column] = panel.groupby('signal_date')[clean_column].rank(pct=True)
```

**Bước 4: Lựa chọn lag của BCTC**

Bài giả định dữ liệu BCTC được sử dụng trễ 4 tháng

```python
panel['signal_date'] = (
    panel['period_end'] + pd.DateOffset(months=4)
).dt.to_period('M').dt.to_timestamp('M')
```

Giả định này nhắm tránh data leakage khi chưa có ngày công bố BCTC thật.

**Bước 5: Backtest**

Mỗi tháng, chiến lược chọn top 20% cổ phiếu theo signal và equal-weight. Tôi so sánh `value_score` với `conditioned_value_score`, phí giao dịch là 20 bps.

```python
data['rank_pct'] = data[signal].rank(pct=True)
top = data[data['rank_pct'] >= TOP_QUANTILE].copy()
weight.loc[dt, top['ticker']] = 1.0 / len(top)
```

## 3. Kết quả

Backtest chạy từ 2013-01-31 đến 2025-12-31, với 359 mã có dữ liệu giá sau khi lọc universe.

| Signal | Annualized Return | Sharpe | Max Drawdown | Ending Value |
|---|---:|---:|---:|---:|
| `value_score` | 22.48% | 1.1494 | -43.08% | 13.96x |
| `conditioned_value_score` | 24.71% | 1.3204 | -38.90% | 17.64x |

Trong mẫu dữ liệu này, conditioned value tốt hơn value thuần: return cao hơn, Sharpe cao hơn và drawdown thấp hơn.

![Equity growth](./value_research_rewritten_assets/equity_growth.png)

Nhìn vào equity curve, hai chiến lược đi khá sát nhau trong giai đoạn đầu. Sau 2020, conditioned value bắt đầu tách ra rõ hơn. Điều này gợi ý rằng lớp lọc chất lượng có thể giúp danh mục tránh một phần các cổ phiếu rẻ nhưng yếu.

![Drawdown](./value_research_rewritten_assets/drawdown.png)
![Monthly Returns of value](image.png)
![Montly Returns of conditioned-value](image-1.png)

Tuy vậy, drawdown vẫn rất lớn. Conditioned value giảm max drawdown từ khoảng -43.08% xuống -38.90%, nhưng đây vẫn là mức rủi ro cao. Nếu triển khai thật, cần thêm kiểm soát thanh khoản, size position và market regime.


## 4. Kết luận

Ý tưởng chính của notebook là: mua cổ phiếu rẻ, nhưng không mua rẻ một cách mù quáng. Khi thêm bộ lọc chất lượng, tăng trưởng, dòng tiền và nợ, chiến lược value cho kết quả tốt hơn.
 

