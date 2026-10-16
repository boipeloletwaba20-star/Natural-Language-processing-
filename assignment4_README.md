# NLP of Bitcoin Tweets: Topic Modelling and Sentiment vs. Volatility

**What does Bitcoin Twitter talk about, and does its mood predict how volatile Bitcoin will be tomorrow?**

*EDAB 6808 Assignment 4 · B.Com Business Analytics (Hons), University of the Free State · Pair project, 2026*

---

## Problem
- **Q1, topic modelling:** find the dominant themes in Bitcoin-related tweets and how they change over time.
- **Q2, sentiment and volatility:** score tweet sentiment, validate it by hand, and test whether lagged sentiment improves next-day Bitcoin volatility forecasts beyond market variables alone.

## Data
| Source | What | Size |
|---|---|---|
| [Bitcoin Tweets Dataset 2025–2026 (Kaggle)](https://www.kaggle.com/datasets/pokeash/bitcoin-tweets-dataset-20252026) | Tweet text and metadata | **4,689,305 tweets** (a default CSV reader wrongly reports 11.2M rows) |
| [Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD/) via `yfinance` | Daily BTC-USD open, high, low, close and volume | Matched to the tweet days |

The tweet data is downloaded with `kagglehub` and is too large to store here.

> **Data integrity note:** date-anchored topics (Tokyo Olympics, COVID vaccination, Bitcoin Pizza Day, El Salvador's Bitcoin law) show that the tweets are a **re-dated 2021–2022 corpus**, with every date exactly four years off. Q1 is unaffected. Q2 pairs those tweets with 2025–2026 prices, so its results demonstrate the pipeline rather than a finding about Bitcoin.

## Tools
Python · Google Colab · `polars` / `pyarrow` · `pandas` · `nltk` · `emoji` · `scikit-learn` (TF-IDF, NMF, LDA) · `gensim` (c_v coherence) · `sentence-transformers` · `BERTopic` · `vaderSentiment` · `transformers` (CryptoBERT) · `statsmodels` · `arch` · `xgboost` · `wordcloud` · `matplotlib` · `seaborn`

## Method

### Cleaning (shared)
1. Parsed with PyArrow to handle line breaks inside tweets and backslash escapes. Only 50 rows (0.001%) were unresolvable.
2. Two cleaned text columns: a lowercase, letters-only column for topics, and one that keeps case, punctuation and emoji for sentiment models.
3. URLs and mentions removed, hashtag words kept, tweets under four words or over 30% non-ASCII dropped.
4. Deduplication removed **35%** of tweets (1.65M), mostly bot price tickers hidden behind unique tracking links. Accounts with very low variety in their tweet openings were also removed (threshold 0.15).
5. A **199,992-tweet sample**, stratified by month, was used for topic modelling.

### Q1: Topic modelling
- **Three representations:** TF-IDF with bigrams (for NMF), raw counts (for LDA) and `all-MiniLM-L6-v2` embeddings (for BERTopic).
- **Choosing the number of topics:** NMF was fitted for k = 5 to 30 and scored by **c_v coherence**, which **peaked at k = 10 (0.862)**.
- **Comparison:** NMF (c_v 0.76–0.86) far outperformed LDA (≤ 0.56). BERTopic found 132 clusters and labelled 47.5% of tweets as outliers.

### Q2: Sentiment and volatility
- **Sentiment:** VADER (extended with 16 crypto terms) and **CryptoBERT** (polarity = P(bullish) − P(bearish)), scored on a day-stratified sample of 295,961 tweets.
- **Manual validation:** 60 tweets labelled by two raters (80% agreement, Cohen's κ = 0.686).
- **Volatility target:** the **Garman-Klass** estimator, which uses the daily open, high, low and close, is efficient, and does not overlap between consecutive days.
- **Design:** all predictors lagged by one day. The baseline uses HAR volatility terms, the lagged return, the high-low range and volume. The enhanced model adds five sentiment features.
- **Models:** OLS with HAC errors, GARCH(1,1), Random Forest and XGBoost, tested with an expanding-window, one-day-ahead backtest (59 forecasts).

## Key findings

### Q1: Ten topics (NMF, k = 10)
| Topic | Share | Leading keywords |
|---|---|---|
| Market speculation | 49.4% | time, market, price, going, people |
| Crypto news and infrastructure | 12.0% | ethereum, blockchain, cryptonews |
| Airdrop promotion | 8.4% | project, airdrop, bsc |
| NFTs | 8.1% | nft, opensea, metaverse |
| Altcoin tickers | 7.7% | eth, doge, ada, xrp, shib |
| BSC yield farming promotion | 5.6% | bnb, binance, defi, pancakeswap |
| Trading talk | 4.7% | buy, dip, sell, hodl |
| Lightning faucet referrals | 1.9% | lightning, sats, faucet |
| Publish0x referrals | 1.3% | earn, referral, publish |
| Coin Hunt World referrals | 0.9% | coinhuntworld, vault |

- **Five of the ten topics are promotional templates, not conversation** (18.1% of tweets).
- "Market speculation" holds 49.4% of tweets and acts as a catch-all residual topic.

### Q2: Sentiment does not improve volatility forecasts
| Model | MAE | RMSE | SMAPE | Direction correct |
|---|---|---|---|---|
| **OLS, market only** | **0.00986** | **0.01419** | **37.4%** | 51.7% |
| OLS + sentiment | 0.01020 | 0.01461 | 38.6% | 44.8% |
| Random Forest, market only | 0.01014 | 0.01532 | 38.8% | 48.3% |
| Random Forest + sentiment | 0.01015 | 0.01505 | 39.1% | 36.2% |
| XGBoost, market only | 0.01109 | 0.01657 | 41.8% | 48.3% |
| XGBoost + sentiment | 0.01036 | 0.01485 | 39.7% | 51.7% |

- In the regression, adding sentiment raised adjusted R² by only 0.0017 and worsened AIC and BIC. The joint F-test gave p = 0.373.
- **Market structure predicts volatility:** yesterday's return (p < 0.001) and high-low range (p = 0.007).
- Against the human labels, CryptoBERT scored **48.3%** and VADER **40.0%**. CryptoBERT was much better on bearish tweets (recall 0.72 versus 0.44).
- No model beats a coin flip at predicting the direction of volatility.

![Topic coherence by number of topics](images/coherence_by_k.png)

## How to run
1. Open the notebook in Google Colab with a **GPU runtime**: [Open in Colab](COLAB_LINK_HERE)
2. **Runtime → Run all.** The first cell installs the packages, and the dataset downloads with `kagglehub`, which needs a Kaggle account and API token configured in Colab.
3. Outputs are written to the working folder: `q1_topic_assignments.csv`, `q2_tweet_sentiment.csv`, `q2_daily_sentiment.csv`, `q2_modelling_table.csv` and `q2_backtest_results.csv`.
4. **To correct the dates**, subtract four years from the daily index before merging with prices.

## Repository structure
```
bitcoin-tweets-nlp/
├── README.md
├── bitcoin_tweets_nlp.ipynb
├── outputs/
│   ├── q1_topic_assignments.csv      (if under 100 MB, otherwise link to Drive)
│   ├── q2_daily_sentiment.csv
│   └── q2_backtest_results.csv
├── validation/
│   └── validation_labelled.csv
└── images/
    ├── coherence_by_k.png
    ├── topic_share_over_time.png
    ├── wordcloud.png
    └── sentiment_vs_volatility.png
```

## Report and presentation
- [Full report](REPORT_LINK_HERE) (view-only)
- [Presentation slides](SLIDES_LINK_HERE) (view-only)
- [Portfolio page](EPORTFOLIO_LINK_HERE)

## References
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics, 7*(2), 174–196.
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *The Journal of Business, 53*(1), 67–78.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of ICWSM-14*.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics, 33*(1), 159–174.
