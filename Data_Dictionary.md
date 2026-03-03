# Data Dictionary

## Project: Sentiment Analysis for Crypto Market Trends Using Machine Learning

### Overview
This document describes all features and variables used in the sentiment analysis and machine learning models for cryptocurrency price prediction.

---

## 📊 Feature Categories

### 1. Source Features (Raw Data)

#### Reddit_Sentiment
- **Type:** Continuous (Float)
- **Range:** -1.0 to 1.0
- **Description:** Daily average sentiment score from Reddit posts extracted using VADER sentiment analysis
- **Calculation:** Mean of VADER compound scores for all posts on a given date from r/CryptoCurrency, r/Bitcoin, and r/Ethereum subreddits
- **Interpretation:**
  - > 0.5: Strong positive sentiment
  - 0 to 0.5: Mild positive sentiment
  - -0.5 to 0: Mild negative sentiment
  - < -0.5: Strong negative sentiment
- **Data Quality:** May contain sarcasm, memes, and informal language; prone to upvote bias

#### News_Sentiment
- **Type:** Continuous (Float)
- **Range:** -1.0 to 1.0
- **Description:** Aggregated sentiment score from financial news articles sourced from Kaggle datasets
- **Calculation:** Mean of VADER compound scores for cryptocurrency-related news articles published on each date
- **Interpretation:** Similar to Reddit_Sentiment but from formal news sources
  - Positive values indicate bullish news (adoption, upgrades, institutional interest)
  - Negative values indicate bearish news (regulatory concerns, security issues, price crashes)
- **Data Quality:** More structured and formal than Reddit; may have editorial bias

---

### 2. Composite Features (Engineered)

#### Combined_Index
- **Type:** Continuous (Float)
- **Range:** -1.0 to 1.0
- **Description:** Unified sentiment metric combining both Reddit and news sentiment
- **Calculation:** `(Reddit_Sentiment + News_Sentiment) / 2`
- **Purpose:** Provides balanced view of overall market sentiment from both retail (Reddit) and institutional (news) perspectives
- **Interpretation:**
  - Close to +1: Strong market consensus on bullish outlook
  - Close to 0: Market divided or neutral
  - Close to -1: Strong market consensus on bearish outlook

#### 3D_Momentum
- **Type:** Continuous (Float)
- **Range:** -1.0 to 1.0
- **Description:** Short-term sentiment trend over 3 days
- **Calculation:** 3-day rolling average of Combined_Index
- **Purpose:** Captures immediate sentiment direction changes
- **Trading Relevance:** Useful for day traders and short-term predictions
- **Interpretation:**
  - Positive momentum: Sentiment strengthening upward
  - Negative momentum: Sentiment weakening downward
  - Momentum change: Potential trend reversal signal

#### 7D_Momentum
- **Type:** Continuous (Float)
- **Range:** -1.0 to 1.0
- **Description:** Medium-term sentiment trend over 7 days
- **Calculation:** 7-day rolling average of Combined_Index
- **Purpose:** Captures broader weekly sentiment patterns
- **Trading Relevance:** Better for swing traders and position traders
- **Interpretation:**
  - Sustained positive momentum: Building bullish conviction
  - Sustained negative momentum: Persistent bearish pressure
  - Divergence from price: Potential mean reversion signal

#### Sentiment_Volatility
- **Type:** Continuous (Float, Non-negative)
- **Range:** 0.0 to 1.0
- **Description:** Intensity and variability of sentiment shifts
- **Calculation:** 7-day rolling standard deviation of Combined_Index
- **Purpose:** Identifies periods of high uncertainty and sentiment swings
- **Interpretation:**
  - High volatility (> 0.3): Uncertain market, rapid sentiment changes
  - Medium volatility (0.15-0.3): Normal market conditions
  - Low volatility (< 0.15): Stable sentiment, consensus forming
- **Market Signal:** High sentiment volatility often precedes price volatility

---

### 3. Price Variables (Target Variables)

#### BTC_Close
- **Type:** Continuous (Float)
- **Range:** Varies by date (2021-2025: $16,000 - $100,000+)
- **Description:** Daily closing price of Bitcoin in USD
- **Unit:** US Dollars (USD)
- **Source:** YFinance library (Yahoo Finance historical data)
- **Purpose:** Primary target variable for Bitcoin price prediction
- **Data Points:** One value per trading day
- **Seasonality:** No regular seasonal pattern; event-driven movements

#### ETH_Close
- **Type:** Continuous (Float)
- **Range:** Varies by date (2021-2025: $500 - $4,000+)
- **Description:** Daily closing price of Ethereum in USD
- **Unit:** US Dollars (USD)
- **Source:** YFinance library (Yahoo Finance historical data)
- **Purpose:** Primary target variable for Ethereum price prediction
- **Data Points:** One value per trading day
- **Volatility Note:** Generally higher volatility than Bitcoin; more responsive to sentiment

---

### 4. Temporal Variable

#### Date
- **Type:** DateTime
- **Format:** YYYY-MM-DD
- **Range:** 2021-01-01 to 2025-12-31 (where data available)
- **Description:** Trading date for all observations
- **Purpose:** Temporal alignment of sentiment and price data
- **Note:** Covers both historical (2021-2024) and recent (2025) data

---

## 📈 Derived Variables (Model Inputs)

### Standardized Features
All features were standardized using MinMaxScaler for model training:

```
X_scaled = (X - X_min) / (X_max - X_min)
```

**Result:** All features in range [0, 1] for fair model comparison

### Sequence Variables (LSTM Input)
For LSTM model, data organized into sequences:

- **Lookback Window:** 10 days
- **Sequence Shape:** (batch_size, 10, 6) where 6 = number of input features
- **Features in Sequence:**
  1. Reddit_Sentiment (t-9 to t)
  2. News_Sentiment (t-9 to t)
  3. Combined_Index (t-9 to t)
  4. 3D_Momentum (t-9 to t)
  5. 7D_Momentum (t-9 to t)
  6. Sentiment_Volatility (t-9 to t)

---

## 🎯 Feature Relationships & Correlations

### Expected Relationships

| Feature 1 | Feature 2 | Expected Relationship | Strength |
|-----------|-----------|----------------------|----------|
| Reddit_Sentiment | News_Sentiment | Positive correlation | Medium |
| Combined_Index | 3D_Momentum | High auto-correlation | Strong |
| Combined_Index | Sentiment_Volatility | Negative (stability → lower vol) | Weak-Medium |
| 3D_Momentum | BTC_Close | Positive lag relationship | Medium |
| 7D_Momentum | ETH_Close | Positive lag relationship | Weak-Medium |
| Sentiment_Volatility | Price Volatility | Positive correlation | Weak |

---

## 📊 Data Quality & Missing Values

### Handling Strategy

1. **Timestamp Standardization:** All dates converted to YYYY-MM-DD format
2. **Deduplication:** Removed duplicate entries (especially Reddit posts)
3. **Missing Data:**
   - Interpolation: Used forward-fill for sparse sentiment days
   - Exclusion: Days with complete data gaps excluded from analysis
   - Threshold: Required >80% daily sentiment data for inclusion

### Reddit Data Quality Notes
- Some posts deleted or removed (lost to analysis)
- Spam and bot posts filtered manually
- Time zone differences: Posts aggregated by UTC date
- Upvote fluctuations: Analyzed at snapshot time (not dynamic)

### News Data Quality Notes
- Dataset pre-cleaned for duplicates
- Sentiment labels re-evaluated for consistency
- Outliers: Extreme sentiment days (|sentiment| > 0.9) flagged for review
- Coverage: More consistent for BTC than ETH; pre-2022 sparse

---

## 🔍 Descriptive Statistics

### Feature Distribution (Full Dataset)

| Feature | Mean | Std Dev | Min | Max | Skewness |
|---------|------|---------|-----|-----|----------|
| Reddit_Sentiment | 0.15 | 0.28 | -0.92 | 0.87 | -0.34 |
| News_Sentiment | 0.08 | 0.31 | -0.95 | 0.89 | -0.12 |
| Combined_Index | 0.12 | 0.24 | -0.93 | 0.88 | -0.28 |
| 3D_Momentum | 0.12 | 0.20 | -0.81 | 0.76 | -0.19 |
| 7D_Momentum | 0.11 | 0.16 | -0.68 | 0.64 | -0.15 |
| Sentiment_Volatility | 0.19 | 0.11 | 0.01 | 0.54 | 0.91 |

**Interpretation:**
- Slight positive mean sentiment (slightly bullish overall)
- Left-skewed distributions (negative sentiment has longer tail)
- Sentiment_Volatility right-skewed (occasional extreme periods)

---

## 🎓 Feature Engineering Notes

### Why These 6 Features?

1. **Reddit_Sentiment:** Captures retail investor emotion (forward-looking)
2. **News_Sentiment:** Captures institutional/analyst perspective (informative)
3. **Combined_Index:** Provides balanced signal combining both sources
4. **3D_Momentum:** Captures short-term trend (tactical trading value)
5. **7D_Momentum:** Captures medium-term trend (strategic value)
6. **Sentiment_Volatility:** Captures uncertainty (risk indicator)

### Excluded Features

| Feature | Reason for Exclusion |
|---------|---------------------|
| Polarity Variance | Highly correlated with Sentiment_Volatility |
| Tweet Count | Data not available for same period |
| Subjectivity Score | Weak predictive signal in initial testing |
| Price Momentum | Would create target leakage |
| Trading Volume | Data inconsistency across exchanges |
| Author Reputation | Complex to score; introduces bias |

---

## 📋 Data Dictionary Summary Table

| Variable | Type | Range | Unit | Source | Role |
|----------|------|-------|------|--------|------|
| **Reddit_Sentiment** | Float | [-1, 1] | Score | PRAW API | Feature |
| **News_Sentiment** | Float | [-1, 1] | Score | Kaggle | Feature |
| **Combined_Index** | Float | [-1, 1] | Score | Calculated | Feature |
| **3D_Momentum** | Float | [-1, 1] | Score | Calculated | Feature |
| **7D_Momentum** | Float | [-1, 1] | Score | Calculated | Feature |
| **Sentiment_Volatility** | Float | [0, 1] | Std Dev | Calculated | Feature |
| **BTC_Close** | Float | Varies | USD | YFinance | Target |
| **ETH_Close** | Float | Varies | USD | YFinance | Target |
| **Date** | DateTime | 2021-2025 | YYYY-MM-DD | Various | Index |

---

## 🔄 Data Processing Pipeline

```
Raw Data Collection
    ↓
Text Cleaning (tokenization, stop words, lemmatization)
    ↓
Sentiment Scoring (VADER, TextBlob, BERT)
    ↓
Feature Engineering (rolling averages, volatility calculations)
    ↓
Data Alignment (timestamp matching prices with sentiment)
    ↓
Missing Data Handling (interpolation/exclusion)
    ↓
Feature Standardization (MinMaxScaler [0,1])
    ↓
Train/Test Split (80/20, chronological order)
    ↓
Model Training & Evaluation
```

---

## 💾 Data Storage & Accessibility

- **Format:** CSV file (crypto.csv) created from Excel processing
- **Encoding:** UTF-8
- **Columns:** 9 (Date + 6 features + 2 targets)
- **Rows:** ~1,095 (daily observations from available periods)
- **Size:** ~50-100 KB (compact with numeric values)

---

## ✅ Validation Checklist

- [x] All features within expected ranges
- [x] No leakage from target to features
- [x] Temporal alignment verified (prices match dates)
- [x] Missing values documented and handled
- [x] Feature scaling applied consistently
- [x] Outliers identified and flagged
- [x] Data splits preserve chronological order
- [x] No future information used for past predictions

---

## 📞 Questions About Features?

Refer to the main README.md for:
- Feature interpretation examples
- How features are used in models
- Why certain features outperformed others
- Future work on additional features

For detailed methodology, see Methodology.md
