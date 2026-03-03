# Sentiment Analysis for Crypto Market Trends Using Machine Learning

**Masters Research Project | MS5131 Major Business Analytics**  
*J.E. Cairnes School of Business and Economics, University of Galway*

[![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

---

## 📋 Executive Summary

This research investigates how public sentiment extracted from **Reddit discussions** and **financial news articles** influences cryptocurrency price movements. Using **Natural Language Processing (NLP)** and **Machine Learning**, we developed a predictive framework analyzing sentiment data from 2021-2025 for Bitcoin (BTC) and Ethereum (ETH).

### 🎯 Main Finding
While sentiment alone cannot predict crypto prices accurately (R² ≈ 0.01), it provides **valuable behavioral signals** for identifying market trends and volatility patterns.

---

## 🔬 Research Objectives

✅ Examine sentiment's influence on cryptocurrency price trends (2021-2025)  
✅ Compare sentiment patterns with historical price movements  
✅ Evaluate predictive power of sentiment-based ML models  
✅ Identify which sentiment source (Reddit vs. News) is more reliable  
✅ Contribute to behavioral finance through NLP + predictive analytics  

---

## 📊 Key Results at a Glance

### Model Performance

| Model | Asset | RMSE | Accuracy | F1 Score |
|-------|-------|-------|----------|----------|
| **🏆 XGBoost** | **BTC** | 12,927 | 49% | 0.49 |
| **🏆 XGBoost** | **ETH** | 862 | **53%** | **0.54** |
| Random Forest | ETH | 844 | 52% | 0.51 |
| LSTM | ETH | 847 | 48% | 0.41 |

### Key Insights

✨ **XGBoost outperformed** all other models, especially for Ethereum  
✨ **Ethereum more sentiment-driven** than Bitcoin (53% vs 49% accuracy)  
✨ **Reddit sentiment leads price changes** by 1-3 days  
✨ **Ensemble methods beat deep learning** for noisy sentiment data  
⚠️ **Sentiment explains only ~1% of price variance** (R² = 0.0084)  
⚠️ **Bitcoin less responsive** to retail sentiment (influenced by macro factors)  

---

## 🛠️ Technical Approach

### Data Collection

```
📊 Data Sources (2021-2025)
├── Reddit: r/CryptoCurrency, r/Bitcoin, r/Ethereum (PRAW API)
├── News: Kaggle crypto news datasets
└── Prices: YFinance daily closing prices
```

**Timeline:** 2021-2024 news + 2022, 2025 Reddit + prices

### Sentiment Analysis Pipeline

Three complementary NLP models ensure robustness:

```python
# 1. VADER - Rule-based, optimized for social media
vader_sentiment = analyze_sentiment_vader(text)

# 2. TextBlob - Lexicon-based polarity/subjectivity
textblob_polarity = TextBlob(text).sentiment.polarity

# 3. BERT - Transformer-based, deep contextual understanding
bert_sentiment = fine_tuned_bert_model(text)

# Combined Index
combined_sentiment = (vader + textblob + bert) / 3
```

### Feature Engineering

**8 sentiment and market features:**

| Feature | Description | Calculation |
|---------|-------------|-------------|
| Reddit_Sentiment | Daily avg VADER score from Reddit | Mean of all daily posts |
| News_Sentiment | Aggregated financial news sentiment | Weighted average |
| Combined_Index | Unified sentiment score | (Reddit + News) / 2 |
| 3D_Momentum | Short-term sentiment trend | 3-day rolling average |
| 7D_Momentum | Medium-term sentiment trend | 7-day rolling average |
| Sentiment_Volatility | Sentiment intensity shifts | 7-day rolling std dev |
| BTC_Close | Bitcoin daily closing price | Target variable |
| ETH_Close | Ethereum daily closing price | Target variable |

### Machine Learning Models

**Ensemble Methods (Winners):**
```python
# Random Forest: 100 estimators
rf_model = RandomForestRegressor(n_estimators=100)

# XGBoost: Gradient boosting framework
xgb_model = XGBRegressor(n_estimators=100)  # BEST PERFORMER
```

**Deep Learning:**
```python
# LSTM Architecture
model = Sequential([
    LSTM(64, input_shape=(10, 6)),      # 10-day window, 6 features
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(2)  # BTC and ETH predictions
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50)
```

### Evaluation Metrics
- **RMSE:** Prediction error magnitude
- **MAE:** Average absolute deviation
- **R²:** Variance explained by model
- **Accuracy:** ±10% margin predictions
- **F1 Score:** Precision-recall balance

---

## 💡 Business Value

### For Traders & Investors
- **Complementary signals** alongside technical analysis
- **Early hype detection** before bubbles inflate
- **Volatility forecasting** from sentiment swings
- **Behavioral insights** into retail investor psychology

### For Exchanges & Platforms
- **Real-time sentiment dashboards** for engagement
- **Risk alerts** during extreme sentiment conditions
- **Fraud detection** (pump-and-dump scheme identification)
- **User experience improvements** based on sentiment trends

### For Analysts & Managers
- **Retail investor behavior** understanding
- **Portfolio optimization** incorporating sentiment
- **Risk modeling** with behavioral components
- **Product design** aligned with community sentiment

### For Regulators
- **Market surveillance** for manipulation
- **Early warning systems** for unusual activity
- **Systemic risk** monitoring
- **Regulation** informed by retail behavior analysis

---

## 🔍 Key Findings

### Why Sentiment Matters (But Isn't Enough)

**Supporting Evidence:**
- Positive Reddit sentiment spikes → +1-3 day price increases
- Negative news → sustained price declines
- Ethereum responds faster than Bitcoin to sentiment shifts
- Directional predictions work better than magnitude predictions

**The Limitation:**
- Sentiment explains only **1% of price variance**
- Major missing factors:
  - Regulatory announcements (SEC decisions, etc.)
  - Macroeconomic policy (Fed interest rates, etc.)
  - Large institutional transactions ("whale" movements)
  - Technical factors (order book, exchange flows)

### Ethereum vs Bitcoin

**Why Ethereum shows higher sentiment correlation:**
- **Larger retail investor base** (more influenced by community)
- **More cohesive community** (DeFi development discussions)
- **Lower institutional dominance** (compared to Bitcoin)
- **Direct development roadmap** (sentiment about upgrades matters)

**Why Bitcoin is more resistant to sentiment:**
- **Institutional adoption** (macro factors dominate)
- **Store-of-value narrative** (not tied to community enthusiasm)
- **Regulatory attention** (policy changes > online sentiment)
- **Macro sensitivity** (correlates with broader markets)

### Model Performance Analysis

**XGBoost Victory Reasons:**
✓ Handles non-linear sentiment-price relationships  
✓ Robust to noisy, unstructured text data  
✓ Built-in feature importance identification  
✓ Better than linear models for complex patterns  

**LSTM Underperformance Factors:**
✗ Shallow architecture (64 units) due to computational limits  
✗ Limited training data (only 2-3 year periods)  
✗ Requires more sequential complexity to shine  
✗ Overfitting risks with small datasets  

---

## 📈 Challenges & Limitations

### Data Quality Issues

**Reddit Limitations:**
- Informal language, slang, memes
- Sarcasm/irony cause sentiment misclassification
- Selection bias (Reddit users ≠ all crypto investors)
- Upvote system biases toward sensational posts

**News Limitations:**
- Editorial perspectives and bias
- Negativity bias (bad news gets more coverage)
- Sensationalism vs. balanced reporting
- Geographic coverage differences

**Temporal Gaps:**
- Irregular posting patterns
- Variable news publication frequencies
- Some days missing data (interpolated/excluded)
- 24/7 market vs. business hours news cycles

### Analysis Limitations

**NLP Constraints:**
- VADER lacks context and domain-specific knowledge
- No sarcasm/irony detection
- Cryptocurrency jargon misunderstandings
- Limited emoji interpretation

**Model Scope:**
- Only BTC and ETH (no altcoins)
- Simple LSTM due to computational limits
- Missing blockchain metrics
- No macroeconomic variables
- Excluded Twitter, Telegram, YouTube data

---

## 🚀 Future Research Directions

### 1. Advanced NLP Models
```
FinBERT → Financial domain-specific BERT
RoBERTa → More robust contextual embeddings
GPT-4   → Larger language model capabilities
```

### 2. Expanded Data Integration
- Multi-platform: Twitter, Telegram, Discord
- Blockchain metrics: Transaction volumes, wallet movements
- Order book data: Bid-ask spreads, order imbalances
- Macroeconomic indicators: Rates, inflation, VIX

### 3. Enhanced Machine Learning
- Hybrid models combining multiple data types
- Deeper LSTM architecture (256+ units)
- Attention mechanisms for important periods
- Explainable AI (SHAP, LIME) for transparency

### 4. Real-World Applications
- Live sentiment dashboard with alerts
- Automated trading signals
- Risk management tools
- Regulatory compliance monitoring

---

## 👥 Team

| Name |
|------|
| **Varun Ajjampur Govindaraju** 
| **Manish Pawar** 
| **Dhanaraj Kundapura Shankar** 


---

## 📚 Key References

1. **Valencia et al. (2019)** - Price movement prediction using sentiment + ML
2. **Kristoufek (2013)** - Bitcoin search volume → price causality
3. **Mai et al. (2018)** - Social media impact on Bitcoin value
4. **McNally et al. (2018)** - Machine learning for Bitcoin price prediction
5. **Sezer et al. (2020)** - Deep learning for financial time series
6. **Chen & Guestrin (2016)** - XGBoost: Scalable tree boosting
7. **Devlin et al. (2019)** - BERT: Language model pre-training

---

## 📁 Files in This Repository

```
crypto-sentiment-analysis/
├── README.md                  # Project overview (this file)
├── BIS_Final_Report.docx      # Complete masters thesis
├── LICENSE                    # MIT License
└── Documentation/
    ├── Data_Dictionary.md     # Feature definitions
    ├── Methodology.md         # Technical approach details
    └── Results_Summary.md     # Key findings & visualizations
```

---

## 🎓 Academic Integrity

> "In submitting this work, we confirm that it is entirely our own. We acknowledge that we may be invited to interview if there is any concern in relation to the integrity, and we are aware that any breach will be subject to the University's Procedures for dealing with plagiarism."

**Data Ethics:** All data sourced from public APIs and platforms in accordance with their terms of service and ethical research standards.

---

## 📖 How to Use This Repository

### For Hiring Managers:
1. Start with this README for overview
2. Skim Key Findings section
3. Review Results table
4. Open BIS_Final_Report.docx for complete analysis

### For Researchers:
1. Read Methodology section
2. Check Data Dictionary for feature definitions
3. Review References section
4. Examine BIS_Final_Report.docx for literature review & detailed results

### For Developers:
1. Study Feature Engineering section
2. Review model architectures (Random Forest, XGBoost, LSTM)
3. Examine evaluation metrics
4. See "Future Work" for extension ideas

---

## 🔗 Quick Links

- 📊 [Full Project Report](BIS_Final_Report.docx) - Complete thesis
- 📈 [Data Dictionary](Documentation/Data_Dictionary.md) - Feature explanations
- 🎯 [Methodology Details](Documentation/Methodology.md) - Technical approach
- ✨ [Key Findings](Documentation/Results_Summary.md) - Results & insights

---

## 📝 Citation

```bibtex
@mastersthesis{2025crypto,
  title={Sentiment Analysis for Crypto Market Trends Using Machine Learning},
  author={Pawar, Manish and Kundapura Shankar, Dhanaraj and Govindaraju, Varun A.},
  school={University of Galway, J.E. Cairnes School of Business and Economics},
  year={2025},
  course={MS5131 Major Business Analytics Project}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙋 Questions?

- **Project inquiries:** Contact team members via email
- **Research collaboration:** Open to discussing extensions
- **Issues or feedback:** Open an issue on GitHub

---

## ⭐ Recognition

This project was completed as part of the **MS5131 Major Business Analytics Project** at the University of Galway. We acknowledge the valuable feedback from our supervisors and the University's support throughout the research process.

---

**Last Updated:** March 2025  
**Status:** ✅ Completed  
**Language:** English  
**Institution:** University of Galway
