# CGNIVEX: Mental Health Trend Analysis and Prediction Using NLP, Graph Neural Networks, and Continual Learning

> **DISCLAIMER**: CGNIVEX is a research-oriented data analytics and machine learning prototype. It provides a Mental Health Risk Indicator based on public community discussion trends. This system is **NOT** a medical tool and does **NOT** provide medical or clinical diagnoses.

---

## 📖 1. Abstract
The exponential growth of online social communities has made social platforms a vital medium for individuals to express emotional distress, stress, and anxiety. **CGNIVEX** (Continual Graph Neural Inductive Variation Explanation) is a college final-year Data Science & AI prototype system designed to analyze mental-health-related discussions from online communities. The system integrates DistilBERT contextual text representations, GraphSAGE heterogeneous graph neural networks, multi-task MLP prediction heads, temporal anomaly detection, feature attribution explainability, and incremental continual learning. An interactive Streamlit dashboard enables dynamic visualization of trends, node connections, and live inference.

---

## 🎯 2. Project Guide & Quick Start

### Installation & Environment Setup
```bash
# Navigate to project directory
cd d:/Data Science/CGNIVEX

# Install required dependencies
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
```bash
# Ingest data, preprocess, build graph, train models, and populate SQLite database
python main.py --mode sample
```

### Running Automated Unit Tests
```bash
pytest tests/
```

### Launching the Streamlit Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 3. Project Directory Structure
```
CGNIVEX/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_posts.csv
├── preprocessing/
│   ├── cleaner.py
│   ├── tokenizer.py
│   ├── language.py
│   └── privacy.py
├── scraper/
│   └── scraper.py
├── models/
│   ├── transformer_model.py
│   ├── gnn_model.py
│   ├── hybrid_model.py
│   └── baseline_models.py
├── graph/
│   ├── graph_builder.py
│   └── graph_visualizer.py
├── analytics/
│   ├── sentiment.py
│   ├── emotion.py
│   ├── topics.py
│   ├── trends.py
│   ├── anomaly.py
│   └── risk_score.py
├── explainability/
│   └── explainer.py
├── continual_learning/
│   └── incremental_update.py
├── database/
│   └── database.py
├── dashboard/
│   └── app.py
├── tests/
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 📑 4. Project Report & System Documentation

### Introduction
Mental health challenges among college students and young adults are increasingly expressed through online forums and communities. Timely population-level analytics are essential for identifying rising distress trends and providing targeted community support.

### Problem Statement
Traditional sentiment analysis tools evaluate posts in isolation, ignoring social network relationships, community topics, temporal evolution, and model drift over time.

### Existing System vs Proposed CGNIVEX System
- **Existing Systems**: Classical bag-of-words / TF-IDF models evaluating isolated posts. No graph structural context, no explainability, static offline training.
- **Proposed CGNIVEX System**: Combines DistilBERT text embeddings with GraphSAGE heterogeneous graph embeddings into a multi-task hybrid network with concept-drift detection and feature attribution explainer.

### Objectives
1. Build an intelligent ingestion layer with automated PII scrubbing (emails, phones, URLs) and SHA-256 user ID hashing.
2. Construct a heterogeneous contextual network graph (Users, Posts, Topics, Keywords, Emotions).
3. Implement a hybrid architecture merging NLP text representations and GraphSAGE node embeddings.
4. Provide research-oriented Mental Health Risk Indicators and feature attribution explanations.
5. Deploy a 11-page interactive Streamlit analytics dashboard.

### System Architecture Explanation
1. **Scraping Layer**: Ingests online posts using adaptive rate-limiting and mock dataset support.
2. **Preprocessing Engine**: Cleans text, normalizes emojis/slang, removes stopwords, lemmatizes words, and scrubs PII.
3. **Core Engine**: DistilBERT contextual text encoder + PyG GraphSAGE neighbor aggregator.
4. **Analytics Layer**: Multi-class emotion classifier, TF-IDF/LDA topic modeling, temporal trend aggregation, and Z-score volume anomaly detection.
5. **Dashboard**: 11 Streamlit pages with Plotly visualizations.

### Hardware & Software Requirements
- **Software**: Python 3.10+, PyTorch 2.0+, HuggingFace Transformers, PyTorch Geometric, NetworkX, Streamlit, Plotly, SQLite3.
- **Hardware**: Standard Intel i5/i7 or AMD Ryzen CPU, 8GB+ RAM (GPU optional due to fallback mode).

---

## ❓ 5. Viva Voce Questions & Detailed Answers

**Q1: What does CGNIVEX stand for and what is its core objective?**  
*Answer*: CGNIVEX stands for *Continual Graph Neural Inductive Variation Explanation*. Its core objective is to analyze mental-health-related online discussion trends by merging NLP text representations, Graph Neural Networks, and continual learning into an explainable analytics system.

**Q2: How does CGNIVEX protect user privacy?**  
*Answer*: The system strips emails, phone numbers, URLs, and `@mentions` using regular expressions in `preprocessing/privacy.py`. Furthermore, user IDs are converted to one-way SHA-256 cryptographic hashes (`usr_<hash>`), preventing personal identification.

**Q3: What nodes and edges make up the heterogeneous graph?**  
*Answer*: Nodes represent Users, Posts, Topics, Keywords, and Emotions. Edges represent User-authored-Post, Post-categorized-under-Topic, Post-expresses-Emotion, Post-contains-Keyword, and User-co-participates-with-User.

**Q4: Why combine Transformer embeddings with Graph Embeddings?**  
*Answer*: Transformers capture semantic and contextual meaning of post text, while GNNs capture structural context (user interaction patterns, shared community topics, and keyword co-occurrences). Combining both yields superior predictive performance.

**Q5: How does the GraphSAGE model aggregate neighbor information?**  
*Answer*: GraphSAGE samples local 2-hop node neighborhoods and aggregates their feature vectors using mean or pooling functions, producing node representations that reflect local graph topology.

**Q6: How is the Mental Health Risk Indicator computed?**  
*Answer*: The `RiskScoreCalculator` combines multi-factor signals including sentiment polarity, detected emotion, topic domain, and keyword intensity into a composite score (0 to 10), categorizing risk into Low, Moderate, or High.

**Q7: How does CGNIVEX handle missing dependencies or weak hardware?**  
*Answer*: CGNIVEX features automated fallback mechanisms: if PyTorch Geometric or GPU CUDA tools are missing, it falls back to NetworkX spectral adjacency embeddings and TF-IDF classical ML classifiers.

**Q8: What is concept drift and how does continual learning address it?**  
*Answer*: Concept drift occurs when post vocabulary or discussion trends change over time. CGNIVEX monitors statistical shifts in stress post ratios and performs incremental fine-tuning on new batches without retraining from scratch.

**Q9: Why does CGNIVEX explicitly state it is NOT a medical tool?**  
*Answer*: AI systems analyzing public social media posts lack clinical validation, diagnostic rigor, and medical context. CGNIVEX is designed strictly as a population-level research and trend analytics prototype.

**Q10: What anomaly detection method is used for trend spikes?**  
*Answer*: CGNIVEX uses Z-score deviation and rolling moving average on time-series post volume data in `analytics/anomaly.py` to identify statistical surges in mental health discussions.
