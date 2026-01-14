# Market Regime Detection Dashboard

A real-time cryptocurrency market regime detection dashboard using unsupervised machine learning. Analyze BNB/FDUSD market conditions and detect 4 distinct trading regimes with live predictions.

## 🎯 What It Does

Automatically detects and classifies market behavior into 4 regimes:

- **🔴 High Volatility & Trending** - Strong momentum with elevated volatility
- **🟢 Low Volatility & Range-Bound** - Calm conditions oscillating in a band  
- **🟠 Mean-Reverting, Medium Volatility** - Swings back to mean with moderate noise
- **🔵 High Liquidity with Directional Pressure** - Deep book pushing one direction

### Key Features

✅ **Live Market Analysis** - Real-time BNB/FDUSD regime classification  
✅ **Regime Characteristics** - View volatility, returns, volume, and market dynamics  
✅ **Transition Probabilities** - Understand how markets move between regimes  
✅ **Model Comparison** - See why HDBSCAN outperforms K-Means and GMM  
✅ **Interactive Dashboard** - Beautiful, responsive UI with Tailwind CSS  
✅ **Production Ready** - Deployed on Vercel + Render with CORS enabled  

---

## 🏗️ Architecture

### Backend (FastAPI)
- **Language**: Python 3.9+
- **Framework**: FastAPI + Uvicorn
- **ML**: scikit-learn, HDBSCAN, GMM, Pandas
- **Data**: Binance API integration for live 1-minute candle data
- **Models**: Pre-trained scaler, PCA, HDBSCAN clustering

### Frontend (React + Vite)
- **Framework**: React 18 + Vite
- **Styling**: Tailwind CSS + PostCSS
- **Charts**: Recharts for data visualization
- **State**: React hooks + fetch API

---

## 🚀 Quick Start (Local)

### 1. Setup Backend (5 minutes)

```bash
# Navigate to project
cd market_regime_detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Generate fitted models
python save_fitted_models.py

# Start API server
python -m uvicorn api:app --reload
# Server runs on http://localhost:8000
```

### 2. Setup Frontend (5 minutes)

```bash
# Open new terminal
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
# Dashboard runs on http://localhost:5173
```

### 3. View Dashboard
- Open browser to `http://localhost:5173`
- See 4 market regimes with statistics
- Click "Check Current Market Regime" for live prediction
- Explore model comparison and transition matrix

---

## 📊 Dashboard Features

### Market Regimes Panel
Displays 4 regime cards with:
- **Samples**: Percentage of training data in regime
- **Volatility**: Market price volatility percentage
- **Returns**: Average returns in regime
- **Volume**: Relative trading volume

### Model Evaluation
Bar chart comparing clustering algorithms:
- K-Means (k=3, 5, 7)
- Gaussian Mixture Models (3, 5 components)
- HDBSCAN (mcs=5, 10, 15)

Winner: **HDBSCAN** with 0.512 silhouette score

### Regime Transitions
4×4 matrix showing probability of transitioning between regimes

### Live Market Regime
Real-time prediction using latest Binance candle data

---

## 🔌 API Endpoints

```bash
GET /regime-characteristics       # Get 4 regimes with stats
GET /transition-matrix             # Get regime transition probabilities
GET /model-evaluation              # Get model comparison results
GET /current-regime                # Get live market regime prediction
POST /predict-regime               # Predict regime from 45 features
```

---

## 📁 Project Structure

```
market_regime_detection/
├── api.py                         # FastAPI backend
├── save_fitted_models.py          # Generate fitted models
├── requirements.txt               # Dependencies
├── frontend/                      # React dashboard
│   ├── Dashboard.jsx
│   ├── App.jsx
│   ├── main.jsx
│   └── package.json
├── ml_service/
│   ├── models/fitted_*.pkl        # Pre-trained models
│   └── regime_detector.py
├── data_pipeline/
│   └── feature_engineering/       # Feature extraction
└── results/                       # Pre-computed analysis
    ├── regime_characteristics_kmeans_5.csv
    └── model_evaluation.csv
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI, Uvicorn, Gunicorn |
| **Frontend** | React 18, Vite, Tailwind CSS |
| **ML** | scikit-learn, HDBSCAN, Pandas, NumPy |
| **Data** | Binance API, 1-minute candles |
| **Deployment** | Render (backend), Vercel (frontend) |

---

## 🚀 Production Deployment

### Deploy Backend to Render.com
1. Connect GitHub repo
2. Build: `pip install -r requirements.txt && python save_fitted_models.py`
3. Start: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app`
4. Copy backend URL

### Deploy Frontend to Vercel.com
1. Import GitHub repo
2. Root directory: `frontend`
3. Environment: `VITE_API_URL=https://your-backend.onrender.com`
4. Deploy

---

## 🔍 Model Performance

| Model | Silhouette | Regimes |
|-------|-----------|---------|
| HDBSCAN (mcs=10) | **0.512** | 4 |
| K-Means (k=5) | 0.385 | 5 |
| GMM (5 comp) | 0.421 | 5 |

---

## 📝 Trading Strategies

| Regime | Strategy |
|--------|----------|
| High Volatility & Trending | Trend-Following |
| Low Volatility & Range-Bound | Range Trading |
| Mean-Reverting | Mean-Reversion |
| High Liquidity Directional | Liquidity-Taking |

---

**Built with FastAPI, React, and scikit-learn**

## Documentation

The complete documentation is available in the following formats:

- [Project Report (PDF)](./report.pdf) - Detailed analysis and methodology
- [Results Summary](./results/market_regime_analysis.md) - Summary of detected regimes and insights

## Features

- **Comprehensive Feature Engineering**: 
  - Order book metrics (imbalance, depth, spread)
  - Price-based features (volatility, momentum, z-scores)
  - Volume analysis
  - Multi-timeframe calculations
  - Advanced volatility estimators (Parkinson, directional)
  
- **Multiple Clustering Algorithms**:
  - K-Means
  - Gaussian Mixture Models (GMM)
  - HDBSCAN
  
- **Robust Evaluation Framework**:
  - Silhouette score
  - Davies-Bouldin index
  - Calinski-Harabasz index
  
- **Visualization Tools**:
  - 2D cluster visualization (UMAP, t-SNE, PCA)
  - Regime evolution over time
  - Cluster quality comparisons
  
- **Regime Analysis**:
  - Characteristic profiling
  - Transition probability analysis
  - Statistical summaries

## Project Structure

```
Directory structure:
└── jayXkush-market_regime_detection/
    ├── clustering_executor.py # Execution pipeline for clustering
    ├── regime_analyzer.py
    ├── run_regime_detection.py
    ├── data_pipeline/  # Data acquisition and processing
    │   ├── data_loader.py  # Loading raw market data
    │   └── feature_engineering/ # Feature extraction and normalization
    │       ├── feature_extractor.py  # Extract features from raw data
    │       ├── feature_normalization_validation.py
    │       └── normalizer.py    # Feature normalization and reduction
    ├── ml_service/ # Machine learning components
    │   ├── backtesting_engine.py
    │   ├── regime_detector.py # Main market regime detection service
    │   └── models/ # Clustering models
    │       ├── _init_.py
    │       ├── base_model.py
    │       ├── gmm.py  # Gaussian mixture model
    │       ├── hdbscan.py   # HDBSCAN clustering
    │       ├── kmeans.py # K-means clustering
    │       └── __pycache__/
    └── results/ # Output directory for results and visualizations
        ├── market_regime_analysis.md
        ├── model_evaluation.csv
        ├── regime_characteristics_gmm_3_diag.csv
        ├── regime_characteristics_gmm_3_full.csv
        ├── regime_characteristics_gmm_5_diag.csv
        ├── regime_characteristics_gmm_5_full.csv
        ├── regime_characteristics_hdbscan_10.csv
        ├── regime_characteristics_hdbscan_15.csv
        ├── regime_characteristics_hdbscan_5.csv
        ├── regime_characteristics_kmeans_3.csv
        ├── regime_characteristics_kmeans_5.csv
        ├── regime_characteristics_kmeans_7.csv
        ├── regime_report_hdbscan_10.json
        ├── synthetic_features.csv
        └── transition_matrix_hdbscan_10.csv
                     # Output directory for results and visualizations
```

## Installation

```bash
# Clone the repository
git clone https://github.com/jayXkush/market-regime-detection.git
cd market-regime-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from ml_service.regime_detector import MarketRegimeDetector
from data_pipeline.feature_engineering.feature_extractor import FeatureExtractor

# Extract features
extractor = FeatureExtractor(data_dir='path/to/data')
features = extractor.extract_multi_day_features(
    symbol='BNBFDUSD',
    start_date='20240101',
    end_date='20240131',
    resample_interval='5min',
    window_sizes=[5, 15, 30]
)

# Detect regimes
detector = MarketRegimeDetector(config={
    'normalization_method': 'standard',
    'pca_components': 20,
    'selected_model': 'kmeans'
})

detector.fit(features)
regime_labels = detector.predict(features)

# Get regime characteristics
regime_info = detector.analyze_regimes()
print(regime_info)

# Visualize regimes
detector.visualize_regimes(method='tsne')
```

### Advanced Usage with Clustering Executor

```python
from clustering_executor import ClusteringExecutor

# Initialize with preprocessed features
executor = ClusteringExecutor(features_df)

# Normalize features
normalized_features = executor.normalize_features(
    method='standard',
    pca_components=15
)

# Setup and run all clustering models
executor.setup_models()
labels = executor.run_clustering()

# Evaluate models
evaluation = executor.evaluate_models()
print(evaluation)

# Find best model and visualize
best_model = executor.get_best_model(metric='silhouette_score')
executor.visualize_clusters(method='umap', model_name=best_model)
```

## Example Results

The market regime detection system typically identifies several distinct market regimes, such as:

1. **Trending & High Volatility**: Strong directional movement with large price swings
2. **Mean-reverting & Low Liquidity**: Oscillating price action with thin order books
3. **High Liquidity & Low Volatility**: Stable price action with deep order books
4. **Transitional**: Brief periods between major regimes

These regimes can be visualized and analyzed to inform trading decisions and risk management.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- UMAP-learn
- HDBSCAN




