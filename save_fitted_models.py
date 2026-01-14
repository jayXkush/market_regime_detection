

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.feature_engineering.feature_extractor import FeatureExtractor
from clustering_executor import ClusteringExecutor


def save_models():
    
    print("=" * 80)
    print("SAVING FITTED MODELS FOR DEPLOYMENT")
    print("=" * 80)
    
    # Step 1: Load and extract features
    print("\n1. Loading market data and extracting features...")
    symbol = 'BNBFDUSD'
    dates = ['20250314', '20250315', '20250316', '20250317']
    extractor = FeatureExtractor(data_dir='data')
    all_features = []

    for date in dates:
        print(f"   - Processing {symbol} on {date}...")
        features = extractor.extract_features(symbol=symbol, date_str=date)
        if features.empty:
            print(f"     ! No data found for {date}, skipping")
            continue

        # Use timestamp as index; keep numeric columns for modeling
        features = features.set_index('timestamp')
        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            print(f"     ! No numeric features for {date}, skipping")
            continue

        all_features.append(numeric_features)

    if not all_features:
        raise ValueError("No features extracted; check data files and paths")

    features_df = pd.concat(all_features, axis=0).sort_index()
    print(f"   ✓ Extracted {len(features_df)} samples with {len(features_df.columns)} features")
    
    # Step 2: Run clustering pipeline
    print("\n2. Running clustering pipeline...")
    executor = ClusteringExecutor(features_df)
    executor.normalize_features(method='standard', pca_components=20)
    executor.setup_models()
    executor.run_clustering()
    executor.evaluate_models()

    # Create output directory for models
    model_dir = Path(__file__).parent / 'ml_service' / 'models'
    model_dir.mkdir(exist_ok=True)

    # Step 3: Save normalized scaler
    print("\n3. Saving StandardScaler...")
    scaler_path = model_dir / 'fitted_scaler.pkl'
    joblib.dump(executor.normalizer.scaler, scaler_path)
    print(f"   ✓ Saved: {scaler_path}")

    # Step 4: Save PCA model
    print("\n4. Saving PCA model...")
    pca_path = model_dir / 'fitted_pca.pkl'
    joblib.dump(executor.normalizer.pca, pca_path)
    print(f"   ✓ Saved: {pca_path}")

    # Step 5: Run clustering and save best model
    print("\n5. Running clustering with best model (HDBSCAN mcs=10)...")
    best_model = executor.models.get('hdbscan_10')
    if best_model is None:
        print("   ! Warning: HDBSCAN model not found, using available model")
        best_model = list(executor.models.values())[0]

    hdbscan_path = model_dir / 'fitted_hdbscan_best.pkl'
    joblib.dump(best_model.model, hdbscan_path)
    print(f"   ✓ Saved: {hdbscan_path}")
    
    # Step 6: Verify saved files
    print("\n6. Verifying saved models...")
    for path in [scaler_path, pca_path, hdbscan_path]:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {path.name}: {size_mb:.2f} MB")
        else:
            print(f"   ✗ {path.name}: NOT FOUND")
    
    print("\n" + "=" * 80)
    print("MODELS SAVED SUCCESSFULLY")
    print("=" * 80)
    print("\nDeployment models ready at:")
    print(f"  - Scaler: {scaler_path}")
    print(f"  - PCA: {pca_path}")
    print(f"  - HDBSCAN: {hdbscan_path}")
    print("\nNext step: Run 'python api.py' to start the API server")


if __name__ == '__main__':
    save_models()
