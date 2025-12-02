"""
Main training script for the LightGBM model.

Uses temporal split with absolute date threshold to ensure methodologically
correct validation without data leakage from future timestamps.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, constants
from .features import add_aggregate_features, handle_missing_values, add_time_features
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def train() -> None:
    """Runs the model training pipeline with temporal split.

    Loads prepared data from data/processed/, performs temporal split based on
    absolute date threshold, computes aggregate features on train split only,
    and trains a single LightGBM model. This ensures methodologically correct
    validation without data leakage from future timestamps.

    Note: Data must be prepared first using prepare_data.py
    """
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train and test sets
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Check for timestamp column
    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(
            f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train set. "
            "Make sure data was prepared with timestamp preserved."
        )

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Perform temporal split
    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    # Split data
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Verify temporal correctness
    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
    print(f"Max train timestamp: {max_train_timestamp}")
    print(f"Min validation timestamp: {min_val_timestamp}")

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError(
            f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
            f"is not greater than max train timestamp ({max_train_timestamp})."
        )
    print("✅ Temporal split validation passed: all validation timestamps are after train timestamps")

    # Compute aggregate features on train split only (to prevent data leakage)
    print("\nComputing aggregate features on train split only...")
    train_split_with_agg = add_aggregate_features(train_split.copy(), train_split)
    val_split_with_agg = add_aggregate_features(val_split.copy(), train_split)  # Use train_split for aggregates!
    

    train_split_with_agg = add_time_features(train_split_with_agg, train_split)
    val_split_with_agg = add_time_features(val_split_with_agg, train_split)


    # Handle missing values (use train_split for fill values)
    print("Handling missing values...")
    train_split_final = handle_missing_values(train_split_with_agg, train_split)
    val_split_final = handle_missing_values(val_split_with_agg, train_split)

    # Define features (X) and target (y)
    # Exclude timestamp, source, target, prediction columns
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    features = [col for col in train_split_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = train_split_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_train = train_split_final[features]
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features]
    y_val = val_split_final[config.TARGET]

    print(f"Training features: {len(features)}")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train single model
    print("\nTraining LightGBM model...")
    # model = lgb.LGBMRegressor(**config.LGB_PARAMS)

    models = []
    val_predictions = []
    seeds = [42, 123, 2025, 777, 999]  # магические сиды, которые всегда работают

    best_mae = 999.0
    best_pred = None

    for i, seed in enumerate(seeds):
        print(f"\nМодель {i+1}/5 | seed = {seed}")
        
        model = lgb.LGBMRegressor(
            objective="mae",
            metric="mae",
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=255,
            min_child_samples=30,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l1=0.1,
            lambda_l2=0.1,
            verbose=-1,
            n_jobs=-1,
            seed=seed,
            boosting_type="gbdt",
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                lgb.log_evaluation(0), 
            ],
        )

        # Предсказание + пост-процессинг (ОБЯЗАТЕЛЬНО!)
        raw_pred = model.predict(X_val)
        pred = np.clip(raw_pred, 1.0, 10.0)
        pred = np.round(pred * 2) / 2  # до 0.5

        mae = mean_absolute_error(y_val, pred)
        print(f"   → MAE = {mae:.5f}")

        models.append(model)
        val_predictions.append(pred)

        if mae < best_mae:
            best_mae = mae
            best_pred = pred

    # === УСРЕДНЕНИЕ ВСЕХ 5 МОДЕЛЕЙ ===
    final_val_pred = np.mean(val_predictions, axis=0)

    # Финальный пост-процессинг ансамбля
    final_val_pred = np.clip(final_val_pred, 1.0, 10.0)
    final_val_pred = np.round(final_val_pred * 2) / 2

    final_mae = mean_absolute_error(y_val, final_val_pred)
    final_rmse = np.sqrt(mean_squared_error(y_val, final_val_pred))
    final_score = 1 - (0.5 * final_rmse / 10 + 0.5 * final_mae / 10)

    print("\n" + "="*60)
    print("ФИНАЛЬНЫЙ АНСАМБЛЬ ГОТОВ!")
    print(f"   Лучшая одиночная модель:  MAE = {best_mae:.5f}")
    print(f"   Ансамбль из 5 моделей:    MAE = {final_mae:.5f}   ← ЭТО ТВОЙ РЕЗУЛЬТАТ!")
    print(f"   Оценка Public LB Score:   {final_score:.5f}")
    print("="*60)

    # === Сохраняем ВСЕ модели + финальные веса для predict.py ===
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Сохраняем каждую модель
    for i, model in enumerate(models):
        path = config.MODEL_DIR / f"lgb_model_seed_{seeds[i]}.txt"
        model.booster_.save_model(str(path))

    # Сохраняем финальный ансамбль-предикт на валидации (для дебага)
    np.save(config.MODEL_DIR / "final_val_pred.npy", final_val_pred)

    print(f"Сохранено 5 моделей + ансамбль в {config.MODEL_DIR}")
    print("ГОТОВО К САБМИТУ! ОТПРАВЛЯЙ С predict.py (он уже должен усреднять 5 моделей)")


if __name__ == "__main__":
    train()
