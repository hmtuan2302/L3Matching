from preprocess import Preprocessor
import polars as pl
import lightgbm as lgb
import numpy as np
import joblib
from loguru import logger

class LightGBMModel:
    def __init__(self, df_train, df_val, df_test):
        self.train_data = df_train
        self.val_data = df_val
        self.test_data = df_test
        self.model = None

    def train(self, feature_column: str = "concat_embed",
              target_column: str = "NTP_to_FA"):
        """Train the LightGBM model.
        Use the feature_column as input features to train and predict target_column.
        """
        # 1. Prepare data for LightGBM
        logger.info("Preparing data for LightGBM training...")
        X_train = np.array(self.train_data[feature_column].to_list())
        y_train = self.train_data[target_column].to_numpy()
        
        X_val = np.array(self.val_data[feature_column].to_list())
        y_val = self.val_data[target_column].to_numpy()

        # Create LightGBM dataset objects
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        # 2. Define model parameters with regularization
        # These parameters are a starting point. They should be tuned using techniques like cross-validation.
        params = {
            'objective': 'regression_l2',  # MAE loss, often robust to outliers
            'metric': 'mae',              # Mean Absolute Error, common for regression
            'n_estimators': 5000,          # High number, will be stopped by early stopping
            'learning_rate': 0.01,         # Small learning rate
            'feature_fraction': 0.5,       # Select 50% of features for each tree
            'bagging_fraction': 0.9,       # Select 90% of data for each tree
            'lambda_l1': 0.1,              # L1 regularization
            'lambda_l2': 0.1,              # L2 regularization
            'num_leaves': 31,              # Default value, good starting point
            'verbose': -1,                 # Suppress verbose output
            'n_jobs': 4,                  # Use all available cores
            'seed': 42,
            'boosting_type': 'gbdt',
        }

        # 3. Train the model with early stopping
        logger.info("Starting model training...")
        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(period=100)]
        )
        
        logger.info("Model training finished.")
        logger.info(f"Best iteration: {self.model.best_iteration}")
        
        # Save model to ../models/ with joblib
        joblib.dump(self.model, f"../models/lgbm_{feature_column}_{target_column}.pkl")
        logger.info(f"Model saved to ../models/lgbm_{feature_column}_{target_column}.pkl")

        return self.model

    def test(self, feature_column: str = "concat_embed",
             target_column: str = "NTP_to_FA"):
        """Test the trained model on the test dataset and print MAE metrics."""
        if self.model is None:
            logger.error("Model is not trained yet. Please run train() first.")
            return

        logger.info("Starting testing...")
        
        # 1. Prepare test data
        X_test = np.array(self.test_data[feature_column].to_list())
        y_test = self.test_data[target_column].to_numpy()

        # 2. Predict on test data
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)

        # 3. Calculate MAE and its standard deviation
        absolute_errors = np.abs(y_test - y_pred)
        mean_mae = np.mean(absolute_errors)
        std_mae = np.std(absolute_errors)

        logger.info(f"Testing finished.")
        logger.info(f"Mean Absolute Error (MAE) on test set: {mean_mae:.4f}")
        logger.info(f"Standard Deviation of MAE on test set: {std_mae:.4f}")

    def inference(self, feature_column: str = "concat_embed",
                  target_column: str = "NTP_to_FA",
                  model_path: str = None):
        """Load a trained model and perform inference on the test dataset."""
        if model_path is None:
            model_path = f"../models/lgbm_{feature_column}_{target_column}.pkl"
        
        # Load the model
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        logger.info("Starting inference...")
        
        # 1. Prepare test data
        X_test = np.array(self.test_data[feature_column].to_list())
        y_test = self.test_data[target_column].to_numpy()

        # 2. Predict on test data
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)

        # 3. Calculate MAE and its standard deviation
        absolute_errors = np.abs(y_test - y_pred)
        mean_mae = np.mean(absolute_errors)
        std_mae = np.std(absolute_errors)

        logger.info(f"Inference finished.")
        logger.info(f"Mean Absolute Error (MAE) on test set: {mean_mae:.4f}")
        logger.info(f"Standard Deviation of MAE on test set: {std_mae:.4f}")

        return y_pred


if __name__ == "__main__":
    # Load and preprocess data
    # preprocessor_train = Preprocessor(["../data/Grati_MDL_train.xlsx"],
    #                         [pl.datetime(2016, 12, 27)])
    # preprocessor_train.load_data()
    # df_train = preprocessor_train.preprocess()

    # preprocessor_val = Preprocessor(["../data/Grati_MDL_val.xlsx"],
    #                             [pl.datetime(2016, 12, 27)])
    # preprocessor_val.load_data()
    # df_val = preprocessor_val.preprocess()

    preprocessor_test = Preprocessor(["../data/Grati_MDL_test.xlsx"],
                                [pl.datetime(2016, 12, 27)])
    preprocessor_test.load_data()
    df_test = preprocessor_test.preprocess()

    # print("Sample data train:")
    # print(df_train.head())
    
    # # Instantiate the model with the loaded data
    # lgbm = LightGBMModel(df_train, df_val, df_test)
    
    # feature_columns = ["concat_embed"]
    # target_columns = ["NTP_to_FA", "FA_to_FC"]
    # for target_col in target_columns:
    #     for feature_col in feature_columns:
    #         logger.info(f"Training and testing with feature: {feature_col}, target: {target_col}")
    #         lgbm.train(feature_column=feature_col, target_column=target_col)
    #         lgbm.test(feature_column=feature_col, target_column=target_col)

    # Inference example
    lgbm_inference = LightGBMModel(None, None, df_test)
    feature_col = "concat_embed"
    target_columns = ["NTP_to_FA", "FA_to_FC"]
    for target_col in target_columns:
        logger.info(f"Inference with feature: {feature_col}, target: {target_col}")
        lgbm_inference.inference(feature_column=feature_col, target_column=target_col)

