"""
LSTM Model for Stock Price Prediction
Supports multi-feature input with technical indicators
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockLSTMModel:
    """
    Bidirectional LSTM model for stock price prediction.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 1,
        units: list = [128, 64],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.feature_columns = []

    def build_model(self) -> Sequential:
        """Build the Bidirectional LSTM architecture."""
        model = Sequential(name="StockPredictor_LSTM")

        # First Bidirectional LSTM layer
        model.add(
            Bidirectional(
                LSTM(
                    self.units[0],
                    return_sequences=True,
                    input_shape=(self.sequence_length, self.n_features),
                ),
                name="bi_lstm_1",
            )
        )
        model.add(Dropout(self.dropout_rate, name="dropout_1"))
        model.add(BatchNormalization(name="batch_norm_1"))

        # Second Bidirectional LSTM layer
        model.add(
            Bidirectional(
                LSTM(self.units[1], return_sequences=False),
                name="bi_lstm_2",
            )
        )
        model.add(Dropout(self.dropout_rate, name="dropout_2"))
        model.add(BatchNormalization(name="batch_norm_2"))

        # Dense layers
        model.add(Dense(32, activation="relu", name="dense_1"))
        model.add(Dropout(0.1, name="dropout_3"))
        model.add(Dense(1, activation="linear", name="output"))

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="huber",
            metrics=["mae"]
        )

        # Explicitly build model before any parameter access
        model.build(
            input_shape=(None, self.sequence_length, self.n_features)
        )

        self.model = model
        logger.info("Model built successfully")

        return model

    def prepare_sequences(
        self, data: np.ndarray, target_col_idx: int = 0
    ) -> tuple:
        """Create sliding window sequences for LSTM input."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length: i])
            y.append(data[i, target_col_idx])

        return np.array(X), np.array(y)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = "Close",
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.15,
        model_dir: str = "saved_models",
    ) -> dict:
        """Train the model."""
        self.feature_columns = feature_cols
        self.n_features = len(feature_cols)

        # Scale features
        scaled_features = self.feature_scaler.fit_transform(df[feature_cols])
        target_idx = feature_cols.index(target_col)

        # Create sequences
        X, y = self.prepare_sequences(
            scaled_features,
            target_col_idx=target_idx
        )

        logger.info(f"Training sequences: X={X.shape}, y={y.shape}")

        # Build model
        self.build_model()

        # Callbacks
        os.makedirs(model_dir, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=os.path.join(
                    model_dir,
                    "best_model.keras"
                ),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        self.history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=False,
        )

        return self.history.history

    def predict_next_day(self, df: pd.DataFrame) -> dict:
        """Predict next day's closing price."""
        if self.model is None:
            raise RuntimeError(
                "Model not trained. Call fit() first."
            )

        recent = df[self.feature_columns].tail(
            self.sequence_length
        )

        if len(recent) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} rows"
            )

        scaled = self.feature_scaler.transform(recent)

        X = scaled.reshape(
            1,
            self.sequence_length,
            self.n_features
        )

        pred_scaled = self.model.predict(
            X,
            verbose=0
        )[0, 0]

        close_idx = self.feature_columns.index("Close")

        dummy = np.zeros((1, self.n_features))
        dummy[0, close_idx] = pred_scaled

        pred_price = self.feature_scaler.inverse_transform(
            dummy
        )[0, close_idx]

        # Monte Carlo Dropout for uncertainty
        mc_predictions = []

        for _ in range(50):
            p = self.model(
                X,
                training=True
            ).numpy()[0, 0]

            dummy2 = np.zeros((1, self.n_features))
            dummy2[0, close_idx] = p

            mc_predictions.append(
                self.feature_scaler.inverse_transform(
                    dummy2
                )[0, close_idx]
            )

        mc_predictions = np.array(mc_predictions)

        lower = np.percentile(mc_predictions, 5)
        upper = np.percentile(mc_predictions, 95)

        last_close = df["Close"].iloc[-1]
        change_pct = (
            (pred_price - last_close) / last_close
        ) * 100

        return {
            "predicted_price": float(pred_price),
            "last_close": float(last_close),
            "change_pct": float(change_pct),
            "direction": "UP" if change_pct > 0 else "DOWN",
            "confidence_lower": float(lower),
            "confidence_upper": float(upper),
            "confidence_range": float(upper - lower),
        }

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate model performance."""
        feature_cols = self.feature_columns

        scaled = self.feature_scaler.transform(
            df[feature_cols]
        )

        X, y_true_scaled = self.prepare_sequences(
            scaled,
            target_col_idx=feature_cols.index("Close")
        )

        y_pred_scaled = self.model.predict(
            X,
            verbose=0
        ).flatten()

        close_idx = feature_cols.index("Close")
        n = self.n_features

        def inv(vals):
            dummy = np.zeros((len(vals), n))
            dummy[:, close_idx] = vals
            return self.feature_scaler.inverse_transform(
                dummy
            )[:, close_idx]

        y_true = inv(y_true_scaled)
        y_pred = inv(y_pred_scaled)

        return {
            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred
                    )
                )
            ),
            "mae": float(
                mean_absolute_error(
                    y_true,
                    y_pred
                )
            ),
            "r2": float(
                r2_score(
                    y_true,
                    y_pred
                )
            ),
            "mape": float(
                np.mean(
                    np.abs(
                        (y_true - y_pred) /
                        (y_true + 1e-8)
                    )
                ) * 100
            ),
            "directional_accuracy": float(
                np.mean(
                    np.sign(np.diff(y_true)) ==
                    np.sign(np.diff(y_pred))
                ) * 100
            ),
        }

    def save(self, path: str):
        """Save model and scaler."""
        os.makedirs(path, exist_ok=True)

        self.model.save(
            os.path.join(
                path,
                "lstm_model.keras"
            )
        )

        joblib.dump(
            self.feature_scaler,
            os.path.join(
                path,
                "feature_scaler.pkl"
            )
        )

        metadata = {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "feature_columns": self.feature_columns,
            "units": self.units,
            "dropout_rate": self.dropout_rate,
        }

        pd.Series(metadata).to_json(
            os.path.join(
                path,
                "metadata.json"
            )
        )

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load saved model."""
        import json

        with open(
            os.path.join(
                path,
                "metadata.json"
            )
        ) as f:
            meta = json.load(f)

        obj = cls(
            sequence_length=meta["sequence_length"],
            n_features=meta["n_features"],
            units=meta["units"],
            dropout_rate=meta["dropout_rate"],
        )

        obj.feature_columns = meta["feature_columns"]

        obj.model = load_model(
            os.path.join(
                path,
                "lstm_model.keras"
            )
        )

        obj.feature_scaler = joblib.load(
            os.path.join(
                path,
                "feature_scaler.pkl"
            )
        )

        logger.info(f"Model loaded from {path}")

        return obj
