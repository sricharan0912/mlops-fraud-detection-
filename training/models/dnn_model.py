"""Optional TF/Keras tabular DNN challenger model.

Install tensorflow extra: pip install fraud-platform[tf]
"""

from __future__ import annotations

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


def build_dnn(input_dim: int, class_weight: dict | None = None):
    if not _TF_AVAILABLE:
        raise ImportError("tensorflow is not installed. Run: pip install fraud-platform[tf]")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="PR", name="avg_precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
        ],
    )
    return model
