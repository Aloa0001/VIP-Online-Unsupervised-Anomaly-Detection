import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
from config.features_config import SENSOR_COLS  # Load from YAML if needed

SENSOR_COLS = [
    "temperature_value",
    "humidity_value",
    "co2_value",
    "light_value",
    "radon_value",
    "airquality_value",
]
WINDOW_SIZE = 10


class StreamPreprocessor:
    """Handles scaling and windowing for streaming data."""

    def __init__(self):
        self.scalers_path = Path("models/scaler_per_room.pkl")
        with open(self.scalers_path, "rb") as f:
            self.scalers = pickle.load(f)

    def preprocess_stream(self, data_dict, resourceid):
        """Scale new data and build window (mock buffer for real streaming)."""
        scaler = self.scalers.get(resourceid)
        if scaler is None:
            raise ValueError(f"No scaler for {resourceid}")

        # Extract sensors
        sensors = np.array([data_dict[col] for col in SENSOR_COLS]).reshape(1, -1)
        scaled = scaler.transform(sensors)

        # Build window (in real: sliding buffer)
        window = np.tile(scaled, (WINDOW_SIZE, 1))  # Mock; use buffer in prod
        return torch.tensor(window.flatten(), dtype=torch.float32).unsqueeze(
            0
        )  # (1, 60)
