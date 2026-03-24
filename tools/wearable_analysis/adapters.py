"""
Device adapters — abstract base + WHOOP implementation + stubs for Oura/Garmin.

Each adapter reads device-specific raw data and returns a standardized DataFrame
following CORE_SCHEMA from config.py.
"""
from abc import ABC, abstractmethod
import pandas as pd


class DeviceAdapter(ABC):
    """Base class for wearable device data adapters."""

    @abstractmethod
    def ingest(self, data_dir: str, output_dir: str) -> pd.DataFrame:
        """Read raw device data and return standardized DataFrame."""
        ...

    @abstractmethod
    def get_device_name(self) -> str:
        """Return lowercase device identifier."""
        ...


class WHOOPAdapter(DeviceAdapter):
    """WHOOP adapter — delegates to existing ingest.py"""

    def get_device_name(self) -> str:
        return 'whoop'

    def ingest(self, data_dir: str, output_dir: str) -> pd.DataFrame:
        from .ingest import ingest_whoop, add_derived_features
        result = ingest_whoop(data_dir, output_dir)
        df = result[0] if isinstance(result, tuple) else result
        return add_derived_features(df)


class OuraAdapter(DeviceAdapter):
    """Oura Ring adapter — stub for future implementation.

    Oura data can be exported via:
    - Oura Cloud API (JSON)
    - Oura app CSV export
    - Google Takeout (if connected)

    Key differences from WHOOP:
    - Sleep staging uses different labels (deep/light/rem/awake)
    - HRV is measured as rMSSD during sleep (similar to WHOOP)
    - No strain score — use activity calories + steps instead
    - Readiness score ≈ Recovery score
    - Temperature deviation is a core metric
    """

    def get_device_name(self) -> str:
        return 'oura'

    def ingest(self, data_dir: str, output_dir: str) -> pd.DataFrame:
        raise NotImplementedError(
            "Oura adapter not yet implemented. "
            "Export your Oura data as CSV and use the GenericCSVAdapter instead. "
            "Column mapping: readiness_score -> recovery, "
            "rmssd -> hrv, lowest_heart_rate -> rhr, "
            "total_sleep_duration -> sleep_hours."
        )


class GarminAdapter(DeviceAdapter):
    """Garmin adapter — stub for future implementation.

    Garmin data sources:
    - Garmin Connect CSV export (Activities, Sleep, Steps, etc.)
    - FIT files (binary, need fitparse library)
    - Garmin Health API (requires developer access)

    Key differences from WHOOP:
    - Body Battery ≈ Recovery (0-100 scale)
    - Stress score is 0-100 (inverted from WHOOP's time-based)
    - Sleep staging: deep/light/rem/awake
    - VO2max is built-in (Firstbeat algorithm)
    - Training Status / Training Load are unique metrics
    """

    def get_device_name(self) -> str:
        return 'garmin'

    def ingest(self, data_dir: str, output_dir: str) -> pd.DataFrame:
        raise NotImplementedError(
            "Garmin adapter not yet implemented. "
            "Export your data from Garmin Connect as CSV and use GenericCSVAdapter."
        )


class GenericCSVAdapter(DeviceAdapter):
    """Generic CSV adapter for any wearable data.

    Reads all CSV files from data_dir, concatenates them,
    and uses the specified date column as the index.

    For best results, ensure your CSV columns match CORE_SCHEMA names
    (see config.py) or provide a column mapping in user_config.yaml.
    """

    def __init__(self, date_column: str = 'date'):
        self.date_column = date_column

    def get_device_name(self) -> str:
        return 'csv'

    def ingest(self, data_dir: str, output_dir: str) -> pd.DataFrame:
        import os
        import glob

        csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)

        # Try to parse and set date index
        if self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            df = df.set_index(self.date_column).sort_index()
            # Remove duplicate dates (keep last)
            df = df[~df.index.duplicated(keep='last')]

        return df


# Registry of available adapters
ADAPTERS = {
    'whoop': WHOOPAdapter,
    'oura': OuraAdapter,
    'garmin': GarminAdapter,
    'csv': GenericCSVAdapter,
}


def get_adapter(device: str) -> DeviceAdapter:
    """Get adapter instance by device name.

    Args:
        device: one of 'whoop', 'oura', 'garmin', 'csv'

    Returns:
        DeviceAdapter instance

    Raises:
        ValueError: if device is not supported
    """
    cls = ADAPTERS.get(device)
    if cls is None:
        raise ValueError(
            f"Unknown device: {device}. "
            f"Supported: {list(ADAPTERS.keys())}"
        )
    return cls()
