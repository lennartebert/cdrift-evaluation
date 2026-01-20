"""Complexity-based drift detection using PELT segmentation."""

import hashlib
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import ruptures as rpt
from pm4py.objects.log.obj import EventLog
from pm4py.util import xes_constants as xes

# Add project root to sys.path to access utils module
# This file is at: plugins/cdrift_evaluation/cdrift/approaches/complexity_drift_detection/complexity_drift_detection.py
# Project root is 5 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.complexity.assessors import assess_complexity_via_fixed_sized_windows

# Cache directory for complexity scores
CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "complexity_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _generate_log_hash(log: EventLog, log_id: Optional[str] = None) -> str:
    """Generate a hash identifier for a log.
    
    Parameters
    ----------
    log : EventLog
        The event log.
    log_id : str, optional
        Optional explicit log identifier. If provided, this is used instead
        of generating a hash.
    
    Returns
    -------
    str
        Hash identifier for the log.
    """
    if log_id:
        return hashlib.md5(log_id.encode()).hexdigest()
    
    # Generate hash from log characteristics
    # Use: log length, first trace activities, last trace activities
    hash_input = f"{len(log)}"
    
    if len(log) > 0:
        # First trace activities
        first_trace = log[0]
        first_activities = tuple(ev.get(xes.DEFAULT_NAME_KEY, "") for ev in first_trace)
        hash_input += str(first_activities)
        
        # Last trace activities
        last_trace = log[-1]
        last_activities = tuple(ev.get(xes.DEFAULT_NAME_KEY, "") for ev in last_trace)
        hash_input += str(last_activities)
    
    return hashlib.md5(hash_input.encode()).hexdigest()


def _get_cache_path(log_hash: str, window_size: int, first_window_start: int) -> Path:
    """Get the cache file path for given parameters.
    
    Parameters
    ----------
    log_hash : str
        Hash identifier for the log.
    window_size : int
        Window size.
    first_window_start : int
        First window start index.
    
    Returns
    -------
    Path
        Path to the cache file.
    """
    cache_filename = f"{log_hash}_w{window_size}_s{first_window_start}.pkl"
    return CACHE_DIR / cache_filename


def detect_change(
    log: EventLog,
    window_size: int,
    penalty: float,
    complexity_metric: str,
    min_consecutive_windows: int = 2,
    first_window_start: int = 0,
    use_cache: bool = True,
    log_id: Optional[str] = None,
) -> List[int]:
    """Detect change points using PELT segmentation on complexity metrics.

    Parameters
    ----------
    log : EventLog
        The event log to analyze.
    window_size : int
        Size of non-overlapping windows.
    penalty : float
        Penalty parameter for PELT algorithm.
    complexity_metric : str
        Name of the complexity metric to use (e.g., "Avg. Trace Length",
        "Normalized Sequence Entropy").
    min_consecutive_windows : int, optional
        Minimum number of consecutive windows for PELT segmentation.
        Default is 2.
    first_window_start : int, optional
        Starting index for the first window. Default is 0.
    use_cache : bool, optional
        Whether to use cached complexity scores if available. Default is True.
    log_id : str, optional
        Optional explicit log identifier for caching. If not provided, a hash
        is generated from log characteristics.

    Returns
    -------
    List[int]
        List of detected change point trace indices.
    """
    # Check if log is too short
    if len(log) < window_size + first_window_start:
        return []

    # Generate log hash for caching
    log_hash = _generate_log_hash(log, log_id)
    cache_path = _get_cache_path(log_hash, window_size, first_window_start)
    
    # Try to load from cache
    df = None
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                # Verify cache matches expected parameters
                if (cached_data.get('window_size') == window_size and
                    cached_data.get('first_window_start') == first_window_start and
                    cached_data.get('log_hash') == log_hash):
                    df = cached_data['df']
        except Exception as e:
            # If cache loading fails, recalculate
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
            df = None
    
    # Calculate complexity scores if not loaded from cache
    if df is None:
        # Slice log if first_window_start > 0
        if first_window_start > 0:
            log_sliced = EventLog(log[first_window_start:])
        else:
            log_sliced = log
        
        # Use existing complexity assessment infrastructure
        # Use dummy values for dataset_key, configuration_name, approach_name
        # since we're not saving results to disk
        # Use both adapters to get all available metrics:
        # - "local" provides basic metrics like Avg. Trace Length
        # - "vidgof_sample" provides entropy metrics like Normalized Sequence Entropy
        df = assess_complexity_via_fixed_sized_windows(
            pm4py_log=log_sliced,
            window_size=window_size,
            offset=window_size,  # Non-overlapping windows
            dataset_key="temp",
            configuration_name="temp",
            approach_name="temp",
            adapter_names=["local", "vidgof_sample"],
            add_prefix=True,
            include_adapter_name=False,
        )
        
        # Adjust first_index and last_index if we sliced the log
        if first_window_start > 0:
            df['first_index'] = df['first_index'] + first_window_start
            df['last_index'] = df['last_index'] + first_window_start
        
        # Save to cache if caching is enabled
        if use_cache:
            try:
                cache_data = {
                    'df': df,
                    'window_size': window_size,
                    'first_window_start': first_window_start,
                    'log_hash': log_hash,
                }
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            except Exception as e:
                print(f"Warning: Failed to save cache to {cache_path}: {e}")

    if len(df) == 0:
        return []

    # Extract the specified complexity metric
    # Metric columns have "measure_" prefix
    metric_column = f"measure_{complexity_metric}"
    if metric_column not in df.columns:
        # Try alternative column name formats
        # Some metrics might have different naming
        possible_columns = [
            metric_column,
            f"measure_{complexity_metric.lower()}",
            f"measure_{complexity_metric.replace(' ', '_')}",
        ]
        metric_column = None
        for col in possible_columns:
            if col in df.columns:
                metric_column = col
                break

        if metric_column is None:
            # Try to find column that contains the metric name
            for col in df.columns:
                if col.startswith("measure_") and complexity_metric.lower() in col.lower():
                    metric_column = col
                    break

        if metric_column is None:
            raise ValueError(
                f"Complexity metric '{complexity_metric}' not found in DataFrame. "
                f"Available measure columns: {[c for c in df.columns if c.startswith('measure_')]}"
            )

    # Extract metric values and drop NaN
    # Keep track of original DataFrame indices for mapping back
    series = df[metric_column].dropna()

    if len(series) < min_consecutive_windows:
        return []

    # Get metric values and window information
    y = series.to_numpy()
    t_indices = np.arange(len(y))
    
    # Store the original DataFrame indices for mapping PELT results back
    series_df_indices = series.index.tolist()

    # Create signal for PELT (2D: [y, t])
    signal = np.column_stack([y, t_indices])

    # Run PELT segmentation
    algo = rpt.Pelt(model="linear", min_size=min_consecutive_windows).fit(signal)
    bkps = algo.predict(pen=penalty)

    # Change points are segment ends excluding the last endpoint
    cps_window_indices = [b for b in bkps if b < len(y)]

    if len(cps_window_indices) == 0:
        return []

    # Convert window indices to trace indices
    # PELT returns indices in the y array (0, 1, 2, ...), which correspond to
    # positions in the series after dropna(). Map these back to DataFrame rows.
    change_points = []
    for cp_window_idx in cps_window_indices:
        if cp_window_idx < len(series_df_indices):
            # Get the DataFrame row index for this window
            df_row_idx = series_df_indices[cp_window_idx]
            # Get the last_index (end of window) as the change point trace index
            cp_trace_index = int(df.loc[df_row_idx, "last_index"])
            change_points.append(cp_trace_index)

    return sorted(change_points)
