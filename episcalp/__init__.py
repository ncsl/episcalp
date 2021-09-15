"""EPiscalp software for doing analysis on epilepsy scalp eeg."""
name = "episcalp"
__version__ = "0.0.1"

from .io import read_scalp_eeg
from .utils import (
    _parse_sz_events,
    _parse_events_for_description,
    get_best_matching_montage,
)
from .preprocess import run_pyprep
