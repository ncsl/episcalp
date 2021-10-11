import json
import numpy as np
from datetime import datetime, date


def compute_sample_points(ntimes, winsize, stepsize, clip=True):
    sample_points = []
    start_samp = 0
    while start_samp < ntimes:
        end_samp = start_samp + winsize
        if end_samp > ntimes:
            if clip:
                return sample_points
            else:
                end_samp = ntimes
        sample_points.append((start_samp, end_samp))
        start_samp = start_samp + stepsize
    return sample_points


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.

    Pass to json.dump(), or json.load().
    """

    def default(self, obj):  # noqa
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)
