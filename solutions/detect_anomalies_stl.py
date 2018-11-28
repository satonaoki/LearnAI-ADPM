import numpy as np
from rstl import STL

def detect_anomalies_stl(ts, tolerance=4):
    
    stl = STL(ts, freq=24*7, s_window='periodic')
    
    m = np.mean(stl.remainder)
    std = np.std(stl.remainder)

    # find the array indices of extreme values (anomalies)
    idx = np.where(stl.remainder > m + tolerance * std)[0].tolist()

    # create an array that is all NaN, except for the anomalies
    anoms = np.full(stl.remainder.shape[0], np.nan)
    anoms[idx] = stl.remainder[idx] # copy the value of the anomaly
    
    return anoms, stl.remainder
