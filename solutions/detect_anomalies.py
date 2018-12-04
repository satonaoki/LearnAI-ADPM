import numpy as np

def detect_anomalies(ts, tolerance=4):
    m = np.mean(ts)
    std = np.std(ts)
    
    cutoff = m + tolerance * std
    
    # find the array indices of extreme values (anomalies)
    idx = np.where(ts > cutoff)[0].tolist()

    # create an array that is all NaN, except for the anomalies
    anoms = np.full(ts.shape[0], np.nan)
    anoms[idx] = ts[idx] # copy the value of the anomaly
    
    return anoms, m, cutoff
