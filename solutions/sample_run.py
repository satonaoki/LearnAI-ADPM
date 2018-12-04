import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score
import time

from pyculiarity import detect_ts

def sample_run(df, anoms_ref, window_size = 500, com = 12, n_epochs=10):
    """
    This functions expects a dataframe df as mandatory argument.  
    The first column of the df should contain timestamps, the second machine IDs
    
    arguments:
    df: a pandas data frame with two columns: 1. timestamp, 2. value
    anoms_ref: reference anomaly detection results 
    
    Keyword arguments:
    window_size: the size of the window of data points that are used for anomaly detection
    com: decay in terms of center of mass (this approximates averageing over about twice as many hours)
    """

    p_anoms = .1

    def detect_ts_online(df_smooth, window_size, stop):
        is_anomaly = False
        run_time = 9999
        start_index = max(0, stop - window_size)
        df_win = df_smooth.iloc[start_index:stop, :]
        start_time = time.time()
        results = detect_ts(df_win, alpha=0.05, max_anoms=0.02, only_last=None, longterm=False, e_value=False, direction='both')
        run_time = time.time() - start_time
        if results['anoms'].shape[0] > 0:
            timestamp = df_win['timestamp'].tail(1).values[0]
            if timestamp == results['anoms'].tail(1)['timestamp'].values[0]:
                is_anomaly = True
        return is_anomaly, run_time

    def running_avg(ts, com=6):
        rm_o = np.zeros_like(ts)
        rm_o[0] = ts[0]
    
        for r in range(1, len(ts)):
            curr_com = float(min(com, r))
            rm_o[r] = rm_o[r-1] + (ts[r] - rm_o[r-1])/(curr_com + 1)
    
        return rm_o

    # create arrays that will hold the results of batch AD (y_true) and online AD (y_pred)
    y_true = [False] * n_epochs
    y_pred = [True] * n_epochs
    run_times = []
    
    # check which unique machines, sensors, and timestamps we have in the dataset
    machineIDs = df['machineID'].unique()
    sensors = df.columns[2:]
    timestamps = df['datetime'].unique()[window_size:]
    
    # sample n_machines_test random machines and sensors 
    random_machines = np.random.choice(machineIDs, n_epochs)
    random_sensors = np.random.choice(sensors, n_epochs)

    # we intialize an array with that will later hold a sample of timetamps
    random_timestamps = np.random.choice(timestamps, n_epochs)
    
    for i in range(0, n_epochs):
        # take a slice of the dataframe that only contains the measures of one random machine
        df_s = df[df['machineID'] == random_machines[i]]
        
        # smooth the values of one random sensor, using our running_avg function
        smooth_values = running_avg(df_s[random_sensors[i]].values, com)
        
        # create a data frame with two columns: timestamp, and smoothed values
        df_smooth = pd.DataFrame(data={'timestamp': df_s['datetime'].values, 'value': smooth_values})

        # load the results of batch AD for this machine and sensor
        anoms_s = anoms_ref[((anoms_ref['machineID'] == random_machines[i]) & (anoms_ref['errorID'] == random_sensors[i]))]
                
        # find the location of the t'th random timestamp in the data frame
        if np.random.random() < p_anoms:
            anoms_timestamps = anoms_s['datetime'].values
            np.random.shuffle(anoms_timestamps)
            counter = 0
            while anoms_timestamps[0] < timestamps[0]:
                if counter > 100:
                    return 0.0, 9999.0
                np.random.shuffle(anoms_timestamps)
                counter += 1
            random_timestamps[i] = anoms_timestamps[0]
            
        # select the test case
        test_case = df_smooth[df_smooth['timestamp'] == random_timestamps[i]]
        test_case_index = test_case.index.values[0]


        # check whether the batch AD found an anomaly at that time stamps and copy into y_true at idx
        y_true_i = random_timestamps[i] in anoms_s['datetime'].values

        # perform online AD, and write result to y_pred
        y_pred_i, run_times_i = detect_ts_online(df_smooth, window_size, test_case_index)
        
        y_true[i] = y_true_i
        y_pred[i] = y_pred_i
        run_times.append(run_times_i)
            
    return fbeta_score(y_true, y_pred, beta=2), np.mean(run_times)
    
