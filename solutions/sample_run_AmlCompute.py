
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""

This script was modified from the sample_run function of lab 1.2, such that it can be run on AmlCompute.

"""

import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score
import os
import time

from pyculiarity import detect_ts

from azureml.core import Run

import argparse # for parsing input arguments

def running_avg(ts, com=6):
    rm_o = np.zeros_like(ts)
    rm_o[0] = ts[0]
    
    for r in range(1, len(ts)):
        curr_com = float(min(com, r))
        rm_o[r] = rm_o[r-1] + (ts[r] - rm_o[r-1])/(curr_com + 1)
    
    return rm_o


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


def sample_run(df, anoms_batch, run, window_size = 500, com = 12, n_epochs=10):

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
        
        # smooth the values of one random sensor, using our run_avg function
        smooth_values = running_avg(df_s[random_sensors[i]].values, com)

        # create a data frame with two columns: timestamp, and smoothed values
        df_smooth = pd.DataFrame(data={'timestamp': df_s['datetime'].values, 'value': smooth_values})

        # load the results of batch AD for this machine and sensor
        anoms_s = anoms_batch[((anoms_batch['machineID'] == random_machines[i]) & (anoms_batch['errorID'] == random_sensors[i]))]
        
        # only do anomaly detection online, if the batch solution actually found an anomaly
        if anoms_s.shape[0] > 0:
            # Let's make sure we have at least one anomaly in our sample! Otherwise it doesn't make sense to calculate
            # any performance metric.  We can just use the timestamp of the last anomalys
            if i == 0:
                anoms_timestamps = anoms_s['datetime'].values
                random_timestamps[i] = anoms_timestamps[-1:][0]

            # select the row of the test case
            test_case = df_smooth[df_smooth['timestamp'] == random_timestamps[i]]
            test_case_index = test_case.index.values[0]

            # check whether the batch AD found an anomaly at that time stamps and copy into y_true at idx
            y_true_i = random_timestamps[i] in anoms_s['datetime'].values

            # perform online AD, and write result to y_pred
            y_pred_i, run_times_i = detect_ts_online(df_smooth, window_size, test_case_index)
        else:
            y_pred_i, y_true_i = 0, 0
            
        y_true[i] = y_true_i
        y_pred[i] = y_pred_i
        run_times.append(run_times_i)
        
        score = np.float(fbeta_score(y_true, y_pred, beta=2))
        print("fbeta_score: %s" % round(score, 2))
        
        run.log('run_time', np.mean(run_times))
        run.log('fbeta_score', score)
        
    run.log('final_fbeta_score', np.float(score))

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    parser.add_argument('--window_size', type=int, dest='window_size', default=100, help='window size')
    parser.add_argument('--com', type=int, dest='com', default=12, help='Specify decay in terms of center of mass for running avg')
    parser.add_argument('--n_epochs', type=int, dest='n_epochs', default=1000, help='Specify decay in terms of center of mass for running avg')
    args = parser.parse_args()

    data_folder = os.path.join(args.data_folder, 'telemetry')
    window_size = args.window_size
    com = args.com
    n_epochs = args.n_epochs
    
    # start an Azure ML run
    run = Run.get_context()

    print("Reading data ... ", end="")
    df = pd.read_csv(os.path.join(data_folder, 'telemetry.csv'))
    print("Done.")

    print("Parsing datetime...", end="")
    df['datetime'] = pd.to_datetime(df['datetime'], format="%m/%d/%Y %I:%M:%S %p")
    print("Done.")
    
    print("Reading data ... ", end="")
    anoms_batch = pd.read_csv(os.path.join(data_folder, 'anoms.csv'))
    anoms_batch['datetime'] = pd.to_datetime(anoms_batch['datetime'], format="%Y-%m-%d %H:%M:%S")
    print("Done.")

    print('Dataset is stored here: ', data_folder)

    sample_run(df, anoms_batch, run, window_size, com, n_epochs)

    
if __name__== "__main__":
      main()
