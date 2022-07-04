import math
from itertools import chain
import pandas as pd
import numpy as np


def simulation_input():

    # turn these into input parameters
    carrier_prop = 0.1
    max_tour_len = math.inf
    region = [-math.inf, math.inf, -math.inf, math.inf]

    # read data
    data = pd.read_csv('Input/Tours_REF.csv')

    # fileter data based on maximum tour length
    data = data[data['TRIP_ARRTIME'] < max_tour_len]
    # filter based on the proportion of carriers to include in study
    carriers = data['CARRIER_ID'].unique()
    data = data[data['CARRIER_ID'].isin(carriers[:int(len(carriers)*carrier_prop)])]
    # filter based on region
    data = data[(region[0] <= data['X_ORIG']) & (data['X_ORIG'] <= region[1])]
    data = data[(region[0] <= data['X_DEST']) & (data['X_DEST'] <= region[1])]
    data = data[(region[2] <= data['Y_DEST']) & (data['Y_DEST'] <= region[3])]
    data = data[(region[2] <= data['Y_ORIG']) & (data['Y_ORIG'] <= region[3])]

    data['CARRIER_ID'] = data['CARRIER_ID'].astype(int)
    data['TOUR_ID'] = data['TOUR_ID'].apply(lambda x: int(x.split('_')[-1]))
    data['TRIP_ID'] = data['TRIP_ID'].apply(lambda x: int(x.split('_')[-1]))

    # sort data
    data = data.sort_values(by=['TOUR_DEPTIME', 'CARRIER_ID', 'TOUR_ID', 'TRIP_ID'])
    data_np = data.values

    # calculate buffer values
    buffer = np.zeros(len(data_np))
    for i in range(len(data_np)):
        if data_np[i, 2] == 0:
            buffer[i] = data_np[i, 16] - data_np[i, 15]
        else:
            buffer[i] = data_np[i, 16] - data_np[i-1, 17]
    data['buffer'] = buffer

    # new df with only values we need
    input_data = pd.DataFrame()
    input_data['vehicle_id'] = data['TOUR_ID']
    input_data['trip_id'] = data['TRIP_ID']
    input_data['tour_departure'] = data['TOUR_DEPTIME'] * 60
    input_data['buffer_duration'] = data['buffer'] * 60
    input_data['departure_time'] = data['TRIP_DEPTIME'] * 60
    input_data['moving_duration'] = (data['TRIP_ARRTIME'] - data['TRIP_DEPTIME']) * 60

    begin_times = input_data.groupby(['vehicle_id'], sort=False).first()['tour_departure']
    begin_times_np = begin_times.values

    vid = input_data['vehicle_id'].unique()
    n_vh = len(vid)
    act_seq = [[] for _ in range(n_vh)]
    tour_len = input_data['vehicle_id'].value_counts(sort=False).to_numpy()

    temp = pd.DataFrame(columns=['vid', 'tid', 'dists'])
    temp['vid'] = input_data[['vehicle_id']]
    temp['tid'] = input_data[['trip_id']]
    # only buffer and moving
    temp['dists'] = input_data[input_data.columns[-3:]].apply(lambda x: [x[0], 0, x[2]], axis=1)
    temp2 = temp.groupby(['vid'], sort=False)['dists'].apply(lambda x: list(x[:]))
    act_dist = [list(chain(*x)) for x in temp2.values]

    for v in range(len(vid)):
        act_seq[v] = ['buffer', 'TO Queue', 'Moving'] * tour_len[v]
        act_seq[v].insert(0, 'Signed in')
        act_seq[v].append('Signed off')
        act_dist[v].insert(0, begin_times_np[v])
        act_dist[v].append(math.inf)

    return n_vh, act_seq, act_dist, begin_times_np

    # for testing and validation
    # np.array_equal(temp2.index.values, input_data['vehicle_id'].unique())
    # np.array_equal(begin_times.index.values, input_data['vehicle_id'].unique())
    # len_dist = np.array([len(x) for x in act_dist])
    # len_seq = np.array([len(x) for x in act_seq])
    # np.array_equal(len_seq, len_dist)
    # np.min(input_data['buffer_duration'])
