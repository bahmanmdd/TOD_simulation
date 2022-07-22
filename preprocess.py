import math
from itertools import chain
import pandas as pd
import numpy as np
import scipy.stats


def select_tours(tour_len, tour_begin):

    # read data
    data = pd.read_csv('Input/Tours_REF.csv')

    # filter based on tour begin and tour duration
    data_filtered = data[(data['TOUR_DEPTIME'] >= tour_begin) & (data['TRIP_ARRTIME'] <= (tour_begin+tour_len))]

    # save filtered tours
    data_filtered.to_csv('Input/Tours_filtered.csv', index=False)


def simulation_input(takeover_time):

    # proportion of carriers to include
    ## Todo: later include all or filter another way
    proportion = 0.01

    # read data
    data = pd.read_csv('Input/Tours_filtered.csv')

    # fix id issues
    data['CARRIER_ID'] = data['CARRIER_ID'].astype(int)
    data['TOUR_ID'] = data['TOUR_ID'].apply(lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1])))
    data['TRIP_ID'] = data['TRIP_ID'].apply(lambda x: int(x.split('_')[-1]))

    # filter based on the proportion of carriers to include in study
    carriers = data['CARRIER_ID'].unique()
    data = data[data['CARRIER_ID'].isin(carriers[:int(len(carriers)*proportion)])]

    # sort data
    data = data.sort_values(by=['TOUR_DEPTIME', 'CARRIER_ID', 'TOUR_ID', 'TRIP_ID'])

    # calculate buffer values
    # buffer = np.zeros(len(data_np))
    # for i in range(len(data_np)):
    #     if data_np[i, 2] == 0:
    #         buffer[i] = data_np[i, 16] - data_np[i, 15]
    #     else:
    #         buffer[i] = data_np[i, 16] - data_np[i-1, 17]
    # data['buffer'] = buffer

    # new df with only values we need
    input_data = pd.DataFrame()
    input_data['vehicle_id'] = data['TOUR_ID']
    input_data['trip_id'] = data['TRIP_ID']
    input_data['tour_departure'] = data['TOUR_DEPTIME'] * 60
    # input_data['buffer_duration'] = data['buffer'] * 60
    rv = scipy.stats.expon(loc=15, scale=20)
    input_data['buffer_duration'] = rv.rvs(size=len(data['TRIP_ID']))
    input_data['moving_duration'] = (data['TRIP_ARRTIME'] - data['TRIP_DEPTIME']) * 60

    begin_times = input_data.groupby(['vehicle_id'], sort=False).first()['tour_departure']

    vid = input_data['vehicle_id'].unique()
    n_vh = len(vid)
    act_seq = [[] for _ in range(n_vh)]
    tour_len = input_data['vehicle_id'].value_counts(sort=False).to_numpy()

    temp = pd.DataFrame(columns=['vid', 'tid', 'dists'])
    temp['vid'] = input_data[['vehicle_id']]
    temp['tid'] = input_data[['trip_id']]
    # only buffer and moving
    temp['dists'] = input_data[input_data.columns[-2:]].apply(lambda x: [x[0], 0, takeover_time, x[1]], axis=1)
    temp2 = temp.groupby(['vid'], sort=False)['dists'].apply(lambda x: list(x[:]))
    act_dist = [list(chain(*x)) for x in temp2.values]

    for v in range(len(vid)):
        act_seq[v] = ['Idle', 'TO Queue', 'Takeover', 'Teleoperated'] * tour_len[v]
        act_seq[v].append('Signed off')
        act_dist[v].append(0)

    return n_vh, act_seq, act_dist, begin_times.values

    # for testing and validation
    # np.array_equal(temp2.index.values, input_data['vehicle_id'].unique())
    # np.array_equal(begin_times.index.values, input_data['vehicle_id'].unique())
    # len_dist = np.array([len(x) for x in act_dist])
    # len_seq = np.array([len(x) for x in act_seq])
    # np.array_equal(len_seq, len_dist)
    # np.min(input_data['buffer_duration'])
