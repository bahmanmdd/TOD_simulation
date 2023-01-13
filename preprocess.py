"""
preprocessing MassGT data for simulation of teleoperated driving in shipping processes
created by: Bahman Madadi
"""

from itertools import chain
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


def select_tours(tour_len, tour_begin, runs, proportion):

    # read data
    data_full = pd.read_csv('Input/Tours_REF.csv')

    # shift all trips by a uniform random number between 0-1 hour (to avoid same start times)
    tour_sizes = data_full.groupby('TOUR_ID', sort=False).size()
    tour_delay = data_full.groupby('TOUR_ID', sort=False).apply(lambda x: np.random.uniform(0, 1))
    # tour_delay = pd.Series(np.random.uniform(0, 1, len(tour_sizes)))
    trip_delay = tour_delay.reindex(tour_delay.index.repeat(tour_sizes)).values
    data_full['TOUR_DEPTIME'] = data_full['TOUR_DEPTIME'] + trip_delay
    data_full['TRIP_DEPTIME'] = data_full['TRIP_DEPTIME'] + trip_delay
    data_full['TRIP_ARRTIME'] = data_full['TRIP_ARRTIME'] + trip_delay

    # filter based on tour begin and tour duration
    data = data_full[(data_full['TOUR_DEPTIME'] >= tour_begin) & (data_full['TRIP_ARRTIME'] <= (tour_begin+tour_len))]

    # fix id issues
    data[['CARRIER_ID', 'TOUR_ID', 'TRIP_ID']] = data[['CARRIER_ID', 'TOUR_ID', 'TRIP_ID']].astype(int)
    data['TOUR_ID'] = data[['CARRIER_ID', 'TOUR_ID']].apply(tuple, axis=1)

    # sort data
    data = data.sort_values(by=['TOUR_DEPTIME', 'CARRIER_ID', 'TOUR_ID', 'TRIP_ID'])
    tours = data['TOUR_ID'].unique()

    for r in range(runs):

        # rng seed
        np.random.seed(seed=r)

        # filter based on the proportion of carriers to include in study
        tours_sample = sorted(np.random.choice(tours, int(len(tours) * proportion), replace=False))
        data_sample = data[data['TOUR_ID'].isin(tours_sample)]

        # new df with only values we need
        input_data = pd.DataFrame()
        input_data['vehicle_id'] = data_sample['TOUR_ID']
        input_data['trip_id'] = data_sample['TRIP_ID']
        input_data['tour_departure'] = data_sample['TOUR_DEPTIME'] * 60
        input_data['moving_duration'] = (data_sample['TRIP_ARRTIME'] - data['TRIP_DEPTIME']) * 60

        # calculate buffer values (extract MassGT generated values)
        data_np = data_sample.values
        buffer = np.zeros(len(data_np))
        for i in range(len(data_np)):
            if data_np[i, 2] == 0:
                buffer[i] = data_np[i, 16] - data_np[i, 15]
            else:
                buffer[i] = data_np[i, 16] - data_np[i - 1, 17]
        data_sample['buffer'] = buffer

        # buffer from MassGT generated values
        input_data['buffer_duration'] = data_sample['buffer'] * 60

        # # save filtered tours
        input_data.to_csv('Input/Tours_filtered_S' + str(r) + '.csv', index=False)


def simulation_input(run_number, takeover_time):

    # read data
    input_data = pd.read_csv('Input/Tours_filtered_S' + str(run_number) + '.csv')

    input_data['moving_duration'] = np.around(input_data['moving_duration'], 4)
    input_data['buffer_duration'] = np.around(input_data['buffer_duration'], 4)

    begin_times = input_data.groupby(['vehicle_id'], sort=False).first()['tour_departure']
    total_to_planned = np.sum(input_data['moving_duration'])

    vid = input_data['vehicle_id'].unique()
    n_vh = len(vid)
    act_seq = [[] for _ in range(n_vh)]
    tour_len = input_data['vehicle_id'].value_counts(sort=False).to_numpy()

    temp = pd.DataFrame(columns=['vid', 'tid', 'dists'])
    temp['vid'] = input_data[['vehicle_id']]
    temp['tid'] = input_data[['trip_id']]
    # only buffer and moving
    temp['dists'] = input_data[input_data.columns[-2:]].apply(lambda x: [x[1], 0, takeover_time, x[0]], axis=1)
    temp2 = temp.groupby(['vid'], sort=False)['dists'].apply(lambda x: list(x[:]))
    act_dist = [list(chain(*x)) for x in temp2.values]

    for v in range(len(vid)):
        act_seq[v] = ['Idle', 'TO Queue', 'Takeover', 'Teleoperated'] * tour_len[v]
        act_seq[v].append('Signed off')
        act_dist[v].append(0)

    return n_vh, act_seq, act_dist, np.around(begin_times.values, 3), total_to_planned


