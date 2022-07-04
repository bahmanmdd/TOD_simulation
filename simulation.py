"""
main package for simulation of shipping processes
"""

# import packages
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
import data_prep
# suppress warnings
import visualize

warnings.filterwarnings("ignore")
plt.ioff()


def get_parameters(n_vh):
    """ simulation parameters """

    runs = 1
    case = 'sample'

    to2v = 0.2
    su_t = 0

    n_to = int(round(n_vh * to2v))

    return runs, case, n_vh, n_to, su_t


class Vehicle(object):

    def __init__(self, vid, toid, stage, status, pattern, distribution, q_times):
        self.vid = vid
        self.toid = toid
        self.stage = stage
        self.status = status
        self.pattern = pattern
        self.q_times = q_times
        self.distribution = distribution

    def increment_stage(self):
        self.stage += 1
        self.status = self.pattern[self.stage]

    def get_current_activity(self, current_time):
        duration_act = self.distribution[self.stage]
        return current_time, duration_act, current_time + duration_act, self.status

    def get_next_activity(self, current_time):
        if (self.status == 'Signed off') or (self.pattern[self.stage + 1] == 'Signed off'):
            return current_time, math.inf, math.inf, 'Signed off'
        else:
            duration_act = self.distribution[self.stage + 1]
            return current_time, duration_act, current_time + duration_act, self.pattern[self.stage + 1]


class Teleoperator(object):

    def __init__(self, toid, status, vid):
        self.vid = vid
        self.toid = toid
        self.status = status


def run_simulation(replication_no, output_dir, runs, case, n_vh, n_to, setup_to, act_seq, act_dist, begin_times):

    # todo:
    ## event list to numpy array for efficiency (list of column names in dic and array instead of df)

    ##################
    # initialization #
    ##################

    # rng seed
    np.random.seed(seed=replication_no)

    # initialize variables, lists and objects
    vh_dict = {'V{0}'.format(i + 1): Vehicle('V{0}'.format(i + 1), None, 0, act_seq[i][0], act_seq[i], act_dist[i], [])
               for i in range(n_vh)}
    to_dict = {'TO{0}'.format(i + 1): Teleoperator('TO{0}'.format(i + 1), 'Idle', None) for i in range(n_to)}
    st_list = list(dict.fromkeys([item for sublist in act_seq for item in sublist]))
    qs_list = list(dict.fromkeys(['Vehicles_waiting', 'Queue_length']))
    st_list_to = list(dict.fromkeys(['Idle', 'Busy']))
    extra_event = None
    extra_activity = None

    # instantaneous states & queues
    queues_to_list = []
    queues_to_leng = 0
    states_vh = {}
    states_to = {}
    for status in st_list:
        states_vh[status] = sum(v.status == status for v in vh_dict.values())
    for status in st_list_to:
        states_to[status] = sum(to.status == status for to in to_dict.values())

    # full states & queue history (for statistics)
    states_vh_df = pd.DataFrame(columns=[st for st in st_list])
    states_to_df = pd.DataFrame(columns=[st for st in st_list_to])
    queues_df = pd.DataFrame(columns=[st for st in qs_list])

    # clock & event list
    simulation_time = 0
    simulation_clock = 0
    event_list = pd.DataFrame(columns=['Begin', 'Duration', 'End', 'Event', 'Activity', 'Vehicle', 'TO'])
    event_log = pd.DataFrame(columns=['Begin', 'Duration', 'End', 'Event', 'Activity', 'Vehicle', 'TO'])

    #################
    # time 0 events #
    #################

    # create first events (sign ins) and add to event list
    for vehicle in vh_dict.values():
        vehicle.stage = 0
        first_activity = vehicle.get_current_activity(simulation_time)
        first_event = {'Begin': first_activity[0],
                       'Duration': first_activity[1],
                       'End': first_activity[2],
                       'Event': 'Begin',
                       'Activity': first_activity[3],
                       'Vehicle': vehicle.vid,
                       'TO': vehicle.toid}
        event_list = event_list.append(first_event, ignore_index=True)
        vehicle.stage = -1

    # sort event list
    event_list = event_list.sort_values(by=['Begin'])
    event_log = event_list

    ##############
    # simulation #
    ##############

    # event execution loop
    while True:

        # find next event in event list
        current_event = event_list.iloc[0]

        # update simulation time
        if current_event['Begin'] > simulation_time:
            simulation_time = current_event['Begin']

        # determine vehicle & TO
        vehicle = vh_dict[current_event['Vehicle']]
        if current_event['TO']:
            teleoperator = to_dict[current_event['TO']]
            teleoperator.vid = vehicle.vid
        else:
            teleoperator = None

        ################
        # begin events #
        ################
        if current_event['Event'] == 'Begin':

            # change vehicle status
            if vehicle.status == 'Signed off':
                pass
            elif vehicle.status == 'TO Queue' and current_event['Activity'] == 'TO Queue' and current_event[
                'Event'] == 'Begin':
                pass
            else:
                vehicle.increment_stage()

            # queueing activity
            if current_event['Activity'] == 'TO Queue':

                # assign TO (if available)
                teleoperator = next((to for to in to_dict.values() if to.status == 'Idle'), None)

                # when there is a TO available
                if teleoperator is not None:
                    current_event['Duration'] = 0
                    teleoperator.status = 'Busy'
                    teleoperator.vehicle = vehicle.vid
                    vehicle.toid = teleoperator.toid
                    # if the vehicle was in the Q: update Q
                    if vehicle.vid in queues_to_list:
                        queues_to_list.remove(vehicle.vid)
                        queues_to_leng = len(queues_to_list)
                        vehicle.q_times[-1] = simulation_time - vehicle.q_times[-1]
                    # create current activity with default TO setup time (to be added to event list later)
                    current_activity = (
                    simulation_time, setup_to, simulation_time + setup_to, current_event['Activity'])

                # when no TO is available
                else:
                    if vehicle.vid not in queues_to_list:
                        queues_to_list.append(vehicle.vid)
                    queues_to_leng = len(queues_to_list)
                    vehicle.toid = None
                    vehicle.q_times.append(simulation_time)
                    # create current activity with duration 0 (to be added to event list later)
                    current_activity = (simulation_time, 0, simulation_time, current_event['Activity'])

            # other activities
            else:
                # create current activity (to be added to event list later)
                current_activity = (
                simulation_time, current_event['Duration'], simulation_time + current_event['Duration'],
                current_event['Activity'])

            # create next event to add to event list
            next_event = {'Begin': current_activity[2],
                          'Duration': 0,
                          'End': current_activity[2],
                          'Event': 'End',
                          'Activity': current_activity[3],
                          'Vehicle': vehicle.vid,
                          'TO': vehicle.toid}

        ##############
        # end events #
        ##############
        if current_event['Event'] == 'End':

            if vehicle.status == 'Signed off':
                continue

            # queueing activity
            if current_event['Activity'] == 'TO Queue':

                # if TO was assigned in TOQ begin
                if vehicle.toid and vehicle.toid == teleoperator.toid:
                    next_activity = vehicle.get_next_activity(simulation_time)

                # if still in queue
                else:
                    next_activity = None
                    # next_q_dissolve = event_list.query('Event=="End" & Activity=="Moving"').iloc[0]['Begin']
                    # next_activity = (next_q_dissolve, 0, next_q_dissolve, 'TO Queue')

            # moving activity
            elif current_event['Activity'] == 'Moving':
                # release TO
                teleoperator.status = 'Idle'
                teleoperator.vid = None
                vehicle.toid = None
                next_activity = vehicle.get_next_activity(simulation_time)
                # if there are vehicles in TOQ, create moving event for the first in Q
                if queues_to_list:
                    next_vehicle = vh_dict[queues_to_list[0]]
                    extra_activity = next_vehicle.get_current_activity(simulation_time)
                    # create extra event to add to event list
                    extra_event = {'Begin': extra_activity[0],
                                   'Duration': extra_activity[1],
                                   'End': extra_activity[2],
                                   'Event': 'Begin',
                                   'Activity': extra_activity[3],
                                   'Vehicle': next_vehicle.vid,
                                   'TO': next_vehicle.toid}

            # other activities
            else:
                # find the next relevant Begin (for the same vehicle)
                next_activity = vehicle.get_next_activity(simulation_time)

            # create next event to add to event list
            if next_activity:
                next_event = {'Begin': next_activity[0],
                              'Duration': next_activity[1],
                              'End': next_activity[2],
                              'Event': 'Begin',
                              'Activity': next_activity[3],
                              'Vehicle': vehicle.vid,
                              'TO': vehicle.toid}
            else:
                # in case the vehicle is still in TOQ
                next_event = None

        ###############
        # after event #
        ###############

        # eliminate done event & add next event to event list
        if next_event:
            event_list.iloc[0, :] = next_event
            event_log = event_log.append(next_event, ignore_index=True)
            # sort event list
            event_list = event_list.sort_values(by=['Begin', 'Event'], ascending=[True, False])
        else:
            event_list = event_list.iloc[1:]

        # add moving event for the first vehicle in Q
        if extra_event:
            row = []
            row.insert(0, extra_event)
            event_list = pd.concat([pd.DataFrame(row), event_list], ignore_index=True)
            event_log = event_log.append(extra_event, ignore_index=True)
            extra_activity = None
            extra_event = None

        # update stats
        for status in st_list:
            states_vh[status] = sum(v.status == status for v in vh_dict.values())
        for status in st_list_to:
            states_to[status] = sum(to.status == status for to in to_dict.values())
        states_vh_df.loc[simulation_time] = states_vh
        states_to_df.loc[simulation_time] = states_to
        queues_df.loc[simulation_time, qs_list[1]] = queues_to_leng

        # check for termination conditions
        if all(v.status == 'Signed off' for v in vh_dict.values()):
            n_mins = simulation_time
            break

    ###########
    # wrap up #
    ###########

    # save summary plot
    if replication_no == 1:
        visualize.plot_summary(states_vh_df, states_to_df, queues_df, output_dir, replication_no, runs, case, n_vh, n_to, setup_to)

    # sort event log
    event_log = event_log.sort_values(by=['Begin'], ascending=[True])
    event_log = event_log.query('Event!="End"')
    mask = np.logical_and(event_log['Activity'] == 'TO Queue', event_log['Duration'] == 0)
    event_log = event_log.loc[~mask]
    event_log = event_log.drop('Event', axis=1)

    # status summaries
    event_tmp = event_log[['Activity', 'Duration']]
    event_tmp = event_tmp.query('Activity != "Signed in"')
    event_tmp.Duration = pd.to_numeric(event_tmp.Duration)
    summary_sts = event_tmp.groupby('Activity').describe()
    summary_cnt = summary_sts.iloc[:, 0] / n_vh

    # utilization rates
    utilization_vh_avg = np.sum(event_log.query('Activity=="Moving"')['Duration']) / (n_mins * n_vh)
    utilization_to_avg = np.sum(event_log.query('TO==TO')['Duration']) / (n_mins * n_to)

    # queues
    indices = states_vh_df.index.values
    intervs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
    q_sizes = states_vh_df['TO Queue'].values
    queues_total_time = np.dot(q_sizes[:-1], intervs)
    queues_vh_time_avg = queues_total_time / n_vh
    queues_to_leng_avg = queues_total_time / n_mins
    queues_to_leng_max = np.max(q_sizes)

    wait_times_nested = [v.q_times for v in vh_dict.values() if v.q_times]
    wait_times = [item for sublist in wait_times_nested for item in sublist]
    if not wait_times:  wait_times = 0
    summary_qus = pd.Series(wait_times, name='Q Duration').describe()
    summary_qus = summary_qus.fillna(0)

    # utilization+ summary
    summary_utl = pd.DataFrame({'AVG_vehicle_utilization': np.round(utilization_vh_avg, 2),
                                'AVG_TO_utilization': np.round(utilization_to_avg, 2),
                                'AVG_Q_time_per_vehicle': np.round(queues_vh_time_avg, 2),
                                'AVG_Q_time_per_queue': np.round(np.mean(wait_times), 2),
                                'MAX_Q_time_per_queue': np.round(np.max(wait_times), 2),
                                'AVG_Q_length': np.round(queues_to_leng_avg, 2),
                                'Max_Q_length': queues_to_leng_max}, index=[0])

    # save main stats
    summary_utl.to_csv(output_dir + '/R_{0}'.format(replication_no) + '_summary_utilization.csv', index=False)
    summary_sts.to_csv(output_dir + '/R_{0}'.format(replication_no) + '_summary_status.csv')
    summary_cnt.to_csv(output_dir + '/R_{0}'.format(replication_no) + '_summary_counts.csv')
    summary_qus.to_csv(output_dir + '/R_{0}'.format(replication_no) + '_summary_queues.csv')

    # save detailed stats (only for the first replication)
    if replication_no == 1:
        event_log.to_csv(output_dir + '/R_{0}'.format(replication_no) + '_events.csv', index_label='Simulation_time')
        queues_df.to_csv(output_dir + '/R_{0}'.format(replication_no) + '_queues.csv', index_label='Simulation_time')
        states_vh_df.to_csv(output_dir + '/R_{0}'.format(replication_no) + '_states_vh.csv',
                            index_label='Simulation_time')
        states_to_df.to_csv(output_dir + '/R_{0}'.format(replication_no) + '_states_to.csv',
                            index_label='Simulation_time')
        # plot final results and save graphs
        visualize.plot_results(states_vh_df, states_to_df, queues_df, output_dir, replication_no, n_vh, n_to)

    return summary_utl, summary_sts, summary_cnt, summary_qus, states_vh_df.index[-1]


if __name__ == "__main__":

    Begin_dp = datetime.now()

    # run data preprocessing and return simulation input
    n_vh, act_seq, act_dist, begin_times = data_prep.simulation_input()

    # report data processing run time
    print('Dara preprocessing run time: ')
    print(datetime.now() - Begin_dp)

    runs, case, n_vh, n_to, setup_to = get_parameters(n_vh)

    # create output directory (if it does not exist already)
    output_dir = 'Output/' + case + '_v-{}'.format(n_vh) + '_to2v-{:.2f}'.format(n_to/n_vh) + '_su-{}'.format(setup_to) + '_R-{}'.format(runs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # record simulation run time
    Begin = datetime.now()

    # replications
    print('Simulation in progress...')
    for r in range(runs):
        # run simulation
        print('Running replication {0}'.format(r + 1))
        utl, sts, cnt, qus, srt = run_simulation(r + 1, output_dir, runs, case, n_vh, n_to, setup_to, act_seq, act_dist, begin_times)

        # record stats
        if r == 0:
            utilizations = utl
            statuses = sts.transpose().reset_index()
            counts = pd.DataFrame(cnt).transpose().reset_index()
            queues = pd.DataFrame(qus).transpose().reset_index()
            times = [srt]
        else:
            utilizations = utilizations.append(utl, ignore_index=True)
            statuses = statuses.append(sts.transpose().reset_index(), ignore_index=True)
            counts = counts.append(pd.DataFrame(cnt).transpose().reset_index(), ignore_index=True)
            queues = queues.append(pd.DataFrame(qus).transpose().reset_index(), ignore_index=True)
            times.append(srt)

    # report simulation run time
    print('\nSimulation run time for {0} run(s): '.format(runs))
    print(datetime.now() - Begin)
    print('\nReplication run time: ')
    print((datetime.now() - Begin) / runs)

    # save summary stats
    utilizations.index = utilizations.index + 1
    summary_utilization = utilizations.describe()
    summary_status = statuses.groupby('level_1').mean()
    summary_status.index = summary_status.index.str.strip()
    summary_status = summary_status.reindex(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    summary_count = counts.describe()
    summary_queue = queues.describe()
    summary_times = pd.Series(times, name='Duration').describe()

    summary_utilization.transpose().rename(columns={'count': 'replications'}, inplace=False).to_excel(
        output_dir + '/R_0_summary_utilization.xlsx')
    summary_status.transpose().to_excel(output_dir + '/R_0_summary_status.xlsx')
    summary_count.transpose().rename(columns={'count': 'replications'}, inplace=False).to_excel(
        output_dir + '/R_0_summary_count.xlsx')
    summary_queue.to_excel(output_dir + '/R_0_summary_queues.xlsx')
    summary_times.to_excel(output_dir + '/R_0_summary_times.xlsx')

    utilizations.to_csv(output_dir + '/R_0_full_utilization.csv', index_label='Replication')
