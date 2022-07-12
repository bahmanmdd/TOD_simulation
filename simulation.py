"""
main package for simulation of teleoperated driving in shipping processes
"""

# import packages
import math
import pandas as pd
import numpy as np
import os
from datetime import datetime
import preprocess
import visualize
import report


def parameters():

    ## simulation scenario parameters
    runs = 5
    max_tour_len = math.inf
    region = [-math.inf, math.inf, -math.inf, math.inf]

    # lists of parameter options for batch runs
    to2v_ratio_list = np.array(list(range(5, 105, 5))) / 100
    to2v_ratio_list = [0.1, 0.3, 0.5]
    takeover_time_list = [0, 1, 2, 5]
    takeover_time_list = [0, 1]
    carrier_proportion_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    carrier_proportion_list = [0.005]

    return runs, max_tour_len, region, to2v_ratio_list, takeover_time_list, carrier_proportion_list


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
            duration_act = float(self.distribution[self.stage + 1])
            return current_time, duration_act, current_time + duration_act, self.pattern[self.stage + 1]


class Teleoperator(object):

    def __init__(self, toid, status, vid):
        self.vid = vid
        self.toid = toid
        self.status = status


def run_simulation(replication_no, output_dir, runs, n_vh, n_to, setup_to, act_seq, act_dist):

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
    next_event = None
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
    event_list = []
    names = {'Begin': 0, 'Duration': 1, 'End': 2, 'Event': 3, 'Activity': 4, 'Vehicle': 5, 'TO': 6}

    #################
    # time 0 events #
    #################

    # create first events (sign ins) and add to event list
    for vehicle in vh_dict.values():
        vehicle.stage = 0
        first_activity = vehicle.get_current_activity(simulation_time)
        first_event = np.array([first_activity[0],
                                first_activity[1],
                                first_activity[2],
                                'Begin',
                                first_activity[3],
                                vehicle.vid,
                                vehicle.toid])
        event_list.append(first_event)
        vehicle.stage = -1

    # sort event list
    event_list = np.array(event_list)
    event_list[:, :3] = event_list[:, :3].astype(float)
    event_list = event_list[event_list[:, names['Begin']].argsort()]
    event_list = event_list[event_list[:, names['End']].argsort()]
    event_log = event_list

    ##############
    # simulation #
    ##############

    # event execution loop
    while True:

        # find next event in event list
        current_event = event_list[0]

        # update simulation time
        if current_event[names['Begin']] > simulation_time:
            simulation_time = current_event[names['Begin']]

        # determine vehicle & TO
        vehicle = vh_dict[current_event[names['Vehicle']]]
        if current_event[names['TO']]:
            teleoperator = to_dict[current_event[names['TO']]]
            teleoperator.vid = vehicle.vid
        else:
            teleoperator = None

        ################
        # begin events #
        ################
        if current_event[names['Event']] == 'Begin':

            # change vehicle status
            if vehicle.status == 'Signed off':
                pass
            elif vehicle.status == 'TO Queue' \
                    and current_event[names['Activity']] == 'TO Queue' \
                    and current_event[names['Event']] == 'Begin':
                pass
            else:
                vehicle.increment_stage()

            # queueing activity
            if current_event[names['Activity']] == 'TO Queue':

                # assign TO (if available)
                teleoperator = next((to for to in to_dict.values() if to.status == 'Idle'), None)

                # when there is a TO available
                if teleoperator is not None:
                    current_event[names['Duration']] = 0
                    teleoperator.status = 'Busy'
                    teleoperator.vehicle = vehicle.vid
                    vehicle.toid = teleoperator.toid
                    # if the vehicle was in the Q: update Q
                    if vehicle.vid in queues_to_list:
                        queues_to_list.remove(vehicle.vid)
                        queues_to_leng = len(queues_to_list)
                        vehicle.q_times[-1] = simulation_time - vehicle.q_times[-1]
                    # create current activity with default TO setup time (to be added to event list later)
                    current_activity = (simulation_time,
                                        setup_to,
                                        simulation_time + setup_to,
                                        current_event[names['Activity']])

                # when no TO is available
                else:
                    if vehicle.vid not in queues_to_list:
                        queues_to_list.append(vehicle.vid)
                    queues_to_leng = len(queues_to_list)
                    vehicle.toid = None
                    vehicle.q_times.append(simulation_time)
                    # create current activity with duration 0 (to be added to event list later)
                    current_activity = (simulation_time,
                                        0,
                                        simulation_time,
                                        current_event[names['Activity']])

            # other activities
            else:
                # create current activity (to be added to event list later)
                current_activity = (simulation_time,
                                    current_event[names['Duration']],
                                    simulation_time + current_event[names['Duration']],
                                    current_event[names['Activity']])

            # create next event to add to event list
            next_event = np.array([current_activity[2],
                                   0,
                                   current_activity[2],
                                   'End',
                                   current_activity[3],
                                   vehicle.vid,
                                   vehicle.toid])

        ##############
        # end events #
        ##############
        if current_event[names['Event']] == 'End':

            if vehicle.status == 'Signed off':
                continue

            # queueing activity
            if current_event[names['Activity']] == 'TO Queue':

                # if TO was assigned in TOQ begin
                if vehicle.toid and vehicle.toid == teleoperator.toid:
                    next_activity = vehicle.get_next_activity(simulation_time)

                # if still in queue
                else:
                    next_activity = None

            # moving activity
            elif current_event[names['Activity']] == 'Moving':
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
                    extra_event = np.array([extra_activity[0],
                                            extra_activity[1],
                                            extra_activity[2],
                                            'Begin',
                                            extra_activity[3],
                                            next_vehicle.vid,
                                            next_vehicle.toid])

            # other activities
            else:
                # find the next relevant Begin (for the same vehicle)
                next_activity = vehicle.get_next_activity(simulation_time)

            # create next event to add to event list
            if next_activity is not None:
                next_event = np.array([next_activity[0],
                                       next_activity[1],
                                       next_activity[2],
                                       'Begin',
                                       next_activity[3],
                                       vehicle.vid,
                                       vehicle.toid])
            else:
                # in case the vehicle is still in TOQ
                next_event = None

        ###############
        # after event #
        ###############

        # eliminate done event & add next event to event list
        if next_event is not None:
            event_list[0] = next_event
            event_log = np.append(event_log, [next_event], axis=0)
            # sort event list
            event_list[:, :3] = event_list[:, :3].astype(float)
            event_list = event_list[event_list[:, names['Event']].argsort()[::-1]]
            event_list = event_list[event_list[:, names['End']].argsort()]
            event_list = event_list[event_list[:, names['Begin']].argsort()]
        else:
            event_list = event_list[1:]

        # add moving event for the first vehicle in Q
        if extra_event is not None:
            row = []
            row.insert(0, extra_event)
            event_list = np.vstack([row, event_list])
            event_log = np.append(event_log, [extra_event], axis=0)
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
        visualize.plot_summary(states_vh_df, states_to_df, queues_df, output_dir, replication_no, runs, n_vh, n_to, setup_to)

    # event log to dataframe
    event_log = pd.DataFrame(event_log.astype(str), columns=names.keys())

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
    event_log['Duration'] = pd.to_numeric(event_log['Duration'])
    utilization_vh_avg = np.sum(event_log.query('Activity=="Moving"')['Duration'])/(n_mins * n_vh)
    utilization_to_avg = np.sum(event_log[event_log['TO'].str.contains('TO', regex=False)]['Duration'])/(n_mins * n_to)

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
    if not wait_times: wait_times = 0
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

    # parameters
    runs, max_tour_len, region, to2v_ratio_list, takeover_time_list, carrier_proportion_list = parameters()

    # batch scenario runs
    for carrier_proportion in carrier_proportion_list:
        for to2v_ratio in to2v_ratio_list:
            for takeover_time in takeover_time_list:

                # create output directory (if it does not exist already)
                output_dir = 'Output/' + 'cp-{:.3f}'.format(carrier_proportion) + '_to2v-{:.2f}'.format(to2v_ratio) + '_su-{}'.format(takeover_time) + '_R-{}'.format(runs)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # record simulation run time
                Begin = datetime.now()

                # replications
                print('********************')
                print('Scenario parameters:')
                print('--------------------')
                print('Proportion of carriers included: {}'.format(carrier_proportion))
                print('Teleoperator to vehicle ratio: {}'.format(to2v_ratio))
                print('Teleoperator takeover time: {}'.format(takeover_time))
                print('Number of replications: {}'.format(runs))
                print('Simulation in progress...')
                for r in range(runs):
                    # run data preprocessing and return simulation input
                    print('Preprocessing for replication {0}'.format(r + 1))
                    n_vh, act_seq, act_dist = preprocess.simulation_input(carrier_proportion, max_tour_len, region)
                    n_to = int(round(n_vh * to2v_ratio))
                    # run simulation
                    print('Running replication {0}'.format(r + 1))
                    utl, sts, cnt, qus, srt = run_simulation(r + 1, output_dir, runs, n_vh, n_to, takeover_time, act_seq, act_dist)

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
                print('Simulation run time for {0} run(s): '.format(runs))
                print(datetime.now() - Begin)
                print('Replication run time (including data preprocessing): ')
                print((datetime.now() - Begin) / runs)
                print('***************************************************\n')

                # save summary stats
                report.stats_summary(utilizations, statuses, counts, queues, times, output_dir)

    # create plots to show tradeoffs between queue times and TO2V ratios
    report.tradeoff_plots(to2v_ratio_list, carrier_proportion_list, takeover_time_list, runs)

