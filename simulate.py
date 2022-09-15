"""
main package for simulation of teleoperated driving in shipping processes
"""

import math
import pandas as pd
import numpy as np
import os
from datetime import datetime
import preprocess
import visualize
import report
import event


def parameters():

    ## simulation scenario parameters
    replication = 5
    sample_size = 0.01
    simulation_start = [0]
    simulation_duration = [9, 24]

    ## model variation parameters
    to2v_ratios = np.array(list(range(20, 105, 5))) / 100
    takeover_times = [0, 2, 5]
    max_to_duration = 4.5 * 60
    rest_short = 10
    rest_long = 45

    return replication, sample_size, simulation_duration, simulation_start, to2v_ratios, takeover_times, max_to_duration, rest_short, rest_long


class Vehicle(object):

    def __init__(self, vid, toid, stage, status, pattern, distribution, q_times, q_begin):
        self.vid = vid
        self.toid = toid
        self.stage = stage
        self.status = status
        self.pattern = pattern
        self.q_times = q_times
        self.q_begin = q_begin
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


def run_simulation(replication_no, output_dir, runs, n_vh, n_to, setup_to, act_seq, act_dist, begin_times,
                   max_to_duration, rest_short, rest_long, tour_begin, tour_len, to_total):

    ##################
    # initialization #
    ##################

    # rng seed
    np.random.seed(seed=replication_no)

    # initialize variables, lists and objects
    vh_dict = {'V{0}'.format(i + 1): Vehicle('V{0}'.format(i + 1), None, 0, act_seq[i][0], act_seq[i], act_dist[i], [], None)
               for i in range(n_vh)}
    to_dict = {'TO{0}'.format(i + 1): Teleoperator('TO{0}'.format(i + 1), 'Idle', None)
               for i in range(n_to)}

    st_list = list(dict.fromkeys([item for sublist in act_seq for item in sublist]))
    st_list_to = list(dict.fromkeys(['Idle', 'Busy', 'Resting', 'Takeover']))

    tour_completion = 1
    distance_completion = 1


    next_event = None
    next_event_to = None

    # instantaneous states & queues
    queues_to_list = []
    states_vh = {}
    states_to = {}
    for status in st_list:
        states_vh[status] = sum(v.status == status for v in vh_dict.values())
    for status in st_list_to:
        states_to[status] = sum(to.status == status for to in to_dict.values())

    # full states & queue history (for statistics)
    states_vh_df = pd.DataFrame(columns=[st for st in st_list])
    states_to_df = pd.DataFrame(columns=[st for st in st_list_to])
    queues_df = pd.DataFrame(columns=['Queue length'])

    # clock & event list
    simulation_time = float(tour_begin)
    time_up = (tour_begin + tour_len) * 60
    event_list = []
    names = {'Begin': 0, 'Duration': 1, 'End': 2, 'State': 3, 'Event': 4, 'Vehicle': 5, 'TO': 6}

    #################
    # time 0 events #
    #################

    # create first events and add to event list
    event_list, vh_dict = event.create_first_events(simulation_time, event_list, vh_dict, begin_times, names, time_up)
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
        if current_event[names['Vehicle']]:
            vehicle = vh_dict[current_event[names['Vehicle']]]
        else:
            vehicle = None

        if current_event[names['TO']]:
            teleoperator = to_dict[current_event[names['TO']]]
        else:
            teleoperator = None

        #################
        # process event #
        #################

        if current_event[names['Event']] == 'Idle':
            next_event, vehicle = event.process_idle(simulation_time, vehicle, current_event, names)

        elif current_event[names['Event']] == 'TO Queue':
            next_event, vehicle, teleoperator, queues_to_list = event.process_queue(simulation_time, vehicle, current_event, to_dict, queues_to_list, names)

        elif current_event[names['Event']] == 'Takeover':
            next_event, vehicle = event.process_takeover(simulation_time, vehicle, current_event, names, takeover_time)

        elif current_event[names['Event']] == 'Teleoperated':
            next_event, vehicle, teleoperator, queues_to_list, vh_dict, next_event_to = event.process_teleoperated(
                simulation_time, current_event, names, vehicle, teleoperator, queues_to_list, vh_dict, rest_long, rest_short, max_to_duration)

        elif current_event[names['Event']] == 'Resting':
            next_event, teleoperator, queues_to_list = event.process_resting(simulation_time, teleoperator, current_event, names, queues_to_list, vh_dict)

        elif current_event[names['Event']] == 'Time up':
            next_event, tour_completion, distance_completion = event.time_up(vh_dict, to_total, time_up, event_log)

        elif current_event[names['Event']] == 'Signed off':
            pass

        ###############
        # after event #
        ###############

        # update event list
        event_list, event_log, next_event, next_event_to = event.update_event_list(event_list, event_log, next_event, next_event_to, names)

        # update stats
        for status in st_list:
            states_vh[status] = sum(v.status == status for v in vh_dict.values())
        for status in st_list_to:
            states_to[status] = sum(to.status == status for to in to_dict.values())
        states_vh_df.loc[simulation_time] = states_vh
        states_to_df.loc[simulation_time] = states_to
        queues_df.loc[simulation_time, 'Queue length'] = len(queues_to_list)

        # check for termination conditions
        if all(v.status == 'Signed off' for v in vh_dict.values()):
            n_mins = simulation_time
            break

    ###########
    # wrap up #
    ###########

    # save summary plot
    if replication_no == 1:
        visualize.plot_summary(states_vh_df, states_to_df, queues_df, output_dir, replication_no, runs, n_vh, n_to,
                               time_up)

    # event log to dataframe
    event_log[:, :4] = event_log[:, :4].astype(float)
    event_log = pd.DataFrame(event_log, columns=names.keys())

    # sort event log
    event_log = event_log.drop('State', axis=1)
    event_log = event_log[event_log['Duration'] > 0]
    event_log = event_log.sort_values(by=['Begin', 'Duration'])

    # status summaries
    event_tmp = event_log[['Event', 'Duration']]
    event_tmp = event_tmp.query('Event != "Signed in"')
    event_tmp.Duration = pd.to_numeric(event_tmp.Duration)
    summary_sts = event_tmp.groupby('Event').describe()
    summary_cnt = summary_sts.iloc[:, 0] / n_vh

    # utilization rates
    event_log['Duration'] = pd.to_numeric(event_log['Duration'])
    utilization_vh_avg = np.sum(event_log[(event_log['Event']=='Teleoperated') | (event_log['Event']=='Takeover')]['Duration']) / (n_mins * n_vh)
    utilization_to_avg = np.sum(event_log.query('Event!="Idle"')['Duration']) / (n_mins * n_to)

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

    # utilization + summary
    summary_utl = pd.DataFrame({'AVG_vehicle_utilization': np.round(utilization_vh_avg, 2),
                                'AVG_TO_utilization': np.round(utilization_to_avg, 2),
                                'AVG_Q_time_per_vehicle': np.round(queues_vh_time_avg, 2),
                                'AVG_Q_time_per_queue': np.round(np.mean(wait_times), 2),
                                'MAX_Q_time_per_queue': np.round(np.max(wait_times), 2),
                                'AVG_Q_length': np.round(queues_to_leng_avg, 2),
                                'Max_Q_length': queues_to_leng_max}, index=[0])

    # save main stats
    summary_utl.to_csv(output_dir + '/R_{}'.format(replication_no) + '_summary_utilization.csv', index=False)
    summary_sts.to_csv(output_dir + '/R_{}'.format(replication_no) + '_summary_status.csv')
    summary_cnt.to_csv(output_dir + '/R_{}'.format(replication_no) + '_summary_counts.csv')
    summary_qus.to_csv(output_dir + '/R_{}'.format(replication_no) + '_summary_queues.csv')

    # save detailed stats (only for the first replication)
    if replication_no == 1:
        event_log.to_csv(output_dir + '/R_{}'.format(replication_no) + '_events.csv', index=False)
        queues_df.to_csv(output_dir + '/R_{}'.format(replication_no) + '_queues.csv', index_label='Simulation_time')
        states_vh_df.to_csv(output_dir + '/R_{}'.format(replication_no) + '_states_vh.csv',
                            index_label='Simulation_time')
        states_to_df.to_csv(output_dir + '/R_{}'.format(replication_no) + '_states_to.csv',
                            index_label='Simulation_time')
        # plot final results and save graphs
        visualize.plot_results(states_vh_df, states_to_df, queues_df, output_dir, replication_no, n_vh, n_to, time_up)

    return summary_utl, summary_sts, summary_cnt, summary_qus, (states_vh_df.index[-1] - states_vh_df.index[0]), tour_completion, distance_completion


if __name__ == "__main__":

    # parameters
    runs, proportion, tour_lens, tour_begins, to2v_ratios, takeover_times, max_to_duration, rest_short, rest_long = parameters()

    # batch scenario runs
    for tour_len in tour_lens:
        for tour_begin in tour_begins:

            # select relevant tours based on scenario parameters
            preprocess.select_tours(tour_len, tour_begin, runs, proportion)

            for to2v_ratio in to2v_ratios:
                for takeover_time in takeover_times:

                    # create output directory (if it does not exist already)
                    output_dir = 'Output/' + \
                                 'tl-{}'.format(tour_len) + \
                                 '_tb-{}'.format(tour_begin) + \
                                 '_to2v-{:.2f}'.format(to2v_ratio) + \
                                 '_su-{}'.format(takeover_time) + \
                                 '_R-{}'.format(runs)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # record simulation run time
                    Begin = datetime.now()

                    # replications
                    print('********************')
                    print('Scenario parameters:')
                    print('********************')
                    print('Number of replications: {}'.format(runs))
                    print('Tour proportion: {}'.format(proportion))
                    print('Tour begin time: {}:00'.format(tour_begin))
                    print('Max tour length: {} hours'.format(tour_len))
                    print('Teleoperator to vehicle ratio: {}'.format(to2v_ratio))
                    print('Teleoperator takeover time: {} minute(s)'.format(takeover_time))
                    print('**************************')
                    print('Simulation in progress...')
                    for r in range(runs):
                        # run data preprocessing and return simulation input
                        print('Data preprocessing...')
                        n_vh, act_seq, act_dist, begin_times, to_total = preprocess.simulation_input(r, takeover_time)
                        n_to = int(round(n_vh * to2v_ratio))

                        # run simulation
                        print('Running replication {}'.format(r + 1))
                        utl, sts, cnt, qus, srt, cmpt, cmpd = run_simulation(r + 1, output_dir, runs, n_vh, n_to,
                                                                             takeover_time, act_seq, act_dist,
                                                                             begin_times, max_to_duration, rest_short,
                                                                             rest_long, tour_begin, tour_len, to_total)

                        # record stats
                        if r == 0:
                            utilizations = utl
                            statuses = sts.transpose().reset_index()
                            counts = pd.DataFrame(cnt).transpose().reset_index()
                            queues = pd.DataFrame(qus).transpose().reset_index()
                            times = [srt]
                            completion = np.array([cmpt, cmpd])
                        else:
                            utilizations = utilizations.append(utl, ignore_index=True)
                            statuses = statuses.append(sts.transpose().reset_index(), ignore_index=True)
                            counts = counts.append(pd.DataFrame(cnt).transpose().reset_index(), ignore_index=True)
                            queues = queues.append(pd.DataFrame(qus).transpose().reset_index(), ignore_index=True)
                            times.append(srt)
                            completion = np.vstack([completion, np.array([cmpt, cmpd])])

                    # report simulation run time
                    print('Simulation run time for {} run(s): '.format(runs))
                    print(datetime.now() - Begin)
                    print('Replication run time (including data preprocessing): ')
                    print((datetime.now() - Begin) / runs)
                    print('---------------------------------------------------')
                    print('---------------------------------------------------\n')

                    # save summary stats
                    report.stats_summary(utilizations, statuses, counts, queues, times, completion, output_dir)

    # create plots to show tradeoffs between queue times and TO2V ratios (across scenarios)
    print('Just making some final plots...')
    report.tradeoff_plots(runs, tour_lens, tour_begins, to2v_ratios, takeover_times)
    # remove temp input files
    for r in range(runs):
        os.remove('Input/Tours_filtered_S' + str(r) + '.csv')

    print('Done!')
