"""
cross-scenario and meta-analysis reports for simulation of teleoperated driving in shipping processes
created by: Bahman Madadi
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')


def stats_summary(utilizations, statuses, counts, queues, times, completion, output_dir):

    utilizations.index = utilizations.index + 1
    summary_utilization = utilizations.describe()
    summary_status = statuses.groupby('level_1').mean()
    summary_status.index = summary_status.index.str.strip()
    summary_status = summary_status.reindex(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    summary_count = counts.describe()
    summary_queue = queues.describe()
    summary_times = pd.Series(times, name='Makespan').describe()
    makespan_dist = pd.Series(times, name='Makespan')
    completion_dist = pd.DataFrame(completion, columns=['tour_completion', 'distance_completion', 'delay'])

    summary_utilization.transpose().rename(columns={'count': 'replications'}, inplace=False).to_excel(
        output_dir + '/R_0_summary_utilization.xlsx')
    summary_status.transpose().to_excel(output_dir + '/R_0_summary_status.xlsx')
    summary_count.transpose().rename(columns={'count': 'replications'}, inplace=False).to_excel(
        output_dir + '/R_0_summary_count.xlsx')
    summary_queue.to_excel(output_dir + '/R_0_summary_queues.xlsx')
    summary_times.to_excel(output_dir + '/R_0_summary_makespan.xlsx')
    makespan_dist.to_csv(output_dir + '/R_0_dist_makespan.csv')
    completion_dist.to_csv(output_dir + '/R_0_dist_completion.csv', index_label='Replication')
    utilizations.to_csv(output_dir + '/R_0_full_utilization.csv', index_label='Replication')


def tradeoff_plots(runs, tour_lens, tour_begins, to2v_ratios, takeover_times):

    output_dir = 'Output/0 Ratios'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = pd.DataFrame(columns=['tour_len',
                                 'tour_begin',
                                 'Replication',
                                 'TO_takeover_time',
                                 'TO2vehicle_ratio',
                                 'AVG_queue_time',
                                 'Max_queue_time',
                                 'AVG_queue_per_vehicle',
                                 'Makespan',
                                 'tour_completion',
                                 'distance_completion',
                                 'delay'])

    for tour_len in tour_lens:
        for tour_begin in tour_begins:
            for tot in takeover_times:
                for tov in to2v_ratios:
                    name = 'Output/' + \
                           'tl-{}'.format(tour_len) + \
                           '_tb-{}'.format(tour_begin) + \
                           '_to2v-{:.2f}'.format(tov) + \
                           '_su-{}'.format(tot) + \
                           '_R-{}'.format(runs)

                    msp_temp = pd.read_csv(name + '/R_0_dist_makespan.csv')
                    kpi_temp = pd.read_csv(name + '/R_0_full_utilization.csv')
                    cmp_temp = pd.read_csv(name + '/R_0_dist_completion.csv')

                    row = {'tour_len': [tour_len] * runs, 'tour_begin': [tour_begin] * runs,
                           'Replication': np.array([*range(runs)]) + 1,
                           'TO_takeover_time': [tot] * runs, 'TO2vehicle_ratio': [tov] * runs,
                           'AVG_queue_time': kpi_temp['AVG_Q_time_per_queue'].values,
                           'Max_queue_time': kpi_temp['MAX_Q_time_per_queue'].values,
                           'AVG_queue_per_vehicle': kpi_temp['AVG_Q_time_per_vehicle'].values,
                           'Makespan': msp_temp['Makespan'].values,
                           'tour_completion': cmp_temp['tour_completion'].values,
                           'distance_completion': cmp_temp['distance_completion'].values,
                           'delay': cmp_temp['delay'].values}

                    data_new = pd.DataFrame(row)
                    data = pd.concat([data, data_new], ignore_index=True)

            # filter data for each tour combination
            name_temp = 'tl-' + str(tour_len) + '_tb-' + str(tour_begin)
            data_temp = data[(data['tour_len'] == tour_len) & (data['tour_begin'] == tour_begin)]
            # data_temp.to_excel(output_dir + '/' + name_temp + '_ratios.xlsx', index=False)

            # plot graphs for each tour combination
            sns.lineplot(data=data_temp, x="TO2vehicle_ratio", y="AVG_queue_time", hue="TO_takeover_time",
                         palette=sns.color_palette("bright", n_colors=data_temp["TO_takeover_time"].nunique()))
            plt.xlabel('Teleoperator-to-vehicle ratio')
            plt.ylabel('Average queue duration (minutes)')
            plt.savefig(output_dir + '/' + name_temp + '_avg_q_times.jpeg', dpi=600)
            plt.close()

            sns.lineplot(data=data_temp, x="TO2vehicle_ratio", y="Max_queue_time", hue="TO_takeover_time",
                         palette=sns.color_palette("bright", n_colors=data_temp["TO_takeover_time"].nunique()))
            plt.xlabel('Teleoperator-to-vehicle ratio')
            plt.ylabel('Max queue duration (minutes)')
            plt.savefig(output_dir + '/' + name_temp + '_max_q_times.jpeg', dpi=600)
            plt.close()

            sns.lineplot(data=data_temp, x="TO2vehicle_ratio", y="AVG_queue_per_vehicle", hue="TO_takeover_time",
                         palette=sns.color_palette("bright", n_colors=data_temp["TO_takeover_time"].nunique()))
            plt.xlabel('Teleoperator-to-vehicle ratio')
            plt.ylabel('Average wait time per vehicle (minutes)')
            plt.savefig(output_dir + '/' + name_temp + '_avg_vq_times.jpeg', dpi=600)
            plt.close()

            sns.lineplot(data=data_temp, x="TO2vehicle_ratio", y="Makespan", hue="TO_takeover_time",
                         palette=sns.color_palette("bright", n_colors=data_temp["TO_takeover_time"].nunique()))
            # plt.hlines(y=(tour_begin+tour_len)*60, colors='black', linestyles='--', label='Baseline makespan',
            #            xmin=np.min(data_temp['TO2vehicle_ratio']),
            #            xmax=np.max(data_temp['TO2vehicle_ratio']))
            plt.title('Makespan vs Teleoperator-to-vehicle ratio')
            plt.xlabel('Teleoperator-to-vehicle ratio')
            plt.ylabel('Makespan (minutes)')
            # plt.legend()
            plt.savefig(output_dir + '/' + name_temp + '_total-makespan.jpeg', dpi=600)
            plt.close()

            sns.lineplot(data=data_temp, x="TO2vehicle_ratio", y="tour_completion", hue="TO_takeover_time",
                         palette=sns.color_palette("bright", n_colors=data_temp["TO_takeover_time"].nunique()))
            plt.title('Tour completion rate within the baseline makespan')
            plt.xlabel('Teleoperator-to-vehicle ratio')
            plt.ylabel('Tour completion rate')
            # plt.ylim([0, 1])
            plt.savefig(output_dir + '/' + name_temp + '_completion-tour.jpeg', dpi=600)
            plt.close()

            sns.lineplot(data=data_temp, x="TO2vehicle_ratio", y="distance_completion", hue="TO_takeover_time",
                         palette=sns.color_palette("bright", n_colors=data_temp["TO_takeover_time"].nunique()))
            plt.title('Distance completion rate within the baseline makespan')
            plt.xlabel('Teleoperator-to-vehicle ratio')
            plt.ylabel('Distance completion rate')
            # plt.ylim([0, 1])
            plt.savefig(output_dir + '/' + name_temp + '_completion-distance.jpeg', dpi=600)
            plt.close()

            sns.lineplot(data=data_temp, x="TO2vehicle_ratio", y="delay", hue="TO_takeover_time",
                         palette=sns.color_palette("bright", n_colors=data_temp["TO_takeover_time"].nunique()))
            plt.title('Average trip delay compared to the baseline')
            plt.xlabel('Teleoperator-to-vehicle ratio')
            plt.ylabel('Average trip delay (ratio compared to the baseline)')
            # plt.ylim([0, 1])
            plt.savefig(output_dir + '/' + name_temp + '_completion-delay.jpeg', dpi=600)
            plt.close()

    data = data[['tour_len',
                 'tour_begin',
                 'Replication',
                 'TO_takeover_time',
                 'TO2vehicle_ratio',
                 'AVG_queue_time',
                 'Max_queue_time',
                 'AVG_queue_per_vehicle',
                 'Makespan',
                 'tour_completion',
                 'distance_completion',
                 'delay']]

    data.to_excel(output_dir + '/Full_ratios.xlsx', index=False)
