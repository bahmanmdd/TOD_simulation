import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-paper')


def stats_summary(utilizations, statuses, counts, queues, times, output_dir):

    utilizations.index = utilizations.index + 1
    summary_utilization = utilizations.describe()
    summary_status = statuses.groupby('level_1').mean()
    summary_status.index = summary_status.index.str.strip()
    summary_status = summary_status.reindex(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    summary_count = counts.describe()
    summary_queue = queues.describe()
    summary_times = pd.Series(times, name='Makespan').describe()
    makespan_dist = pd.Series(times, name='Makespan')

    summary_utilization.transpose().rename(columns={'count': 'replications'}, inplace=False).to_excel(output_dir + '/R_0_summary_utilization.xlsx')
    summary_status.transpose().to_excel(output_dir + '/R_0_summary_status.xlsx')
    summary_count.transpose().rename(columns={'count': 'replications'}, inplace=False).to_excel(output_dir + '/R_0_summary_count.xlsx')
    summary_queue.to_excel(output_dir + '/R_0_summary_queues.xlsx')
    summary_times.to_excel(output_dir + '/R_0_summary_makespan.xlsx')
    makespan_dist.to_csv(output_dir + '/R_0_dist_makespan.csv')
    utilizations.to_csv(output_dir + '/R_0_full_utilization.csv', index_label='Replication')


def tradeoff_plots(to2v_ratio_list, carrier_proportion_list, takeover_time_list, runs):

    output_dir = 'Output/0 Ratios'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(columns={'carrier_proportion',
                               'TO_takeover_time',
                               'TO2vehicle_ratio',
                               'AVG_queue_time',
                               'Max_queue_time',
                               'AVG_queue_per_vehicle'})

    for cp in carrier_proportion_list:
        ci_data = pd.DataFrame(columns=['TO takeover time', 'TO2vehicle ratio', 'Replication', 'Makespan'])
        for tot in takeover_time_list:
            for tov in to2v_ratio_list:

                name = 'Output/cp-{:.3f}'.format(cp) + '_to2v-{:.2f}'.format(tov) + '_su-{}'.format(tot) + '_R-{}'.format(runs)

                ms_temp = pd.read_csv(name + '/R_0_dist_makespan.csv')
                df_temp = pd.read_excel(name + '/R_0_summary_utilization.xlsx')
                df_temp = df_temp.set_index('Unnamed: 0')

                row = {'carrier_proportion': cp, 'TO_takeover_time': tot, 'TO2vehicle_ratio': tov,
                       'AVG_queue_time': df_temp.loc['AVG_Q_time_per_queue', 'mean'],
                       'Max_queue_time': df_temp.loc['MAX_Q_time_per_queue', 'mean'],
                       'AVG_queue_per_vehicle': df_temp.loc['AVG_Q_time_per_vehicle', 'mean']}

                df = df.append(row, ignore_index=True)

                ci_temp = {'TO takeover time': [tot]*runs, 'TO2vehicle ratio': [tov]*runs, 'Replication': [*range(runs)], 'Makespan': ms_temp.Makespan.values}
                ci_data_new = pd.DataFrame(ci_temp)
                ci_data = pd.concat([ci_data, ci_data_new], ignore_index=True)

        sns.lineplot(data=ci_data, x="TO2vehicle ratio", y="Makespan", hue="TO takeover time")
        plt.title('Makespan vs Teleoperator-to-vehicle ratio')
        plt.xlabel('Teleoperator-to-vehicle ratio')
        plt.ylabel('Makespan (minutes)')
        # plt.xlim([0, 1])
        plt.savefig(output_dir + '/cp_' + str(cp) + '_makespan.jpeg', dpi=800)
        plt.close()

    df = df[['carrier_proportion',
             'TO_takeover_time',
             'TO2vehicle_ratio',
             'AVG_queue_time',
             'Max_queue_time',
             'AVG_queue_per_vehicle']]

    df.to_excel(output_dir + '/Full_ratios.xlsx', index=False)

    # separate series with queries
    df_carrier_proportions = {}
    for cp in carrier_proportion_list:
        df_carrier_proportions[cp] = df.query('carrier_proportion == @cp')
        df_carrier_proportions[cp].to_excel(output_dir + '/cp_' + str(cp) + '_ratios.xlsx', index=False)
        df_avg = df_carrier_proportions[cp].pivot(index='TO2vehicle_ratio', columns='TO_takeover_time', values='AVG_queue_time')
        df_max = df_carrier_proportions[cp].pivot(index='TO2vehicle_ratio', columns='TO_takeover_time', values='Max_queue_time')
        df_vav = df_carrier_proportions[cp].pivot(index='TO2vehicle_ratio', columns='TO_takeover_time', values='AVG_queue_per_vehicle')

        df_avg.plot()
        plt.xlabel('Teleoperator-to-vehicle ratio')
        plt.ylabel('Average queue duration (minutes)')
        plt.xlim([0, 1])
        plt.savefig(output_dir + '/cp_' + str(cp) + '_avg_q_times.jpeg', dpi=800)
        plt.close()

        df_max.plot()
        plt.xlabel('Teleoperator-to-vehicle ratio')
        plt.ylabel('Max queue duration (minutes)')
        plt.xlim([0, 1])
        plt.savefig(output_dir + '/cp_' + str(cp) + '_max_q_times.jpeg', dpi=800)
        plt.close()

        df_vav.plot()
        plt.xlabel('Teleoperator-to-vehicle ratio')
        plt.ylabel('Average wait time per vehicle (minutes)')
        plt.xlim([0, 1])
        plt.savefig(output_dir + '/cp_' + str(cp) + '_avg_vq_times.jpeg', dpi=800)
        plt.close()



