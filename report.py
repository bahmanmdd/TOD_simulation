import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use("ggplot")


def stats_summary(utilizations, statuses, counts, queues, times, output_dir):

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


def tradeoff_plots(to2v_ratio_list, carrier_proportion_list, takeover_time_list):

    df = pd.DataFrame(columns={'carrier_proportion',
                               'TO_takeover_time',
                               'TO2vehicle_ratio',
                               'AVG_queue_time',
                               'Max_queue_time',
                               'AVG_queue_per_vehicle'})

    for cp in carrier_proportion_list:
        for tov in to2v_ratio_list:
            for tot in takeover_time_list:

                name = 'Output/' + '_cp-{:.3f}'.format(cp) + '_to2v-{:.2f}'.format(tov) + '_su-{}'.format(tot)\
                       + '_R-1/R_0_summary_utilization.xlsx'

                df_temp = pd.read_excel(name)
                df_temp = df_temp.set_index('Unnamed: 0')

                row = {'carrier_proportion': cp, 'TO_takeover_time': tot, 'TO2vehicle_ratio': tov,
                       'AVG_queue_time': df_temp.loc['AVG_Q_time_per_queue', 'mean'],
                       'Max_queue_time': df_temp.loc['MAX_Q_time_per_queue', 'mean'],
                       'AVG_queue_per_vehicle': df_temp.loc['AVG_Q_time_per_vehicle', 'mean']}

                df = df.append(row, ignore_index=True)

    df = df[['carrier_proportion',
             'TO_takeover_time',
             'TO2vehicle_ratio',
             'AVG_queue_time',
             'Max_queue_time',
             'AVG_queue_per_vehicle']]
    df.to_excel('Output/0 Ratios/Full_ratios.xlsx', index=False)

    # separate series with queries
    df_carrier_proportions = {}
    for cp in carrier_proportion_list:
        df_carrier_proportions[cp] = df.query('carrier_proportion == @cp')
        df_carrier_proportions[cp].to_excel('Output/0 Ratios/cp_' + cp + '_ratios.xlsx', index=False)
        df_avg = df_carrier_proportions[cp].pivot(index='TO2vehicle_ratio', columns='TO_takeover_time', values='AVG_queue_time')
        df_max = df_carrier_proportions[cp].pivot(index='TO2vehicle_ratio', columns='TO_takeover_time', values='Max_queue_time')
        df_vav = df_carrier_proportions[cp].pivot(index='TO2vehicle_ratio', columns='TO_takeover_time', values='AVG_queue_per_vehicle')

        df_avg.plot()
        plt.xlabel('Teleoperator-to-vehicle ratio')
        plt.ylabel('Average queue duration (minutes)')
        plt.savefig('Output/0 Ratios/cp_' + cp + '_avg_q_times.jpeg', dpi=800)
        plt.close()

        df_max.plot()
        plt.xlabel('Teleoperator-to-vehicle ratio')
        plt.ylabel('Max queue duration (minutes)')
        plt.savefig('Output/0 Ratios/cp_' + cp + '_max_q_times.jpeg', dpi=800)
        plt.close()

        df_vav.plot()
        plt.xlabel('Teleoperator-to-vehicle ratio')
        plt.ylabel('Average wait time per vehicle (minutes)')
        plt.savefig('Output/0 Ratios/cp_' + cp + '_avg_vq_times.jpeg', dpi=800)
        plt.close()



