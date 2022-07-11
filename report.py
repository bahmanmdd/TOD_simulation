import pandas as pd


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

