# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import scipy
import scipy.stats
import os

# suppress warnings
warnings.filterwarnings("ignore")
# plt.ioff()
plt.style.use("ggplot")


cases = ['ACR', 'RSN', 'VOD_B', 'VOD_C', 'VOD_T', 'ALL']

to2vs = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
su_to = [0, 2, 5]
df = pd.DataFrame(columns={'Case', 'TO setup time', 'TO-to-v ratio', 'AVG queue time', 'Max queue time', 'AVG queue perv'})

for case in cases:
    for tov in to2vs:
        if case == 'ACR':
            nv = 37
        elif case == 'RSN':
            nv = 69
        elif case == 'VOD_B':
            nv = 29
        elif case == 'VOD_C':
            nv = 4
        elif case == 'VOD_T':
            nv = 25
        elif case == 'ALL':
            nv = 450
        nt = int(round(nv * tov))
        for sut in su_to:
            name = 'Output/' + case + '_v-{0}'.format(nv) + '_to-{0}'.format(nt) + '_su-{0}'.format(sut) + '_R-30/R_0_summary_utilization.xlsx'
            df_temp = pd.read_excel(name)
            df_temp = df_temp.set_index('Unnamed: 0')

            row = {'Case': case, 'TO setup time': sut, 'TO-to-v ratio': tov,
                   'AVG queue time': df_temp.loc['AVG_Q_time_per_queue', 'mean'],
                   'Max queue time': df_temp.loc['MAX_Q_time_per_queue', 'mean'],
                   'AVG queue perv': df_temp.loc['AVG_Q_time_per_vehicle', 'mean']}
            df = df.append(row, ignore_index=True)
df = df[['Case', 'TO-to-v ratio', 'TO setup time', 'AVG queue time', 'Max queue time', 'AVG queue perv']]
df.to_excel('Output/0 Ratios/Full_ratios.xlsx', index=False)
# separate series with queries
df_cases = {}
for case in cases:
    df_cases[case] = df.query('Case == @case')
    df_cases[case].to_excel('Output/0 Ratios/' + case + '_ratios.xlsx', index=False)
    df_avg = df_cases[case].pivot(index='TO-to-v ratio', columns='TO setup time', values='AVG queue time')
    df_max = df_cases[case].pivot(index='TO-to-v ratio', columns='TO setup time', values='Max queue time')
    df_vav = df_cases[case].pivot(index='TO-to-v ratio', columns='TO setup time', values='AVG queue perv')

    df_avg.plot()
    plt.xlabel('Teleoperator-to-vehicle ratio')
    plt.ylabel('Average queue duration (minutes)')
    plt.savefig('Output/0 Ratios/' + case + '_avg_q_times.jpeg', dpi=800)
    plt.close()

    df_max.plot()
    plt.xlabel('Teleoperator-to-vehicle ratio')
    plt.ylabel('Max queue duration (minutes)')
    plt.savefig('Output/0 Ratios/' + case + '_max_q_times.jpeg', dpi=800)
    plt.close()

    df_vav.plot()
    plt.xlabel('Teleoperator-to-vehicle ratio')
    plt.ylabel('Average wait time per vehicle (minutes)')
    plt.savefig('Output/0 Ratios/' + case + '_avg_vq_times.jpeg', dpi=800)
    plt.close()


# sns.lineplot(data=df_cases[case], x='TO-to-v ratio', y='AVG queue time', hue='TO setup time')

