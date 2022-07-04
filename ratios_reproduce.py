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

df = pd.read_excel('Output/0 Ratios/Full_ratios.xlsx')

# separate series with queries
df_cases = {}
for case in cases:
    df_cases[case] = df.query('Case == @case')
    df_avg = df_cases[case].pivot(index='TO-to-v ratio', columns='TO setup time', values='AVG queue time')
    df_max = df_cases[case].pivot(index='TO-to-v ratio', columns='TO setup time', values='Max queue time')
    df_vav = df_cases[case].pivot(index='TO-to-v ratio', columns='TO setup time', values='AVG queue perv')

    df_avg.plot()
    plt.xlabel('Teleoperator-to-vehicle ratio', fontweight='bold')
    plt.ylabel('Average queue duration (minutes)', fontweight='bold')
    plt.savefig('Output/0 Ratios/' + case + '_avg_q_times.jpeg', dpi=1000)
    plt.close()

    df_max.plot()
    plt.xlabel('Teleoperator-to-vehicle ratio', fontweight='bold')
    plt.ylabel('Max queue duration (minutes)', fontweight='bold')
    plt.savefig('Output/0 Ratios/' + case + '_max_q_times.jpeg', dpi=1000)
    plt.close()

    df_vav.plot()
    plt.xlabel('Teleoperator-to-vehicle ratio', fontweight='bold')
    plt.ylabel('Average wait time per vehicle (minutes)', fontweight='bold')
    plt.savefig('Output/0 Ratios/' + case + '_avg_vq_times.jpeg', dpi=1000)
    plt.close()


# sns.lineplot(data=df_cases[case], x='TO-to-v ratio', y='AVG queue time', hue='TO setup time')

