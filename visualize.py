import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from math import *

warnings.filterwarnings("ignore")
plt.style.use("ggplot")


def plot_results(states_vh_df, states_to_df, queues_df, output_dir, replication_no, n_vh, n_to):
    plt.close("all")

    duration = states_vh_df.index[-1]

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()

    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax4 = fig4.add_subplot(1, 1, 1)

    ax1.set(xlim=[-0.5, duration])
    ax2.set(xlim=[-0.5, duration])
    ax3.set(xlim=[-0.5, duration])
    ax4.set(xlim=[-0.5, duration])

    ax1.set(ylim=[-n_vh*0.05, n_vh + (n_vh*0.05)])
    ax2.set(ylim=[-(n_vh - n_to)*0.05, n_vh - n_to + (n_vh - n_to)*0.05])
    ax3.set(ylim=[-n_vh*0.05, n_vh + (n_vh*0.05)])
    ax4.set(ylim=[-n_to*0.05, n_to + (n_to*0.05)])

    ax1.hlines(y=n_vh, colors='gray', linestyles='--', xmin=-0.5, xmax=duration, label='Fleet size')
    ax3.hlines(y=n_vh, colors='gray', linestyles='--', xmin=-0.5, xmax=duration, label='Fleet size')
    ax4.hlines(y=n_to, colors='gray', linestyles='--', xmin=-0.5, xmax=duration, label='Available TO')

    ax1.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 10})
    ax2.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 10})
    ax3.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 10})
    ax4.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 10})

    ax1.set_ylabel('# Moving vehicles', fontdict={'fontsize': 10})
    ax2.set_ylabel('TO Queue length', fontdict={'fontsize': 10})
    ax3.set_ylabel('# Vehicles', fontdict={'fontsize': 10})
    ax4.set_ylabel('# TO', fontdict={'fontsize': 10})

    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax4.tick_params(axis='both', which='major', labelsize=8)

    lines1 = ax1.plot(states_vh_df['Moving'], 'royalblue')
    lines2 = ax2.plot(queues_df.iloc[:, 1], 'm')
    lines3 = ax3.plot(states_vh_df.iloc[:, 1:-1], ['r', 'm', 'royalblue'])
    lines4 = ax4.plot(states_to_df)

    lines1[0].set_label('Moving')
    lines2[0].set_label('TO Queue')
    for i in range(len(lines3)):
        lines3[i].set_label(states_vh_df.columns[i + 1])
    for i in range(len(lines4)):
        lines4[i].set_label(states_to_df.columns[i])

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    fig1.set_size_inches(12, 6)
    fig2.set_size_inches(12, 6)
    fig3.set_size_inches(12, 6)
    fig4.set_size_inches(12, 6)

    fig1.savefig(output_dir + '/R_{0}'.format(replication_no) + '_states_vh_m.jpeg', dpi=800)
    fig2.savefig(output_dir + '/R_{0}'.format(replication_no) + '_queues.jpeg', dpi=800)
    fig3.savefig(output_dir + '/R_{0}'.format(replication_no) + '_states_vh.jpeg', dpi=800)
    fig4.savefig(output_dir + '/R_{0}'.format(replication_no) + '_states_to.jpeg', dpi=800)

    plt.close("all")


def plot_summary(states_vh_df, states_to_df, queues_df, output_dir, replication_no, runs, n_vh, n_to, setup_to):

    duration = states_vh_df.index[-1]

    fig = plt.figure()
    fig.patch.set_facecolor('orange')

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.set_facecolor('lightcyan')
    ax2.set_facecolor('darkslategray')
    ax3.set_facecolor('lightcyan')
    ax4.set_facecolor('darkslategray')

    ax1.set(xlim=[-0.5, duration])
    ax2.set(xlim=[-0.5, duration])
    ax3.set(xlim=[-0.5, duration])
    ax4.set(xlim=[-0.5, duration])

    ax1.set(ylim=[-n_vh*0.05, n_vh + (n_vh*0.05)])
    ax2.set(ylim=[-(n_vh - n_to)*0.05, n_vh - n_to + (n_vh - n_to)*0.05])
    ax3.set(ylim=[-n_vh*0.05, n_vh + (n_vh*0.05)])
    ax4.set(ylim=[-n_to*0.05, n_to + (n_to*0.05)])

    ax1.hlines(y=n_vh, colors='gray', linestyles='--', xmin=-0.5, xmax=duration, label='Fleet size')
    ax4.hlines(y=n_to, colors='whitesmoke', linestyles='--', xmin=-0.5, xmax=duration, label='Available TO')

    ax1.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 8})
    ax2.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 8})
    ax3.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 8})
    ax4.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 8})

    ax1.set_ylabel('# Moving vehicles', fontdict={'fontsize': 9})
    ax2.set_ylabel('TO Queue length', fontdict={'fontsize': 9})
    ax3.set_ylabel('# Vehicles', fontdict={'fontsize': 9})
    ax4.set_ylabel('# TO', fontdict={'fontsize': 9})

    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    ax3.tick_params(axis='both', which='major', labelsize=7)
    ax4.tick_params(axis='both', which='major', labelsize=7)

    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax4.grid(False)

    fig.tight_layout()

    lines1 = ax1.plot(states_vh_df['Moving'], 'royalblue')
    lines2 = ax2.plot(queues_df.iloc[:, 1], 'm')
    lines3 = ax3.plot(states_vh_df.iloc[:, 1:-1], ['r', 'm', 'royalblue'])
    lines4 = ax4.plot(states_to_df)

    lines1[0].set_label('Moving')
    lines2[0].set_label('TO Queue')
    for i in range(len(lines3)):
        lines3[i].set_label(states_vh_df.columns[i + 1])
    for i in range(len(lines4)):
        lines4[i].set_label(states_to_df.columns[i])

    ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='small')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='small')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='small')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='small')

    # tight layout and maximize window
    fig.set_size_inches(12, 6)
    fig.tight_layout()
    fig.savefig(output_dir + '/R_{0}'.format(replication_no) + '_summary.jpeg', bbox_inches='tight', dpi=1000)


def visualize_series(moving, standing, drange, op_folder):

    ##################
    # visualize series
    label_moving = 'moving times ' + drange
    label_standing = 'standing times ' + drange

    # histograms
    fig, axs = plt.subplots(2)
    axs[0].hist(moving, bins=round(sqrt(len(moving)))*2, color='g', alpha=0.75, label=label_moving)
    axs[0].set(ylabel='Frequency')
    axs[0].set(xlim=[0, np.percentile(moving, 95)])
    axs[0].legend(loc="upper right")
    axs[1].hist(standing, bins=round(sqrt(len(standing)))*2, color='r', alpha=0.75, histtype='stepfilled', label=label_standing)
    axs[1].set(xlabel='Time')
    axs[1].set(ylabel='Frequency')
    axs[1].set(xlim=[0, np.percentile(standing, 95)])
    axs[1].legend(loc="upper right")
    plt.savefig(op_folder + 'Figures/hist_' + label_moving + '.jpeg', transparent=False)
    plt.show()

    # box plots
    colors = ['green', 'red']
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    bplot1 = ax1.boxplot(x=moving, vert=True, patch_artist=True)
    ax1.set_title(label_moving)
    for box in bplot1['boxes']:
        box.set(facecolor='green')
    bplot2 = ax2.boxplot(x=standing, vert=True, patch_artist=True)
    ax2.set_title(label_standing)
    for box in bplot2['boxes']:
        box.set(facecolor='red')
    plt.savefig(op_folder + 'Figures/box_' + label_moving + '.jpeg', transparent=False)
    plt.show()

    # seaborn distribution plots (with fitting line)
    fig, axs = plt.subplots(2)
    sns.distplot(ax=axs[0], x=moving, kde=True, bins=round(sqrt(len(moving)))*2, color='g', label=label_moving)
    axs[0].set(xlim=[0, np.percentile(moving, 95)])
    axs[0].legend(loc="upper right")
    sns.distplot(ax=axs[1], x=standing, kde=True, bins=round(sqrt(len(standing)))*2, color='r', label=label_standing)
    axs[1].set(xlabel='Time')
    axs[1].set(xlim=[0, np.percentile(standing, 95)])
    axs[1].legend(loc="upper right")
    plt.savefig(op_folder + 'Figures/dist_' + label_moving + '.jpeg', transparent=False)
    plt.show()


def visualize_patterns(patterns, op_folder, show_plot):

    ####################
    # visualize patterns

    yy = patterns['Count'].values
    xx = patterns['Pattern'].values
    xx = ['Pattern {0}'.format(i+1) for i in range(len(xx))]
    colors = ['green', 'y', 'blue', 'cyan', 'red']
    plt.barh(xx, yy, color=colors)
    plt.xticks(rotation=0)
    plt.xlabel('Count')
    plt.title('Recurring activity patterns')
    plt.tight_layout()
    plt.savefig(op_folder + 'Figures/activity_patterns' + '.jpeg', transparent=False)
    if show_plot:
        plt.show()
    plt.clf()
    plt.close()

