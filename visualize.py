import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')


def plot_results(states_vh_df, states_to_df, queues_df, output_dir, replication_no, n_vh, n_to):

    time_min = states_vh_df.index[0]
    time_max = states_vh_df.index[-1]

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()

    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax4 = fig4.add_subplot(1, 1, 1)

    ax1.hlines(y=n_vh, colors='gray', linestyles='--', xmin=time_min, xmax=time_max, label='Fleet size')
    ax3.hlines(y=n_vh, colors='gray', linestyles='--', xmin=time_min, xmax=time_max, label='Fleet size')
    ax4.hlines(y=n_to, colors='gray', linestyles='--', xmin=time_min, xmax=time_max, label='Available TO')

    ax1.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 10})
    ax2.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 10})
    ax3.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 10})
    ax4.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 10})

    ax1.set_ylabel('Number of vehicles', fontdict={'fontsize': 10})
    ax2.set_ylabel('TO Queue length', fontdict={'fontsize': 10})
    ax3.set_ylabel('Number of Vehicles', fontdict={'fontsize': 10})
    ax4.set_ylabel('Number of TO', fontdict={'fontsize': 10})

    ax1.tick_params(axis='both', which='both', labelsize=8)
    ax2.tick_params(axis='both', which='both', labelsize=8)
    ax3.tick_params(axis='both', which='both', labelsize=8)
    ax4.tick_params(axis='both', which='both', labelsize=8)

    lines1 = ax1.plot(states_vh_df['Teleoperated'], 'royalblue')
    lines2 = ax2.plot(queues_df.iloc[:, 0], 'm')
    lines3 = ax3.plot(states_vh_df.iloc[:, 1:-1][::-1])
    lines4 = ax4.plot(states_to_df, alpha=0.7)

    lines1[0].set_label('Teleoperated')
    lines2[0].set_label('TO Queue')
    for i in range(len(lines3)):
        lines3[i].set_label(states_vh_df.columns[i + 1])
    for i in range(len(lines4)):
        lines4[i].set_label(states_to_df.columns[i])

    ## vehicles
    # TO queue
    lines3[0].set_color('m')
    # takeover
    lines3[1].set_color('orange')
    # moving (teleoperated)
    lines3[2].set_color('royalblue')

    ## TOs
    # idle
    lines4[0].set_color('green')
    # busy
    lines4[1].set_color('royalblue')
    # resting
    lines4[2].set_color('red')
    # takeover
    lines4[3].set_color('orange')

    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize='small')

    fig1.set_size_inches(12, 6)
    fig2.set_size_inches(12, 6)
    fig3.set_size_inches(12, 6)
    fig4.set_size_inches(12, 6)

    fig1.savefig(output_dir + '/R_{0}'.format(replication_no) + '_states_vh_m.jpeg', bbox_inches="tight", dpi=800)
    fig2.savefig(output_dir + '/R_{0}'.format(replication_no) + '_queues.jpeg', bbox_inches="tight", dpi=800)
    fig3.savefig(output_dir + '/R_{0}'.format(replication_no) + '_states_vh.jpeg', bbox_inches="tight", dpi=800)
    fig4.savefig(output_dir + '/R_{0}'.format(replication_no) + '_states_to.jpeg', bbox_inches="tight", dpi=800)

    plt.close("all")


def plot_summary(states_vh_df, states_to_df, queues_df, output_dir, replication_no, runs, n_vh, n_to, setup_to):

    time_min = states_vh_df.index[0]
    time_max = states_vh_df.index[-1]

    fig = plt.figure()
    fig.patch.set_facecolor('gold')

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.set_facecolor('white')
    ax2.set_facecolor('darkslategray')
    ax3.set_facecolor('white')
    ax4.set_facecolor('darkslategray')

    ax1.hlines(y=n_vh, xmin=time_min, xmax=time_max, colors='gray', linestyles='--', label='Fleet size')
    ax4.hlines(y=n_to, xmin=time_min, xmax=time_max, colors='whitesmoke', linestyles='--', label='Available TO')

    ax1.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 8})
    ax2.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 8})
    ax3.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 8})
    ax4.set_xlabel('Simulation Time (minutes)', fontdict={'fontsize': 8})

    ax1.set_ylabel('Number of Teleoperated vehicles', fontdict={'fontsize': 9})
    ax2.set_ylabel('TO Queue length', fontdict={'fontsize': 9})
    ax3.set_ylabel('Number of Vehicles', fontdict={'fontsize': 9})
    ax4.set_ylabel('Number of TO', fontdict={'fontsize': 9})

    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    ax3.tick_params(axis='both', which='major', labelsize=7)
    ax4.tick_params(axis='both', which='major', labelsize=7)

    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax4.grid(False)

    fig.tight_layout()

    lines1 = ax1.plot(states_vh_df['Teleoperated'], 'royalblue')
    lines2 = ax2.plot(queues_df.iloc[:, 0], 'm')
    lines3 = ax3.plot(states_vh_df.iloc[:, 1:-1])
    lines4 = ax4.plot(states_to_df)

    lines1[0].set_label('Teleoperated')
    lines2[0].set_label('TO Queue')
    for i in range(len(lines3)):
        lines3[i].set_label(states_vh_df.columns[i + 1])
    for i in range(len(lines4)):
        lines4[i].set_label(states_to_df.columns[i])

    ## vehicles
    # TO queue
    lines3[0].set_color('m')
    # takeover
    lines3[1].set_color('orange')
    # moving (teleoperated)
    lines3[2].set_color('royalblue')

    ## TOs
    # idle
    lines4[0].set_color('green')
    # busy
    lines4[1].set_color('royalblue')
    # resting
    lines4[2].set_color('red')
    # takeover
    lines4[3].set_color('orange')

    ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='small')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='small')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='small')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.35, 1), fontsize='small')

    # tight layout and maximize window
    fig.set_size_inches(12, 6)
    fig.tight_layout()
    fig.savefig(output_dir + '/R_{0}'.format(replication_no) + '_summary.jpeg', bbox_inches='tight', dpi=1000)


