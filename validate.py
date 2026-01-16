import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import glob
import re

# --- CONFIGURATION ---
OUTPUT_DIR = 'Output'
KPI_FILE = 'R_0_dist_makespan.csv'
TARGET_KPI = 'Makespan'
KPI_LABEL = 'Makespan (min)'
CONFIDENCE_LEVEL = 0.95

# Use a nicer style
plt.style.use('seaborn-v0_8-whitegrid')

# Output filenames
TABLE_FILENAME = 'Output/V_and_V_Makespan_Table.csv'


def parse_scenario_name(folder_name):
    """Extracts parameters from the standard folder structure."""
    match = re.search(r'to2v-([\d\.]+)', folder_name)
    if match:
        return float(match.group(1))
    return None


def load_all_scenarios():
    """Scans output directory and loads Makespan data."""
    scenarios = []
    folders = glob.glob(os.path.join(OUTPUT_DIR, '*R-5*'))

    if not folders:
        print("ERROR: No output folders found!")
        return []

    print(f"Found {len(folders)} scenario folders.")

    for folder in folders:
        file_path = os.path.join(folder, KPI_FILE)
        if not os.path.exists(file_path):
            continue

        ratio = parse_scenario_name(os.path.basename(folder))
        if ratio is None:
            continue

        try:
            df = pd.read_csv(file_path)
            # Handle possible missing column names by assuming column 0 if needed
            if TARGET_KPI in df.columns:
                data = df[TARGET_KPI].values
            elif len(df.columns) > 0:
                # Fallback if header is missing/different
                data = df.iloc[:, 0].values
            else:
                continue

            scenarios.append({
                'ratio': ratio,
                'data': data,
                'n': len(data)
            })
        except Exception as e:
            print(f"Skipping {folder}: {e}")

    scenarios.sort(key=lambda x: x['ratio'])
    return scenarios


def calculate_stats(data):
    """Calculates comprehensive statistics for a dataset."""
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    std_err = st.sem(data)
    t_score = st.t.ppf((1 + CONFIDENCE_LEVEL) / 2, n - 1)
    ci_half_width = std_err * t_score
    rel_error = (ci_half_width / mean * 100) if mean > 0.001 else 0.0
    return mean, std_dev, ci_half_width, rel_error


def generate_table(scenarios):
    """Generates and saves a statistical summary table with an Average row."""
    records = []
    all_means = []
    all_cis = []
    all_errs = []

    for sc in scenarios:
        mean, std, ci, rel_err = calculate_stats(sc['data'])
        records.append({
            'TO/V Ratio': sc['ratio'],
            'Replications': sc['n'],
            'Mean Makespan': round(mean, 2),
            'Std Dev': round(std, 2),
            '95% CI (+/-)': round(ci, 2),
            'Relative Error %': round(rel_err, 2)
        })
        all_means.append(mean)
        all_cis.append(ci)
        all_errs.append(rel_err)

    # Calculate Global Averages for the Table
    avg_row = {
        'TO/V Ratio': 'AVERAGE',
        'Replications': scenarios[0]['n'] if scenarios else 0,
        'Mean Makespan': round(np.mean(all_means), 2),
        'Std Dev': round(np.mean([r['Std Dev'] for r in records]), 2),
        '95% CI (+/-)': round(np.mean(all_cis), 2),
        'Relative Error %': round(np.mean(all_errs), 2)
    }
    records.append(avg_row)

    df_table = pd.DataFrame(records)
    print("\n--- STATISTICAL SUMMARY TABLE (MAKESPAN) ---")
    print(df_table.to_string(index=False))
    df_table.to_csv(TABLE_FILENAME, index=False)
    print(f"Table saved to {TABLE_FILENAME}")
    return df_table


def plot_single_convergence(data, title, filename, color):
    """Helper function to create one nice plot."""
    n = len(data)
    cum_means = []
    cum_cis = []

    for j in range(1, n + 1):
        subset = data[:j]
        m = np.mean(subset)
        se = st.sem(subset) if j > 1 else 0
        t = st.t.ppf((1 + CONFIDENCE_LEVEL) / 2, j - 1) if j > 1 else 0
        h = se * t
        cum_means.append(m)
        cum_cis.append(h)

    # Create Plot
    plt.figure(figsize=(7, 5))
    iterations = range(1, n + 1)

    # Plot Confidence Tunnel
    upper = [m + h for m, h in zip(cum_means, cum_cis)]
    lower = [m - h for m, h in zip(cum_means, cum_cis)]

    plt.fill_between(iterations, lower, upper, color=color, alpha=0.15, label='95% Confidence Interval')
    plt.plot(iterations, cum_means, marker='o', linestyle='-', color=color, linewidth=2.5, label='Cumulative Mean')

    # Styling
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Replications', fontsize=11)
    plt.ylabel(KPI_LABEL, fontsize=11)
    plt.xticks(iterations)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Stats Box
    final_mean = cum_means[-1]
    final_rel_err = (cum_cis[-1] / final_mean * 100)
    stats_text = f'Final Mean: {final_mean:.1f}\nRel. Error: {final_rel_err:.2f}%'
    plt.text(0.05, 0.9, stats_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.5'), fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()  # Show to user
    print(f"Saved: {filename}")


def plot_all_convergences(scenarios):
    """Generates 4 separate plots."""

    # 1. Identify specific ratios to plot
    unique_ratios = sorted(list(set(s['ratio'] for s in scenarios)))
    if len(unique_ratios) >= 3:
        target_ratios = [unique_ratios[0], 0.5, 1.0]  # Try to pick 0.3, 0.5, 1.0
        # Fallback if specific ones don't exist
        if 0.5 not in unique_ratios: target_ratios[1] = unique_ratios[len(unique_ratios) // 2]
        if 1.0 not in unique_ratios: target_ratios[2] = unique_ratios[-1]
    else:
        target_ratios = unique_ratios

    colors = ['#d62728', '#1f77b4', '#2ca02c']  # Red, Blue, Green

    # Plot Individual Ratios
    for i, ratio in enumerate(target_ratios):
        # Find matching scenario
        scenario = next((s for s in scenarios if s['ratio'] == ratio), None)
        if scenario:
            plot_single_convergence(
                scenario['data'],
                f'Convergence: TO/V Ratio {ratio}',
                f'Output/Plot_Convergence_Ratio_{ratio}.png',
                colors[i % len(colors)]
            )

    # 2. Plot "GLOBAL" Combined Data
    # Flatten all data from all scenarios into one giant list (or average them)
    # Method: Calculate average makespan across all scenarios per replication index
    # (Assuming all have 5 replications)
    n_reps = scenarios[0]['n']
    global_data = []

    for r in range(n_reps):
        # Get the value of replication 'r' for ALL scenarios
        rep_values = [s['data'][r] for s in scenarios]
        global_data.append(np.mean(rep_values))

    plot_single_convergence(
        global_data,
        'Global Model Stability (All Scenarios)',
        'Output/Plot_Convergence_Global.png',
        '#9467bd'  # Purple
    )


if __name__ == "__main__":
    data_list = load_all_scenarios()

    if data_list:
        generate_table(data_list)
        plot_all_convergences(data_list)
    else:
        print("No valid data found.")