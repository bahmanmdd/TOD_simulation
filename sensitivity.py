import os
import pandas as pd

# Configuration
SENSITIVITY_DIR = "Sensitivity"
OUTPUT_DIR = "Output"
RATIOS_SUBFOLDER = "0 Ratios"
RATIOS_FILE = "Full_ratios.xlsx"

KPI_COLUMNS = ["delay", "distance_completion", "Makespan", "AVG_queue_time"]
TO2VEHICLE_RATIOS = [0.3, 0.5, 0.6, 1]


def calculate_mape(actual, reference):
    """Calculate Mean Absolute Percentage Error, handling zero values."""
    epsilon = 1e-10
    return ((actual - reference).abs() / (reference.abs() + epsilon)) * 100


def load_and_filter_ratios(path, extra_filters=None):
    """Load ratios file and apply standard filters."""
    if not os.path.exists(path):
        return None

    df = pd.read_excel(path)
    df = df[df["TO2vehicle_ratio"].isin(TO2VEHICLE_RATIOS)]

    if extra_filters:
        for col, val in extra_filters.items():
            df = df[df[col] == val]

    return df


def summarize_by_groups(df):
    """Group by TO2vehicle_ratio and TO_takeover_time, average KPIs."""
    return (df.groupby(["TO2vehicle_ratio", "TO_takeover_time"])[KPI_COLUMNS]
            .mean()
            .reset_index())


def summarize_sensitivity_scenario(scenario_path):
    """Summarize a single sensitivity scenario."""
    ratios_path = os.path.join(scenario_path, RATIOS_SUBFOLDER, RATIOS_FILE)
    df = load_and_filter_ratios(ratios_path)
    return summarize_by_groups(df) if df is not None else None


def summarize_output_filtered():
    """Summarize Output ratios filtered for tour_len=9 and tour_begin=8."""
    ratios_path = os.path.join(OUTPUT_DIR, RATIOS_SUBFOLDER, RATIOS_FILE)
    df = load_and_filter_ratios(ratios_path, {"tour_len": 9, "tour_begin": 8})

    if df is None:
        print(f"File not found: {ratios_path}")
        return

    summary = summarize_by_groups(df)
    output_path = os.path.join(OUTPUT_DIR, "summary_filtered.xlsx")
    summary.to_excel(output_path, index=False)


def compute_mape_table(scenario_summary, reference_summary, scenario_name):
    """Compute MAPE between scenario and reference summaries."""
    merged = pd.merge(
        scenario_summary,
        reference_summary,
        on=["TO2vehicle_ratio", "TO_takeover_time"],
        suffixes=("_scenario", "_reference")
    )

    mape = merged[["TO2vehicle_ratio", "TO_takeover_time"]].copy()

    for col in KPI_COLUMNS:
        mape[col] = calculate_mape(
            merged[f"{col}_scenario"],
            merged[f"{col}_reference"]
        )

    mape["scenario"] = scenario_name
    return mape


def process_sensitivity_scenarios(reference_summary):
    """Process all sensitivity scenarios and collect summaries."""
    all_summaries = []
    all_mape_tables = []

    for scenario in os.listdir(SENSITIVITY_DIR):
        scenario_path = os.path.join(SENSITIVITY_DIR, scenario)

        if not os.path.isdir(scenario_path):
            continue

        print(f"Processing scenario: {scenario}")

        summary = summarize_sensitivity_scenario(scenario_path)
        if summary is None:
            print(f"  WARNING: No summary generated for {scenario}")
            ratios_path = os.path.join(scenario_path, RATIOS_SUBFOLDER, RATIOS_FILE)
            print(f"  Looking for: {ratios_path}")
            print(f"  File exists: {os.path.exists(ratios_path)}")
            continue

        print(f"  Summary shape: {summary.shape}")

        # Add scenario info and save individual summary
        summary["scenario"] = scenario
        summary.to_excel(os.path.join(scenario_path, "summary_sensitivity.xlsx"), index=False)
        all_summaries.append(summary)

        # Compute and save MAPE if reference exists
        if reference_summary is not None:
            mape = compute_mape_table(summary, reference_summary, scenario)
            mape.to_excel(os.path.join(scenario_path, "summary_mape.xlsx"), index=False)
            all_mape_tables.append(mape)

    return all_summaries, all_mape_tables


def save_combined_tables(all_summaries, all_mape_tables):
    """Save combined summary and MAPE tables."""
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        # Split scenario column
        combined_summary[['variable_name', 'variable_value']] = combined_summary['scenario'].str.split('-', n=1,
                                                                                                       expand=True)
        combined_summary['variable_value'] = pd.to_numeric(combined_summary['variable_value'], errors='coerce')
        # Sort by variable_name then variable_value
        combined_summary = combined_summary.sort_values(['variable_name', 'variable_value'])
        combined_summary.to_excel(
            os.path.join(SENSITIVITY_DIR, "summary_sensitivity.xlsx"),
            index=False
        )

    if all_mape_tables:
        combined_mape = pd.concat(all_mape_tables, ignore_index=True)
        # Split scenario column
        combined_mape[['variable_name', 'variable_value']] = combined_mape['scenario'].str.split('-', n=1, expand=True)
        combined_mape['variable_value'] = pd.to_numeric(combined_mape['variable_value'], errors='coerce')
        # Sort by variable_name then variable_value
        combined_mape = combined_mape.sort_values(['variable_name', 'variable_value'])
        combined_mape.to_excel(
            os.path.join(SENSITIVITY_DIR, "summary_mape_all.xlsx"),
            index=False
        )


def main():
    # Create reference summary from filtered output
    summarize_output_filtered()

    # Load reference summary
    reference_path = os.path.join(OUTPUT_DIR, "summary_filtered.xlsx")
    reference_summary = pd.read_excel(reference_path) if os.path.exists(reference_path) else None

    # Process all sensitivity scenarios
    all_summaries, all_mape_tables = process_sensitivity_scenarios(reference_summary)

    # Save combined results
    save_combined_tables(all_summaries, all_mape_tables)


if __name__ == "__main__":
    main()