# An Agent-Based Discrete Event Simulation of Teleoperated Driving in Freight Transport Operations: The Fleet Sizing Problem

This repository contains the Python code for the simulation and analysis presented in the research paper: "An Agent-Based Discrete Event Simulation of Teleoperated Driving in Freight Transport Operations: The Fleet Sizing Problem" by Bahman Madadi, Ali Nadi, Gonçalo Homem de Almeida Correia, Thierry Verduijn, and Lóránt Tavasszya.

## Abstract

Teleoperated or remote-controlled driving (TOD) complements automated driving and acts as a transitional technology towards full automation. An economic advantage of teleoperated driving in logistics operations lies in managing fleets with fewer teleoperators compared to vehicles with in-vehicle drivers. This alleviates growing truck driver shortage problems in the logistics industry and can save costs. However, a trade-off exists between the teleoperator-to-vehicle (TO/V) ratio and the service level of teleoperation. This study designs a simulation framework to explore this trade-off, generating multiple performance indicators as proxies for teleoperation service level. By applying the framework, the research identifies factors influencing the trade-off and optimal teleoperator-to-vehicle ratios under different scenarios. A case study on road freight tours in The Netherlands reveals that for any operational settings, a teleoperation-to-vehicle ratio below one can manage all freight truck tours without delay. The minimum teleoperator-to-vehicle ratio for zero-delay operations is never above 0.6, implying a minimum of 40% teleoperation labor cost saving. For operations where a small delay is allowed, teleoperator-to-vehicle ratios as low as 0.4 are shown to be feasible, indicating potential savings of up to 60%.

## Methodology

The study employs a simulation framework that integrates a teleoperation simulator with MASS-GT, a multi-agent simulation system for goods transport.
1.  **MASS-GT:** Simulates logistics decision-making to generate road freight tours, which serve as the demand for teleoperation. It includes:
    * **Shipment Module:** Simulates long-term tactical choices (producer selection, distribution channel, shipment size/vehicle type, delivery time).
    * **Scheduling Module:** Simulates short-term tactical choices (tour formation, delivery time optimization, vehicle type).
2.  **Teleoperation Simulator:** A Discrete Event Simulation (DES) model that processes the tour data from MASS-GT.
    * **Entities:** Trucks with activities (Idle, In Queue, Takeover, Teleoperated, Signed off).
    * **Resources:** Teleoperators with states (Idle, Busy, Resting, Takeover).
    * **Events:** Occurrences triggering state changes (e.g., takeover request, trip completion).
    * **Queues:** Trucks wait for available teleoperators (FIFO).
    * **Key Performance Indicators (KPIs):** Average waiting time per vehicle, waiting time per vehicle in queue, vehicle utilization, teleoperator utilization, makespan, tour completion rate, distance completion rate, and delay.

The simulation explores various scenarios by adjusting parameters like the teleoperator-to-vehicle (TO/V) ratio, takeover times, simulation start times, and working shifts.

## Code Structure

The Python code is organized into the following modules:

* `preprocess.py`: Handles the preprocessing of input data (e.g., MassGT tour data). It selects and filters tours based on specified criteria (tour length, start time, proportion of tours) and prepares the data for the simulation, including calculating buffer times and moving durations.
* `simulate.py`: The main script for running the discrete event simulation. It defines vehicle and teleoperator classes, initializes simulation parameters (e.g., number of replications, TO/V ratios, takeover times), and manages the simulation loop. It calls functions from `event.py` to process events and from `preprocess.py` to get input data. After the simulation, it collects and prepares data for reporting.
* `event.py`: Contains functions for processing different types of events within the simulation, such as vehicle idling, queuing for a teleoperator, teleoperator takeover, teleoperation of a vehicle, and teleoperator resting. It manages the event list and updates the state of vehicles and teleoperators.
* `report.py`: Generates summary statistics and visualizations from the simulation results. It creates reports on utilizations, statuses, queue lengths, makespan, and completion rates. It also produces trade-off plots showing KPIs versus TO/V ratios for different scenarios.
* `visualize.py`: Contains functions for creating plots of the simulation results, such as the number of vehicles in different states over time, queue lengths, and teleoperator statuses. It generates detailed plots for individual replications and summary plots.

## Setup and Installation

1.  **Prerequisites:**
    * Python 3.x
    * The following Python libraries (install via pip):
        * `numpy`
        * `pandas`
        * `matplotlib`
        * `seaborn`

    ```bash
    pip install numpy pandas matplotlib seaborn
    ```

2.  **Input Data:**
    * The simulation requires an input file named `Tours_REF.csv` in an `Input/` directory. This file should contain the tour data generated by MassGT or a similar source, formatted as expected by `preprocess.py`.
    * The `preprocess.py` script will generate temporary filtered tour files (e.g., `Input/Tours_filtered_S0.csv`) during its execution. Note that the input file in this repository is compressed to save space. please unzip it before running the code.

3.  **Output Directory:**
    * The simulation will create an `Output/` directory to store results, including CSV files of statistics and JPEG images of plots. Subdirectories will be created for each scenario.

## Running the Simulation

To run the simulation, execute the `simulate.py` script from the command line:

```bash
python simulate.py
```
The script will iterate through different scenarios defined by parameters within the parameters() function in simulate.py. It will:
* Preprocess input data for each replication.
* Run the simulation for the specified number of replications per scenario.
* Generate reports and visualizations for each scenario.
* Generate overall trade-off plots.
  
### Key Parameters: 
The simulation behavior can be configured by modifying the parameters in the parameters() function within simulate.py:

* replication: Number of simulation runs for each scenario (e.g., 5).

* sample_size: Proportion of total tours to be included in the simulation (e.g., 0.01 for 1%).

* simulation_start: List of simulation start times in hours (e.g., [0, 5, 8]).

* simulation_duration: List of simulation durations in hours (e.g., [9, 24]).

* to2v_ratios: Array of teleoperator-to-vehicle ratios to test (e.g., np.array(list(range(30, 105, 5))) / 100 for 0.30, 0.35, ..., 1.00).

* takeover_times: List of takeover times in minutes (e.g., [0, 1, 2, 3]).

* max_to_duration: Maximum continuous teleoperation duration for a teleoperator before a long rest, in minutes (e.g., 4.5 * 60).

* rest_short: Duration of short rest for teleoperators in minutes (e.g., 10).

* rest_long: Duration of long rest for teleoperators in minutes (e.g., 45).

### Output
The simulation generates the following outputs in the Output/ directory, organized by scenario:

CSV files:

* Summary statistics for utilization, status, counts, queues, and makespan (e.g., R_0_summary_utilization.xlsx, R_0_dist_completion.csv).

* Detailed event logs, queue states, and vehicle/teleoperator states for the first replication of each scenario (e.g., R_1_events.csv).

JPEG images:

* Plots showing the number of vehicles in different states, queue lengths, and teleoperator states over time for the first replication (e.g., R_1_states_vh.jpeg, R_1_queues.jpeg).

* A summary plot combining these for the first replication (e.g., R_1_summary.jpeg).

Overall Trade-off Plots:

* Located in Output/0 Ratios/.

* Plots showing KPIs (average queue time, max queue time, tour completion rate, distance completion rate, delay) versus teleoperator-to-vehicle ratios for different takeover times and tour scenarios (e.g., tl-9_tb-0_avg_q_times.jpeg).

* An Excel file Full_ratios.xlsx containing the aggregated data used for these plots.




## Citation

If you use this research or code, please cite the original paper:

Madadi, B., Nadi, A., Homem de Almeida Correia, G., Verduijn, T., & Tavasszya, L. (2023). An Agent-Based Discrete Event Simulation of Teleoperated Driving in Freight Transport Operations: The Fleet Sizing Problem. arXiv preprint arXiv:2311.14225.


BibTeX:

@misc{madadi2023agentbased,
      title={An Agent-Based Discrete Event Simulation of Teleoperated Driving in Freight Transport Operations: The Fleet Sizing Problem},
      author={Bahman Madadi and Ali Nadi and Gonçalo Homem de Almeida Correia and Thierry Verduijn and Lóránt Tavasszya},
      year={2023},
      eprint={2311.14225},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}



## License

MIT License

Copyright (c) 2023 Bahman Madadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

