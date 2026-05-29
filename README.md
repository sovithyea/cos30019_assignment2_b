# COS30019 Assignment 2B: Traffic-Based Route Guidance System

Predicts traffic flow at 40 signalised intersections in Boroondara, Melbourne and uses those predictions to find the fastest time-dependent routes between any two SCATS sites.

---

## How it works

The project has two halves that connect together.

**Half 1: Traffic prediction.** Three deep learning models (LSTM, GRU, CNN) are trained on real SCATS detector data from October 2006. Each model learns traffic patterns at 40 intersections across the day, rush hours, quiet periods, weekday rhythms. Given the last 3 hours of 15-minute vehicle counts at a site, the model predicts how many vehicles will pass in the next 15 minutes.

**Half 2: Time-dependent route finding.** The 40 intersections are modelled as a directed road graph. When you request a route, Yen's k-shortest-paths algorithm searches the graph using Dijkstra, but instead of fixed distances as edge weights, it calls the trained model at every step to estimate travel time dynamically:

```
entry_time = departure_time + elapsed_travel_time_so_far
predicted_flow -> converted to speed via a flow-speed curve
travel_time = distance / speed + 0.5 min intersection delay
```

As the algorithm simulates travel through the network, it always asks: given I'd arrive at this intersection at this time, how congested will it be? A departure at 08:00 routes differently than one at 14:00.

The default model used for routing is **LSTM**. (subject to change)

---

## Project structure

```
cos30019_assignment2_b/
│
├── data/
│   ├── raws/
│   │   └── Scats Data October 2006.xls        # raw traffic data (not in repo)
│   ├── processed/
│   │   └── traffic_direction_timeseries.csv
│   ├── route_map/
│   │   ├── final_edges.csv                    # directed road graph (from_site, to_site, distance_km)
│   │   ├── final_nodes.csv
│   │   └── travel_cost.csv
│   └── preprocessing_data.ipynb               # data exploration and preprocessing notebook
│
├── graph/
│   ├── build_route_map.py
│   ├── cus1.py                                # Yen's k-shortest paths + Dijkstra search
│   └── travel_cost.py                         # TravelTimeEstimator, loads predictions, estimates edge costs
│
├── models/
│   ├── evaluate.ipynb
│   ├── model_utils.py
│   ├── train_cnn.py                           # CNN model training
│   ├── train_gru.py                           # GRU model training
│   └── train_lstm.py                          # LSTM model training
│
├── saved_models/
│   ├── cnn_model.keras
│   ├── flow_scaler.joblib
│   ├── gru_model.keras
│   ├── lstm_model.keras
│   ├── model_metrics.csv
│   ├── model_test_predictions.csv             # pre-computed predictions for all sites and intervals
│   └── model_training_history.csv
│
├── testcases/
│   ├── run_testcases.py
│   └── TC01.txt … TC16.txt                    # 16 test cases
│
├── config.json                                # default GUI settings
├── gui.py                                     # tkinter desktop UI
├── README.md
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/sovithyea/cos30019_assignment2_b.git
cd cos30019_assignment2_b
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| tensorflow | 2.17.0 |
| scikit-learn | 1.5.1 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| matplotlib | 3.9.2 |
| xlrd | 2.0.1 |
| python-calamine | 0.4.0 |

---

## Dataset

The raw data file is **not required to run the GUI**. Pre-trained model predictions (`saved_models/model_test_predictions.csv`) and the road graph (`data/route_map/final_edges.csv`) are already included in the repo — clone and run.

The raw file is only needed if you want to retrain the models from scratch. Place it at:

```
data/raws/Scats Data October 2006.xls
```

- **Source:** SCATS (Sydney Coordinated Adaptive Traffic System) loop detectors, Boroondara, Melbourne
- **Coverage:** 40 signalised intersections, October 2006
- **Resolution:** 15-minute vehicle counts (96 readings/day)
- **Train split:** October 1-25
- **Test split:** October 26-31

---

## Training the models

If you have the raw data, train each model individually:

```bash
python models/train_lstm.py
python models/train_gru.py
python models/train_cnn.py
```

Trained weights are saved to `saved_models/` (`.keras` files). Running training also produces `saved_models/model_test_predictions.csv`, which the routing system reads at runtime. If these files already exist, training can be skipped.

---

## Configuration

Default GUI settings are stored in `config.json` in the project root:

```json
{
  "default_departure_time": "08:00",
  "default_route_count": 5
}
```

Edit this file to change what the form pre-fills with on startup and what the **Clear** button resets to. If the file is missing, the GUI falls back to the same values automatically.

---

## Running the GUI

```bash
python gui.py
```

A desktop window opens (880×680). Fill in:

| Field | Description | Example |
|---|---|---|
| Origin SCATS Site | 4-digit intersection ID | `3127` |
| Destination SCATS Site | 4-digit intersection ID | `4063` |
| Departure Time | HH:MM or HH:MM:SS | `08:00` |
| Number of Routes | 1–5 (default 5) | `3` |

Click **Find Route**. Results appear below, ranked by estimated travel time. Click **Clear** to reset all fields.

If either site ID is not in the road graph, the results box will show an error instead of routes.

---

## How the routing works (technical detail)

### Edge cost: flow to speed

`travel_cost.py` converts predicted vehicle flow to a travel time using a quadratic flow-speed relationship:

- Below 351 vehicles/hour → speed = 60 km/h (free flow, speed limit)
- Above 351 vehicles/hour → speed is solved from: `flow = a·speed² + b·speed` where `a = -1.46`, `b = 93.75`
- A fixed **0.5 minute intersection delay** is added per hop regardless of flow

### Search: Yen's k-shortest paths

`cus1.py` implements Yen's algorithm:

1. **First route:** standard Dijkstra from origin to destination, expanding lowest-cost nodes first. Edge costs are computed dynamically at the simulated arrival time for each node.
2. **Alternative routes:** for each accepted route, iterate over spur nodes. At each spur point, block edges already used by accepted routes and re-run Dijkstra. Collect candidates in a min-heap, pop the cheapest, repeat until k routes are found.

Loops are prevented, a node cannot appear twice in the same path.

### Prediction lookup

Rather than running the neural network live during search (which would be slow), predictions are pre-computed for all 40 sites across all 96 daily intervals and stored in `saved_models/model_test_predictions.csv`. At runtime, `TravelTimeEstimator` loads this file and averages the predicted flow for each (site, interval) pair across the available test days (October 26–31), giving a representative time-of-day profile for routing.

---

## Models

All three models take **12 timesteps (3 hours)** as input and predict **1 timestep (15 minutes)** of vehicle flow ahead.

| Model | Architecture | Notes |
|---|---|---|
| LSTM | Long Short-Term Memory | Default model used for routing |
| GRU | Gated Recurrent Unit | Lighter alternative to LSTM |
| CNN | 1D Convolutional Network | Captures local temporal patterns |

To switch the routing model, change `DEFAULT_MODEL` in `graph/travel_cost.py`.

---

## Running test cases

```bash
python testcases/run_testcases.py
```