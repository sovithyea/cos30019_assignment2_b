# COS30019 Assignment 2B — Traffic Flow Prediction

Predicts traffic flow at 40 intersections in Boroondara, Melbourne using LSTM, GRU, and CNN models trained on SCATS detector data from October 2006.

---

## Project Structure

```
cos30019_assignment2_b/
│
├── data/
│   └── Scats Data October 2006.xls   # raw traffic data (not pushed to git)
│
├── models/
│   ├── lstm.py                        # LSTM model
│   ├── gru.py                         # GRU model
│   └── cnn.py                         # CNN model
│
├── saved_models/                      # trained models saved here (not pushed to git)
│
├── preprocessing.py                   # data loading, aggregation, sequencing
├── evaluate.py                        # metrics and plots
├── main.py                            # entry point
├── requirements.txt
└── .gitignore
```

---

## Setup

1. Clone the repo
2. Create a virtual environment and install dependencies:

```
pip install -r requirements.txt
```

3. Place `Scats Data October 2006.xls` inside the `data/` folder

---

## Running

```
python main.py
```

---

## Dataset

- 40 SCATS detector sites across Boroondara, Melbourne
- October 2006, readings every 15 minutes (96 readings/day)
- Train: October 1–25, Test: October 26–31
- Input sequences: 12 timesteps (3 hours) → predict next 1 timestep (15 min)