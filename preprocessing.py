import os
import xlrd
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 12
HORIZON = 1
TEST_DAYS = 6


def load_raw(filepath):
    # opens the XLS file and reads every row from the Data sheet
    # each row = one detector at one intersection on one day
    # with 96 traffic readings (V00-V95), one per 15-min interval
    wb = xlrd.open_workbook(filepath)
    sh = wb.sheet_by_name('Data')
    rows = []
    for r in range(2, sh.nrows):  # skip 2 header rows
        row = sh.row_values(r)
        # column 9 is the date stored as an Excel serial number, convert to real date
        dt = xlrd.xldate_as_datetime(row[9], wb.datemode).date()
        rows.append({
            'scats_id': str(row[0]).strip(),
            'location': str(row[1]).strip(),
            'lat': float(row[3]),
            'lon': float(row[4]),
            'date': dt,
            'readings': [float(v) for v in row[10:106]]
        })
    return rows


def build_aggregated_df(rows):
    # each intersection has up to 4 detectors (one per road direction: N, S, E, W)
    # expand the 96 readings per row into individual timestep records
    # then sum all detector directions to get total flow per intersection per timestep
    records = []
    for row in rows:
        for i, val in enumerate(row['readings']):
            records.append({
                'scats_id': row['scats_id'],
                'lat': row['lat'],
                'lon': row['lon'],
                'date': row['date'],
                'interval': i,
                'flow': val
            })
    df = pd.DataFrame(records)
    # sum flow across all directions, average the lat/lon per site
    flow_agg = df.groupby(['scats_id', 'date', 'interval'])['flow'].sum().reset_index()
    coord_agg = df.groupby('scats_id')[['lat', 'lon']].mean().reset_index()
    agg = flow_agg.merge(coord_agg, on='scats_id')
    agg = agg.sort_values(['scats_id', 'date', 'interval']).reset_index(drop=True)
    return agg


def make_sequences(values, dates, lookback=12, horizon=1):
    # creates sliding window sequences for the model
    # each sequence: use `lookback` steps as input X, predict next `horizon` steps as y
    # skips any window that spans a missing day to avoid corrupted sequences
    X, y = [], []
    for i in range(len(values) - lookback - horizon + 1):
        window_dates = dates[i:i + lookback + horizon]
        has_gap = False
        for j in range(1, len(window_dates)):
            if (window_dates[j] - window_dates[j-1]).days > 1:
                has_gap = True
                break
        if not has_gap:
            X.append(values[i:i + lookback])
            y.append(values[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y)


def build_dataset(filepath, lookback=LOOKBACK, horizon=HORIZON, test_days=TEST_DAYS):
    print('Loading raw data...')
    rows = load_raw(filepath)
    print(f'Loaded {len(rows)} detector-day records')

    agg = build_aggregated_df(rows)

    # time-based split: last 6 days = test, everything before = train
    # done BEFORE making sequences to prevent future data leaking into training
    all_dates = sorted(agg['date'].unique())
    cutoff = all_dates[-test_days]
    print(f'Train: {all_dates[0]} to {all_dates[-test_days-1]}, Test: {cutoff} to {all_dates[-1]}')

    train_X_all, train_y_all, test_X_all, test_y_all = [], [], [], []
    scalers, coords = {}, {}

    for site_id, group in agg.groupby('scats_id'):
        # store lat/lon for each site, used later to calculate road distances
        coords[site_id] = (group['lat'].iloc[0], group['lon'].iloc[0])

        values = group['flow'].values.reshape(-1, 1).astype(np.float32)

        # assign each reading its actual date (96 readings per day)
        # needed so make_sequences can detect gaps between days
        date_per_step = []
        for d in group['date'].unique():
            date_per_step.extend([datetime.date.fromisoformat(str(d))] * 96)
        date_per_step = date_per_step[:len(values)]

        # normalise each site independently to 0-1
        # different sites have very different traffic volumes so we scale per site
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values).flatten()
        scalers[site_id] = scaler

        # split values and dates by the cutoff date
        train_mask = [d < cutoff for d in date_per_step]
        test_mask = [d >= cutoff for d in date_per_step]
        train_vals = scaled[train_mask]
        train_dates = [d for d, m in zip(date_per_step, train_mask) if m]
        test_vals = scaled[test_mask]
        test_dates = [d for d, m in zip(date_per_step, test_mask) if m]

        Xt, yt = make_sequences(train_vals, train_dates, lookback, horizon)
        Xv, yv = make_sequences(test_vals, test_dates, lookback, horizon)

        if len(Xt) > 0:
            train_X_all.append(Xt)
            train_y_all.append(yt)
        if len(Xv) > 0:
            test_X_all.append(Xv)
            test_y_all.append(yv)

    # combine all sites and reshape to (samples, timesteps, features) for LSTM/GRU input
    X_train = np.concatenate(train_X_all).reshape(-1, lookback, 1)
    y_train = np.concatenate(train_y_all)
    X_test = np.concatenate(test_X_all).reshape(-1, lookback, 1)
    y_test = np.concatenate(test_y_all)

    print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    print(f'Sites: {len(scalers)}, zero values in raw data: {(agg["flow"]==0).sum()}')

    return X_train, X_test, y_train, y_test, scalers, coords, agg