import xlrd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 12
HORIZON = 1
TEST_DAYS = 6


def load_raw(filepath):
    wb = xlrd.open_workbook(filepath)
    sh = wb.sheet_by_name('Data')
    datemode = wb.datemode
    rows = []
    for r in range(2, sh.nrows):
        row = sh.row_values(r)
        dt = xlrd.xldate_as_datetime(row[9], datemode).date()
        rows.append({
            'scats_id': str(row[0]).strip(),
            'location': str(row[1]).strip(),
            'lat': float(row[3]),
            'lon': float(row[4]),
            'date': dt,
            'readings': row[10:106]
        })
    return rows


def to_timeseries(rows):
    records = []
    for row in rows:
        for i, val in enumerate(row['readings']):
            records.append({
                'scats_id': row['scats_id'],
                'location': row['location'],
                'lat': row['lat'],
                'lon': row['lon'],
                'date': row['date'],
                'interval': i,
                'flow': float(val)
            })
    df = pd.DataFrame(records)
    df = df.sort_values(['scats_id', 'location', 'date', 'interval']).reset_index(drop=True)
    return df


def make_sequences(series, lookback=12, horizon=1):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y)


def build_dataset(filepath, lookback=LOOKBACK, horizon=HORIZON, test_days=TEST_DAYS):
    print('Loading raw data...')
    rows = load_raw(filepath)
    print(f'Loaded {len(rows)} detector-day records')

    df = to_timeseries(rows)

    # aggregate all detector directions per intersection per timestep
    agg = df.groupby(['scats_id', 'date', 'interval'])['flow'].sum().reset_index()
    agg = agg.sort_values(['scats_id', 'date', 'interval']).reset_index(drop=True)

    # coords per site
    coords = df.groupby('scats_id')[['lat', 'lon']].first().to_dict('index')

    # time-based train/test split
    all_dates = sorted(agg['date'].unique())
    split_date = all_dates[-test_days]
    print(f'Train: {all_dates[0]} to {all_dates[-test_days - 1]}, Test: {split_date} to {all_dates[-1]}')

    train_X_all, train_y_all = [], []
    test_X_all, test_y_all = [], []
    scalers = {}

    for site_id, group in agg.groupby('scats_id'):
        group = group.sort_values(['date', 'interval'])
        flow = group['flow'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        flow_scaled = scaler.fit_transform(flow).flatten()
        scalers[site_id] = scaler

        dates = group['date'].values
        train_mask = dates < split_date
        test_mask = dates >= split_date

        Xt, yt = make_sequences(flow_scaled[train_mask], lookback, horizon)
        Xv, yv = make_sequences(flow_scaled[test_mask], lookback, horizon)

        if len(Xt) > 0:
            train_X_all.append(Xt)
            train_y_all.append(yt)
        if len(Xv) > 0:
            test_X_all.append(Xv)
            test_y_all.append(yv)

    # combine all sites and reshape to (samples, timesteps, features) for LSTM/GRU
    X_train = np.concatenate(train_X_all).reshape(-1, lookback, 1)
    y_train = np.concatenate(train_y_all)
    X_test = np.concatenate(test_X_all).reshape(-1, lookback, 1)
    y_test = np.concatenate(test_y_all)

    print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    print(f'Sites: {len(scalers)}, zero values in raw data: {(agg["flow"] == 0).sum()}')

    return X_train, X_test, y_train, y_test, scalers, coords, agg