from preprocessing import build_dataset

FILEPATH = 'data/Scats Data October 2006.xls'

X_train, X_test, y_train, y_test, scalers, coords, agg = build_dataset(FILEPATH)