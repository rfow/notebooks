import pandas as pd
import numpy as np
import datetime as dt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def sarimax_fc(train, test, order, seas_order, exog_train=None, exog_test=None):
    model = SARIMAX(train, order=order, exog=exog_train, seasonal_order=seas_order)
    results = model.fit()
    start, end = len(train), len(test) + len(train) - 1
    pred = results.predict(start, end, exog=exog_test, typ='levels').rename('sarima_predictions')
    rmse_pred, rmse_pred_pct = rmse(test, pred), rmse(test, pred) / test.mean()
    results = {'prediction': pred, 'rmse': rmse_pred, 'rmse_pct': rmse_pred_pct}
    return results


def xgb_fc(xtrain, xtest, ytrain, ytest):
    xgb_model = xgb.XGBRegressor(n_estimators=1000)
    xgb_model.fit(xtrain, ytrain, verbose=False)
    pred = pd.Series(xgb_model.predict(xtest), ytest.index).rename('xgb_predictions')
    rmse_pred, rmse_pred_pct = rmse(ytest[ytest.columns[0]], pred), rmse(ytest[ytest.columns[0]], pred) / ytest[ytest.columns[0]].mean()
    results = {'prediction': pred, 'rmse': rmse_pred, 'rmse_pct': rmse_pred_pct}
    return results


def lstm_fc_unscaled(train, test, length=12, batch=1, epochs=200):
    scaler = MinMaxScaler()
    scaler.fit(train.values.reshape(-1, 1))
    scaled_train = scaler.transform(train.values.reshape(-1, 1))
    generator = TimeseriesGenerator(scaled_train, scaled_train, length, batch_size=batch)
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(length, batch)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(generator, epochs=epochs, verbose=0)
    test_predictions = []
    first_batch = scaled_train[-length:]
    current_batch = first_batch.reshape((1, length, batch))
    for b in range(len(test)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    pred = pd.Series(scaler.inverse_transform(test_predictions).reshape(1, 12)[0], test.index).rename(
        'lstm_predictions')
    rmse_pred, rmse_pred_pct = rmse(test, pred), rmse(test, pred) / test.mean()
    results = {'prediction': pred, 'rmse': rmse_pred, 'rmse_pct': rmse_pred_pct}
    return results


# def test():
#     raw_data = pd.read_csv('data/demand.csv', index_col='Dates', parse_dates=True)
#     start_date = raw_data['ICSG M'].dropna().first_valid_index()  # Jan-15
#     end_date = dt.datetime(2018, 12, 31)
#     global_demand = raw_data[['WM Y', 'ICSG M']][start_date:end_date]
#     global_demand['ICSG M%'] = global_demand['ICSG M'] / global_demand['ICSG M'].resample('Y').sum().resample(
#         'M').last().reindex(pd.date_range(start_date, end_date)).bfill()
#     global_demand['WM M'] = global_demand['WM Y'].bfill() * global_demand['ICSG M%']
#     global_demand.index.freq = 'M'
#
#     # sarima examples
#     sarima_train = global_demand['WM M'][:'20171231']
#     sarima_test = global_demand['WM M']['20180101':]
#     results = sarimax_fc(sarima_train, sarima_test, (0, 1, 1), (1, 0, 1, 12))
#     print(results)
#
#
# if __name__ == '__main__':
#     test()