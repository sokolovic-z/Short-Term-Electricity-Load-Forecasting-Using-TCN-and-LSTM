from darts.metrics import mae, mape, rmse

def backtest_model(model, series, past_covariates=None, forecast_horizon=None, scaler=None):

    forecasts = model.historical_forecasts(
        series=series,
        start=series.start_time(),
        past_covariates=past_covariates,
        forecast_horizon=forecast_horizon, 
        stride=1, 
        retrain=False, 
        verbose=True)
    
    if scaler:
        forecasts = scaler.inverse_transform(forecasts)
        actual = scaler.inverse_transform(series)
    else:
        actual = series

    # calculate metrics
    metrics = {
        "MAE": mae(actual, forecasts),
        "MAPE": mape(actual, forecasts),
        "RMSE": rmse(actual, forecasts),
    }

    return forecasts, metrics

    