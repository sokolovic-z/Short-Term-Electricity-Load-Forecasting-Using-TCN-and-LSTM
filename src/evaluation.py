
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.backtesting import backtest_model
from utils.plotting import plot_weekly_forecasts, plot_lookback
import utils.dataset_loader
from darts.models import (
    BlockRNNModel,
    TCNModel,
)
import pandas as pd
import os

# results dir
OUTPUT_DIR = Path('results/evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():

    # Load processed data
    (
        train_scaled,
        val_scaled,
        test_scaled,
        covs_scaled,
        scaler_target,
        scaler_covs,
    ) = utils.dataset_loader.load_data()
    test = scaler_target.inverse_transform(test_scaled)

    # load models from the best checkpoint
    model_names = os.listdir('darts_logs') # list available models (tcn_l{lookback}_h{horizon} or lstm_l{lookback}_h{horizon})
    models = []
    for model_name in model_names:
        if 'tcn' in model_name:
            models.append(TCNModel.load_from_checkpoint(model_name=model_name, best=True))
        if 'lstm' in model_name:
            models.append(BlockRNNModel.load_from_checkpoint(model_name=model_name, best=True))

    # backtesting
    forecasts = {}
    metrics = {}
    for model in models:
        fcs, mcs = backtest_model(model, test_scaled, covs_scaled, forecast_horizon=model.output_chunk_length, scaler=scaler_target) # return forecasts and metrics
        forecasts[model.model_name] = fcs
        metrics[model.model_name] = mcs

    # sorting metrics
    metrics_sorted = []
    for model_name, mcs in metrics.items():

        parts = model_name.split("_")

        model_type = parts[0]
        lookback = int(parts[1][1:])
        horizon = int(parts[2][1:])

        metrics_sorted.append({
            "model_name": model_name,
            "model": model_type,
            "lookback": lookback,
            "horizon": horizon,
            **mcs
        })
        metrics_sorted = sorted(metrics_sorted, key=lambda x: (x["model"], x["lookback"]))

    # saving metrics to excel
    metrics = pd.DataFrame(metrics_sorted)
    metrics.to_excel(f'{OUTPUT_DIR}/metrics.xlsx', engine='openpyxl')

    # best model by MAPE for each model type
    best_tcn = (metrics[metrics["model"] == "tcn"].sort_values(by="MAPE").iloc[0])["model_name"]
    best_lstm = (metrics[metrics["model"] == "lstm"].sort_values(by="MAPE").iloc[0])["model_name"]
    # extract forecasts only for best TCN and best LSTM
    best_forecasts = {
        "tcn": forecasts[best_tcn],
        "lstm": forecasts[best_lstm],
    }

    # plotting forecasts of both models for each week in the test set
    plot_weekly_forecasts(
        actual=test,
        forecasts=best_forecasts, 
        output_dir=f'{OUTPUT_DIR}/weekly_forecasts'
    )

    # plot lookback vs MAPE for each model type
    plot_lookback(
        metrics=metrics,
        output_dir=f'{OUTPUT_DIR}/lookback_vs_mape'
    )




if __name__ == "__main__":
    main()