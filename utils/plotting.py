import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_loss_curve(history, filepath):
    plt.figure(figsize=(8, 5))

    plt.plot(history.train_loss, label="Train loss")
    plt.plot(history.val_loss, label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_lookback(metrics):
    pass

def plot_weekly_forecasts(actual, forecasts: dict, output_dir):

    # dir path to store results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # compute number of weeks
    ppd = 96 # points per day
    ppw = 7 * ppd # points per week
    n_weeks = len(actual) // ppw

    # find common forecast period
    forecast_starts = [
        fc.start_time()
        for fc in forecasts.values()
    ]
    common_start = max(forecast_starts)
    # align all forecasts
    forecasts = {
        mdl_name: fc.slice(common_start, fc.end_time())
        for mdl_name, fc in forecasts.items()
    }
    # align actual data to common fc start
    actual = actual.slice(common_start, actual.end_time())

    # iterate through weeks and plot weekly forecasts
    for week in range(n_weeks):
        start = week * ppw
        end = start + ppw
        actual_week = actual[start:end]
    
        plt.figure(figsize=(12, 5))
        actual_week.plot(label='Actual')
        for mdl_name, fcs in forecasts.items():
            fcs[start:end].plot(label=mdl_name)

        plt.xlabel("Datetime")
        plt.ylabel("Electricity Consumption [kWh]")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/week_{week + 1}.png', dpi=300)
        plt.close()


def plot_daily_forecasts(actual, forecasts, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 24 hourly points per day
    ppd = 96
    n_days = len(actual) // ppd

    # find common forecast period
    forecast_starts = [
        fc.start_time()
        for fc in forecasts.values()
    ]
    common_start = max(forecast_starts)
    # Align all forecasts
    forecasts = {
        mdl_name: fc.slice(common_start, fc.end_time())
        for mdl_name, fc in forecasts.items()
    }
    # Align actual data to common fc start
    actual = actual.slice(common_start, actual.end_time())

    # iterate through days
    for day in range(n_days//5): # plot only 1/5 of the days to avoid too many plots
        start = day * ppd
        end = start + ppd
        actual_day = actual[start:end]

        plt.figure(figsize=(12, 5))

        actual_day.plot(label="Actual")

        for mdl_name, fcs in forecasts.items():
            fcs[start:end].plot(label=mdl_name)

        plt.xlabel("Datetime")
        plt.ylabel("Electricity Consumption [kWh]")
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(
            output_dir / f"day_{day + 1}.png",
            dpi=300
        )
        plt.close()

    