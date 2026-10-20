# import packages
import sys
from pathlib import Path

# add the project root directory to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent)) 

import argparse
import optuna
import numpy as np

from darts.models import TCNModel, BlockRNNModel
from darts.metrics import smape
from pytorch_lightning.callbacks import EarlyStopping

# utility functions
import utils.dataset_loader

import warnings; warnings.filterwarnings("ignore")

# reproducibility
SEED = 42

# paths
OUTPUT_PATH = Path("results/optimization")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
OPTUNA_DB_FILEPATH = f'sqlite:///{OUTPUT_PATH}/optuna.db'

# define TCN objective function
def objective_tcn(trial):
    
    # select forecast horizon and lookback window lengths
    H = args.forecast_horizon
    out_len = trial.suggest_int("out_len", H, H)
    in_len = trial.suggest_int("in_len", 2 * H, 7 * H, step=H) 

    # other hps
    num_filters = trial.suggest_int("num_filters", 16, 128, step=16)
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    
    # early stopping callback
    early_stopper = EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-4, mode="min", verbose=True)

    # datetime/calendar features
    encoders = {
        "cyclic": {
            "past": [
                "hour",
                "weekday",
                "month",
            ]
        }
    }

    # build TCN model
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        num_filters=num_filters,
        num_layers=None,
        kernel_size=kernel_size,
        weight_norm=weight_norm,
        dropout=dropout,
        dilation_base=2,
        add_encoders=encoders,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={"callbacks": [early_stopper]},
        model_name=f"tcn_model",
        random_state=SEED,
        save_checkpoints=True,
        force_reset=True,
    )

    # train the model
    model.fit(
        series=train_scaled,
        past_covariates=covs_scaled,
        val_series=val_scaled,
        val_past_covariates=covs_scaled,
        load_best=True,
    )

    # evaluate the model on the validation set, using sMAPE
    preds = model.predict(series=train_scaled, n=len(val_scaled))
    smape_val = smape(val_scaled, preds, n_jobs=-1)

    return smape_val if np.isfinite(smape_val) else float('inf')


# define LSTM objective function
def objective_lstm(trial):

    # select forecast horizon and lookback window lengths
    H = args.forecast_horizon
    out_len = trial.suggest_int("out_len", H, H)
    in_len = trial.suggest_int("in_len", 2 * H, 7 * H, step=H) 

    # other hps
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, step=16)
    n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # early stopping callback
    early_stopper = EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-4, mode="min", verbose=True)

    # datetime/calendar features
    encoders = {
        "cyclic": {
            "future": [
                "hour",
                "weekday",
                "month",
            ]
        }
    }

    # build LSTM model
    model = BlockRNNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        model="LSTM",
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        add_encoders=encoders,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={"callbacks": [early_stopper]},
        model_name=f"lstm_model",
        random_state=SEED,
        save_checkpoints=True,
        force_reset=True,
    )

    # train the model
    model.fit(
        series=train_scaled,
        future_covariates=covs_scaled,
        val_series=val_scaled,
        val_future_covariates=covs_scaled,
        load_best=True,
    )

    # evaluate the model on the validation set, using sMAPE
    preds = model.predict(series=train_scaled, n=len(val_scaled))
    smape_val = smape(val_scaled, preds, n_jobs=-1)

    return smape_val if np.isfinite(smape_val) else float('inf')


if __name__ == "__main__":
    
    # CLI arguments setup
    parser = argparse.ArgumentParser(description="Optimize hyperparameters with Optuna")
    parser.add_argument("--model", choices=["tcn", "lstm"], required=True, 
                        help='Choose which model to optimize')
    parser.add_argument("--forecast_horizon", type=int, default=96, 
                        help="Forecast horizon in time steps")
    parser.add_argument("--n_trials", type=int, default=20,
                        help='Number of trials to execute')
    parser.add_argument("--n_epochs", type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='Number of samples per training batch')
    args = parser.parse_args()

    # load processed data
    (train_scaled, val_scaled, test_scaled, covs_scaled,
    scaler_target, scaler_covs) = utils.dataset_loader.load_data()

    # create or load the Optuna study
    study_name = f"{args.model}_hp_optimization"
    study = optuna.create_study(
        direction="minimize", 
        study_name=study_name,
        storage=OPTUNA_DB_FILEPATH,
        load_if_exists=True
    )
    
    # run optimization 
    match args.model:
        case "tcn":
            objective = objective_tcn
        case "lstm":
            objective = objective_lstm
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    
    # visualization of optimization results
    history = optuna.visualization.plot_optimization_history(study)
    importance = optuna.visualization.plot_param_importances(study)
    # save plots as interactive HTML files
    history.write_html(OUTPUT_PATH / f"{args.model}_optimization_history.html")
    importance.write_html(OUTPUT_PATH / f"{args.model}_param_importances.html")