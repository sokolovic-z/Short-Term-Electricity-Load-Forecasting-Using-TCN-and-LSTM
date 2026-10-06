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

import warnings
warnings.filterwarnings("ignore")

# reproducibility
SEED = 42

# make optimization results directory
OUTPUT_DIR = Path("results/optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# optuna database path
OPTUNA_DB_PATH = f'sqlite:///{OUTPUT_DIR}/optuna.db'


# load processed data
(
train_scaled,
val_scaled,
test_scaled,
covs_scaled,
scaler_target,
scaler_covs,
) = utils.dataset_loader.load_data()

# test portion
train_scaled, val_scaled = train_scaled[:2000], train_scaled[2000:3000]



# define objective function
def objective_tcn(trial):
    
    # fixed parameters
    BATCH_SIZE = 64
    N_EPOCHS = 1
    
    # select forecast horizon and lookback window lengths
    out_len = 96
    in_len = trial.suggest_int("in_len", 2 * out_len, 7 * out_len, step=out_len) 

    # other hps
    num_filters = trial.suggest_int("num_filters", 16, 128, step=16)
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    

    # early stopping callback
    early_stopper = EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-4, mode="min", verbose=True)

    # calendar features
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
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
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


def objective_lstm(trial):
    pass


def print_callback(study, trial):
    pass
    # print(f"Trial {trial.number} finished with value: {float(trial.value):.2f} and parameters: {trial.params}")
    # print(f"Best trial so far: {study.best_trial.number} with value: {study.best_trial.value:.2f} and parameters: {study.best_trial.params}")

# optimize hyperparameters by minimizing the sMAPE on the validation set
if __name__ == "__main__":

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tcn", "lstm"], required=True, 
                        help='Choose which model to optimize')
    args = parser.parse_args()

    study_name = f"{args.model}_hp_optimization"
    study = optuna.create_study(
        direction="minimize", 
        study_name=study_name,
        storage=OPTUNA_DB_PATH,
        load_if_exists=True
    )

    if args.model == "tcn":
        study.optimize(objective_tcn, n_trials=20, n_jobs=1, callbacks=[print_callback])
    elif args.model == "lstm":
        study.optimize(objective_lstm, n_trials=20, n_jobs=1, callbacks=[print_callback])
    
    # visualization
    history = optuna.visualization.plot_optimization_history(study)
    importance = optuna.visualization.plot_param_importances(study)
    history.write_html(OUTPUT_DIR / f"{args.model}_optimization_history.html")
    importance.write_html(OUTPUT_DIR / f"{args.model}_param_importances.html")