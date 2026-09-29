# import packages
import sys
from pathlib import Path

# add the project root directory to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent)) 

import optuna
import numpy as np

from darts.models import TCNModel
from darts.metrics import smape
from pytorch_lightning.callbacks import EarlyStopping

# utility functions
import utils.dataset_loader

import warnings
warnings.filterwarnings("ignore")

# reproducibility
SEED = 42

# make training results directory
OUTPUT_DIR = Path("results/training/tcn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
def objective(trial):
    
    # fixed parameters
    BATCH_SIZE = 64
    N_EPOCHS = 5
    
    # select horizon and lookback window lengths
    out_len = 96 
    in_len = trial.suggest_int("in_len", 2 * out_len, 7 * out_len, step=out_len)

    # other hps
    num_filters = trial.suggest_int("num_filters", 16, 128, step=16)
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # early stopping callback
    early_stopper = EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-4, mode="min", verbose=True)

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
        weight_norm=True,
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
        load_best=True
    )

    # evaluate the model on the validation set, using sMAPE
    preds = model.predict(series=train_scaled, n=len(val_scaled))
    smape_val = smape(val_scaled, preds, n_jobs=-1)

    return smape_val if np.isfinite(smape_val) else float('inf')


def print_callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value:.2f} and parameters: {trial.params}")
    print(f"Best trial so far: {study.best_trial.number} with value: {study.best_trial.value:.2f} and parameters: {study.best_trial.params}")

# optimize hyperparameters by minimizing the sMAPE on the validation set
if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize", 
        study_name="hp_optimization",
        storage=f"sqlite:///{OUTPUT_DIR}/optuna.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20, n_jobs=1, callbacks=[print_callback])
    
    # visualize optimization
    history = optuna.visualization.plot_optimization_history(study)
    history.write_html(OUTPUT_DIR / "optimization_history.html")
    importance = optuna.visualization.plot_param_importances(study)
    importance.write_html(OUTPUT_DIR / "param_importances.html")