# import packages
import sys
from pathlib import Path

# add the project root directory to Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT)) 

import optuna
import argparse

from darts.models import TCNModel, BlockRNNModel
from pytorch_lightning.callbacks import EarlyStopping

# utility functions
import utils.dataset_loader
from utils.callbacks import LossHistory
from utils.plotting import plot_loss_history

import warnings; warnings.filterwarnings("ignore")

# reproducibility
SEED = 42

# paths
OUTPUT_PATH = Path("results/training")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
OPTUNA_DB_PATH = f'sqlite:///results/optimization/optuna.db'

def train_tcn(args, study, train, val, covs):
    
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

    # build model with best params
    model = TCNModel(
        n_epochs=args.n_epochs, 
        batch_size=args.batch_size,
        add_encoders=encoders,
        pl_trainer_kwargs={"callbacks": [LossHistory(), early_stopper]},
        model_name=f"tcn_model",
        random_state=SEED,
        save_checkpoints=True,
        force_reset=True,
        **study.best_params # inject optimized params
    )

    # train the model
    model.fit(
        series=train,
        past_covariates=covs,
        val_series=val,
        val_past_covariates=covs,
        load_best=True,
    )

    return model


def train_lstm(args, study, train, val, covs):
   
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
        n_epochs=args.n_epochs, 
        batch_size=args.batch_size,
        add_encoders=encoders,
        pl_trainer_kwargs={"callbacks": [LossHistory(), early_stopper]},
        model_name=f"lstm_model",
        random_state=SEED,
        save_checkpoints=True,
        force_reset=True,
        **study.best_params # inject optimized params
    )

    # train the model
    model.fit(
        series=train,
        future_covariates=covs,
        val_series=val,
        val_future_covariates=covs,
        load_best=True,
    )

    return model

if __name__ == "__main__":
    # CLI arguments
    parser = argparse.ArgumentParser(description='Train models (TCN or LSTM)')
    parser.add_argument("--model", choices=["tcn", "lstm"], required=True)
    parser.add_argument("--n_epochs", type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='Number of samples per training batch')
    args = parser.parse_args()

    # load processed data
    (train_scaled, val_scaled, test_scaled, covs_scaled,
    scaler_target, scaler_covs) = utils.dataset_loader.load_data()

    # load study
    study_name = f"{args.model}_hp_optimization"
    study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_PATH)

    # run training
    match args.model:
        case "tcn":
            model = train_tcn(args, study, train_scaled, val_scaled, covs_scaled)
        case "lstm":
            model = train_lstm(args, study, train_scaled, val_scaled, covs_scaled)

    # visualization of loss history
    plot_loss_history(
        history=model.pl_trainer_kwargs['callbacks'][0],
        filepath=OUTPUT_PATH / f"{args.model}_loss_history.png"
    )
    

