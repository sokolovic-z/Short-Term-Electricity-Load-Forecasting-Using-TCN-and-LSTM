# import packages
import sys
from pathlib import Path

# add the project root directory to Python's module search path
sys.path.append(str(Path(__file__).resolve().parent.parent)) 

import optuna
import argparse

from darts.models import TCNModel
from pytorch_lightning.callbacks import EarlyStopping

# utility functions
import utils.dataset_loader
from utils.callbacks import LossHistory
from utils.plotting import plot_loss_history

import warnings
warnings.filterwarnings("ignore")

# reproducibility
SEED = 42

# make training results directory
OUTPUT_DIR = Path("results/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# optuna database path
OPTUNA_DB_PATH = f'sqlite:///results/optimization/optuna.db'


def train_tcn():
    
    # fixed params
    out_len = 96
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
    # build model with best params
    model = TCNModel(
        output_chunk_length=out_len,
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
        series=train_scaled,
        past_covariates=covs_scaled,
        val_series=val_scaled,
        val_past_covariates=covs_scaled,
        load_best=True,
    )

    return model



def train_lstm():
    pass

if __name__ == "__main__":
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tcn", "lstm"], required=True)
    parser.add_argument("--n_epochs", type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='Number of samples per training batch')
    args = parser.parse_args()

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
    
    # load study
    study_name = f"{args.model}_hp_optimization"
    study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_PATH)

    # run training
    match args.model:
        case "tcn":
            model = train_tcn()
        case "lstm":
            model = train_lstm()

    # visualization of loss history
    plot_loss_history(
        history=model.pl_trainer_kwargs['callbacks'][0],
        filepath=OUTPUT_DIR / f"{args.model}_loss_curve.png"
    )
    

