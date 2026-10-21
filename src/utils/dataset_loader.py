import pickle


def load_data(dataset_filepath="data/electricity_processed.pkl"):
    # load processed data
    with open(dataset_filepath, "rb") as f: 
        data = pickle.load(f)
    return (
        data["train_scaled"],
        data["val_scaled"],
        data["test_scaled"],
        data["covs_scaled"],
        data["scaler_target"],
        data["scaler_covs"]
    )