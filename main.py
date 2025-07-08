import os
import random
import warnings
import urllib
import zipfile
warnings.filterwarnings("ignore")
import numpy as np
import wandb
import torch
import constants as cst
import hydra
from config.config import Config, Dataset, FI_2010, LOBSTER, BTC, COMBINED, Model, MLPLOB, TLOB, BiNCTABL, DeepLOB
from run import run_wandb, run, sweep_init
from preprocessing.lobster import LOBSTERDataBuilder
from preprocessing.btc import BTCDataBuilder
from preprocessing.combined import CombinedDataBuilder
from constants import DatasetType
from constants import SamplingType

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="config", config_name="config")
def hydra_app(config: Config):
    print("Starting hydra_app...")
    set_reproducibility(config.experiment.seed)
    print(f"Using device: {cst.DEVICE}")
    print("Starting configuration setup...")

    # Validate configuration
    print(f"Validating config: model={config.model.type}, dataset={config.dataset.type}, checkpoint={config.experiment.checkpoint_reference}")
    if "EVALUATION" in config.experiment.type:
        checkpoint_path = os.path.join(cst.DIR_SAVED_MODEL, config.experiment.checkpoint_reference.replace("data/checkpoints/", ""))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        print(f"Checkpoint validated at: {checkpoint_path}")

    if cst.DEVICE == "cpu":
        accelerator = "cpu"
    else:
        accelerator = "gpu"
    print(f"Device configured, using accelerator: {accelerator}, setting model hyperparameters...")

    if config.dataset.type == DatasetType.FI_2010:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 144
    elif config.dataset.type == DatasetType.BTC:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 40
    elif config.dataset.type == DatasetType.LOBSTER:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 46
    elif config.dataset.type == DatasetType.COMBINED:
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 40

    if config.dataset.type.value == "LOBSTER" and not config.experiment.is_data_preprocessed:
        print("Preparing LOBSTER dataset...")
        data_builder = LOBSTERDataBuilder(
            stocks=config.dataset.training_stocks,
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
        print("LOBSTER dataset preparation complete.")

    elif config.dataset.type.value == "FI_2010" and not config.experiment.is_data_preprocessed:
        print("Preparing FI_2010 dataset...")
        try:
            dir = cst.DATA_DIR + "/FI_2010/"
            for filename in os.listdir(dir):
                if filename.endswith(".zip"):
                    filename = dir + filename
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(dir)
            print("FI_2010 data extracted.")
        except Exception as e:
            raise Exception(f"Error downloading or extracting FI_2010 data: {e}")

    elif config.dataset.type == cst.DatasetType.BTC and not config.experiment.is_data_preprocessed:
        print("Preparing BTC dataset...")
        data_builder = BTCDataBuilder(
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
        print("BTC dataset preparation complete.")

    elif config.dataset.type == cst.DatasetType.COMBINED and not config.experiment.is_data_preprocessed:
        print("Preparing COMBINED dataset...")
        try:
            data_builder = CombinedDataBuilder(
                data_dir=cst.DATA_DIR,
                date_trading_days=config.dataset.dates,
                split_rates=cst.SPLIT_RATES,
                sampling_type=config.dataset.sampling_type if hasattr(config.dataset, 'sampling_type') else SamplingType.NONE,
                sampling_time=config.dataset.sampling_time if hasattr(config.dataset, 'sampling_time') else "1s",
                sampling_quantity=config.dataset.sampling_quantity if hasattr(config.dataset, 'sampling_quantity') else 0,
            )
            data_builder.prepare_save_datasets()
            print("COMBINED dataset preparation complete.")
        except Exception as e:
            raise Exception(f"Error preparing COMBINED dataset: {e}")

    print("Dataset preparation complete, starting WandB...")
    if config.experiment.is_wandb:
        print("Initializing WandB run...")
        if config.experiment.is_sweep:
            print("Setting up WandB sweep...")
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME, entity="")
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            print("Calling run_wandb...")
            run_wandb(config, accelerator)
    else:
        print("Running without WandB...")
        run(config, accelerator)

def set_reproducibility(seed):
    print(f"Setting random seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_torch():
    print("Configuring PyTorch settings...")
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision('high')

if __name__ == "__main__":
    print("Starting main.py...")
    try:
        set_torch()
        hydra_app()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise  # Re-raise to preserve Hydra's error handling