import lightning as L
import omegaconf
import torch
import glob
import os
from lightning.pytorch.loggers import WandbLogger
import wandb
import collections
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from config.config import Config, Dataset, FI_2010, LOBSTER, BTC, COMBINED, Model, MLPLOB, TLOB, BiNCTABL, DeepLOB
from models.engine import Engine
from preprocessing.fi_2010 import fi_2010_load
from preprocessing.lobster import lobster_load
from preprocessing.btc import btc_load
from preprocessing.combined import combined_load
from preprocessing.dataset import DataModule, Dataset as TorchDataset
import constants as cst
from constants import DatasetType, SamplingType
from sklearn.metrics import confusion_matrix
from typing import List
import traceback
import collections
from sklearn.metrics import f1_score, accuracy_score

# Add safe globals to allow deserialization of checkpoint
# torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig, omegaconf.base.ContainerMetadata, List, list, collections.defaultdict, dict, int])

# Register subclasses as attributes for checkpoint compatibility (for checkpoint loading)
# setattr(Dataset, "FI_2010", FI_2010)
# setattr(Dataset, "LOBSTER", LOBSTER)
# setattr(Dataset, "BTC", BTC)
# setattr(Dataset, "COMBINED", COMBINED)
# setattr(Model, "MLPLOB", MLPLOB)
# setattr(Model, "TLOB", TLOB)
# setattr(Model, "BINCTABL", BiNCTABL)
# setattr(Model, "DEEPLOB", DeepLOB)
# print("Registered Dataset.FI_2010:", hasattr(Dataset, "FI_2010"))
# print("Registered Model.TLOB:", hasattr(Model, "TLOB"))

def run(config: Config, accelerator):
    seq_size = config.model.hyperparameters_fixed["seq_size"]
    dataset = config.dataset.type.value
    horizon = config.experiment.horizon
    if dataset == "LOBSTER":
        training_stocks = config.dataset.training_stocks
        config.experiment.dir_ckpt = f"{dataset}_{training_stocks}_seq_size_{seq_size}_horizon_{horizon}_seed_{config.experiment.seed}"
    else:
        config.experiment.dir_ckpt = f"{dataset}_seq_size_{seq_size}_horizon_{horizon}_seed_{config.experiment.seed}"

    trainer = L.Trainer(
        accelerator=accelerator,
        precision=cst.PRECISION,
        max_epochs=config.experiment.max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=1, verbose=True, min_delta=0.002),
            TQDMProgressBar(refresh_rate=100)
        ],
        num_sanity_val_steps=0,
        detect_anomaly=False,
        profiler=None,
        check_val_every_n_epoch=1
    )
    train(config, trainer)

def train(config: Config, trainer: L.Trainer, run=None):
    print("Starting train function...")
    print_setup(config)
    print("Beginning data loading...")
    dataset_type = config.dataset.type.value
    seq_size = config.model.hyperparameters_fixed["seq_size"]
    horizon = config.experiment.horizon
    model_type = config.model.type
    checkpoint_ref = config.experiment.checkpoint_reference
    checkpoint_ref_clean = checkpoint_ref.replace("data/checkpoints/", "")
    checkpoint_path = os.path.join(cst.DIR_SAVED_MODEL, checkpoint_ref_clean)
    if dataset_type == "FI_2010":
        path = cst.DATA_DIR + "/FI_2010"
        train_input, train_labels, val_input, val_labels, test_input, test_labels = fi_2010_load(path, seq_size, horizon, config.model.hyperparameters_fixed["all_features"])
        train_set = TorchDataset(train_input, train_labels, seq_size)
        val_set = TorchDataset(val_input, val_labels, seq_size)
        test_set = TorchDataset(test_input, test_labels, seq_size)
        data_module = DataModule(
            train_set=TorchDataset(train_input, train_labels, seq_size),
            val_set=TorchDataset(val_input, val_labels, seq_size),
            test_set=TorchDataset(test_input, test_labels, seq_size),
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,
            num_workers=4
        )
        test_loaders = [data_module.test_dataloader()]
    
    elif dataset_type == "BTC":
        train_input, train_labels = btc_load(cst.DATA_DIR + "/BTC/train.npy", cst.LEN_SMOOTH, horizon, seq_size)
        val_input, val_labels = btc_load(cst.DATA_DIR + "/BTC/val.npy", cst.LEN_SMOOTH, horizon, seq_size)  
        test_input, test_labels = btc_load(cst.DATA_DIR + "/BTC/test.npy", cst.LEN_SMOOTH, horizon, seq_size)
        train_set = TorchDataset(train_input, train_labels, seq_size)
        val_set = TorchDataset(val_input, val_labels, seq_size)
        test_set = TorchDataset(test_input, test_labels, seq_size)
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,
            num_workers=4
        ) 
        test_loaders = [data_module.test_dataloader()]
        
    elif dataset_type == "LOBSTER":
        training_stocks = config.dataset.training_stocks
        testing_stocks = config.dataset.testing_stocks
        for i in range(len(training_stocks)):
            if i == 0:
                for j in range(2):
                    if j == 0:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/train.npy"
                        train_input, train_labels = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
                    if j == 1:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/val.npy"
                        val_input, val_labels = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
            else:
                for j in range(2):
                    if j == 0:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/train.npy"
                        train_labels = torch.cat((train_labels, torch.zeros(seq_size+horizon-1, dtype=torch.long)), 0)
                        train_input_tmp, train_labels_tmp = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
                        train_input = torch.cat((train_input, train_input_tmp), 0)
                        train_labels = torch.cat((train_labels, train_labels_tmp), 0)
                    if j == 1:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/val.npy"
                        val_labels = torch.cat((val_labels, torch.zeros(seq_size+horizon-1, dtype=torch.long)), 0)
                        val_input_tmp, val_labels_tmp = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
                        val_input = torch.cat((val_input, val_input_tmp), 0)
                        val_labels = torch.cat((val_labels, val_labels_tmp), 0)
        test_loaders = []
        for i in range(len(testing_stocks)):
            path = cst.DATA_DIR + "/" + testing_stocks[i] + "/test.npy"
            test_input, test_labels = lobster_load(path, config.model.hyperparameters_fixed["all_features"], cst.LEN_SMOOTH, horizon, seq_size)
            test_set = TorchDataset(test_input, test_labels, seq_size)
            test_dataloader = DataLoader(
                dataset=test_set,
                batch_size=config.dataset.batch_size*4,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=4,
                persistent_workers=True
            )
            test_loaders.append(test_dataloader)
        train_set = TorchDataset(train_input, train_labels, seq_size)
        val_set = TorchDataset(val_input, val_labels, seq_size)
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,
            num_workers=4
        )
    
    elif dataset_type == "COMBINED":
        print("Loading COMBINED dataset...")
        train_input, train_labels = combined_load(cst.DATA_DIR + "/COMBINED/train.npy", cst.LEN_SMOOTH, horizon, seq_size=seq_size)
        val_input, val_labels = combined_load(cst.DATA_DIR + "/COMBINED/val.npy", cst.LEN_SMOOTH, horizon, seq_size=seq_size)
        test_input, test_labels = combined_load(cst.DATA_DIR + "/COMBINED/test.npy", cst.LEN_SMOOTH, horizon, seq_size=seq_size)
        train_set = TorchDataset(train_input, train_labels, seq_size)
        val_set = TorchDataset(val_input, val_labels, seq_size)
        test_set = TorchDataset(test_input, test_labels, seq_size)
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size*4,
            num_workers=4
        )
        test_loaders = [data_module.test_dataloader()]
        print("COMBINED dataset loaded successfully...")
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    counts_train = torch.unique(train_labels, return_counts=True)
    counts_val = torch.unique(val_labels, return_counts=True)
    counts_test = torch.unique(test_labels, return_counts=True)
    print()
    print("Train set shape: ", train_input.shape)
    print("Val set shape: ", val_input.shape)
    print("Test set shape: ", test_input.shape)
    class_names = {0: "up", 1: "stat", 2: "down"}
    train_dist = {class_names[i] if i in class_names else f"class_{i}": count.item() / train_labels.shape[0] for i, count in enumerate(counts_train[1])}
    val_dist = {class_names[i] if i in class_names else f"class_{i}": count.item() / val_labels.shape[0] for i, count in enumerate(counts_val[1])}
    test_dist = {class_names[i] if i in class_names else f"class_{i}": count.item() / test_labels.shape[0] for i, count in enumerate(counts_test[1])}
    print(f"Classes distribution in train set: {', '.join([f'{k}: {v:.2f}' for k, v in train_dist.items()])}")
    print(f"Classes distribution in val set: {', '.join([f'{k}: {v:.2f}' for k, v in val_dist.items()])}")
    print(f"Classes distribution in test set: {', '.join([f'{k}: {v:.2f}' for k, v in test_dist.items()])}")
    print()
    
    experiment_type = config.experiment.type
    num_classes = config.model.hyperparameters_fixed.get("num_classes", 3)
    print(f"Number of classes (from config): {num_classes}")
    print("Data loaded, about to initialize/load model...")

    if "FINETUNING" in experiment_type or "EVALUATION" in experiment_type:
        # Always initialize a new model for evaluation, do not load from checkpoint
        print("Initializing new model for evaluation (checkpoint loading disabled)")
        model_type = config.model.type
        if model_type == cst.ModelType.TLOB:
            model = Engine(
                seq_size=config.model.hyperparameters_fixed["seq_size"],
                horizon=config.experiment.horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=config.model.hyperparameters_fixed["num_heads"],
                is_sin_emb=config.model.hyperparameters_fixed["is_sin_emb"],
                num_classes=num_classes,
                len_test_dataloader=len(test_loaders[0])
            )
        else:
            raise ValueError(f"Unsupported model type {model_type} for new initialization")
    else:
        if model_type == cst.ModelType.MLPLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_classes=num_classes,
                len_test_dataloader=len(test_loaders[0])
            )
            model = model.to(cst.DEVICE)
        elif model_type == cst.ModelType.TLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=config.model.hyperparameters_fixed["num_heads"],
                is_sin_emb=config.model.hyperparameters_fixed["is_sin_emb"],
                num_classes=num_classes,
                len_test_dataloader=len(test_loaders[0])
            )
            model = model.to(cst.DEVICE)
        elif model_type == cst.ModelType.BINCTABL:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_classes=num_classes,
                len_test_dataloader=len(test_loaders[0])
            )
            model = model.to(cst.DEVICE)
        elif model_type == cst.ModelType.DEEPLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_classes=num_classes,
                len_test_dataloader=len(test_loaders[0])
            )
            model = model.to(cst.DEVICE)
    print("Model initialized/loaded.")    
    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))   
    train_dataloader, val_dataloader = data_module.train_dataloader(), data_module.val_dataloader()
    print("Dataloaders created.")

    if "TRAINING" in experiment_type or "FINETUNING" in experiment_type:
        trainer.fit(model, train_dataloader, val_dataloader)
        best_model_path = model.last_path_ckpt
        print("Best model path: ", best_model_path) 
        try:
            best_model = Engine.load_from_checkpoint(best_model_path, map_location=cst.DEVICE, num_classes=num_classes)
        except: 
            print("no checkpoints has been saved, selecting the last model")
            best_model = model
        best_model.experiment_type = ["EVALUATION"]
        for i in range(len(test_loaders)):
            test_dataloader = test_loaders[i]
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs, labels = batch
                    model = model.to(inputs.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    if run is not None and dataset_type == "LOBSTER":
                        run.log({f"f1 {testing_stocks[i]} best": outputs[0]["f1_score"]}, commit=False)
                    elif run is not None and dataset_type == "FI_2010":
                        run.log({f"f1 FI_2010 ": outputs[0]["f1_score"]}, commit=False)
                    elif run is not None and dataset_type == "COMBINED":
                        run.log({f"f1 COMBINED best": outputs[0]["f1_score"]}, commit=False)

                    if dataset_type == "COMBINED" and run is not None:
                        all_preds = []
                        all_labels = []
                        total_loss = 0
                        total_samples = 0
                        for batch in test_dataloader:
                            inputs, labels = batch
                            model = model.to(inputs.device)
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            all_preds.extend(predicted.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                            loss = torch.nn.functional.cross_entropy(outputs, labels)
                            total_loss += loss.item() * inputs.size(0)
                            total_samples += inputs.size(0)

                        # Compute metrics manually
                        accuracy = accuracy_score(all_labels, all_preds)
                        f1 = f1_score(all_labels, all_preds, average='weighted')
                        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

                        # Log a table of predictions and true labels to WandB (first 100 samples)
                        if run is not None:
                            table = wandb.Table(columns=["predicted", "true"])
                            for pred, true in zip(all_preds[:100], all_labels[:100]):
                                table.add_data(pred, true)
                            run.log({"predictions_table": table})

                            run.log({
                                "test_accuracy": accuracy,
                                "test_loss": avg_loss,
                                "f1_COMBINED_best": f1
                            }, commit=False)

                            cm = confusion_matrix(all_labels, all_preds)
                            run.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_labels, preds=all_preds)})

                        print(f"Evaluation for COMBINED - Accuracy: {accuracy}, Loss: {avg_loss}, F1: {f1}")
    else:
        for i in range(len(test_loaders)):
            test_dataloader = test_loaders[i]
            with torch.no_grad():
                for batch in test_dataloader:
                    inputs, labels = batch
                    model = model.to(inputs.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    if run is not None and dataset_type == "LOBSTER":
                        run.log({f"f1 {testing_stocks[i]} best": outputs[0]["f1_score"]}, commit=False)
                    elif run is not None and dataset_type == "FI_2010":
                        run.log({f"f1 FI_2010 ": outputs[0]["f1_score"]}, commit=False)
                    elif run is not None and dataset_type == "COMBINED":
                        print("Test output:", outputs)
                        # Compute metrics manually here if needed

            if dataset_type == "COMBINED" and run is not None:
                model.eval()
                all_preds = []
                all_labels = []
                total_loss = 0
                total_samples = 0
                with torch.no_grad():
                    for batch in test_dataloader:
                        inputs, labels = batch
                        model = model.to(inputs.device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        loss = torch.nn.functional.cross_entropy(outputs, labels)
                        total_loss += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)

                # Compute metrics manually
                accuracy = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='weighted')
                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

                # Log a table of predictions and true labels to WandB (first 100 samples)
                if run is not None:
                    table = wandb.Table(columns=["predicted", "true"])
                    for pred, true in zip(all_preds[:100], all_labels[:100]):
                        table.add_data(pred, true)
                    run.log({"predictions_table": table})

                    run.log({
                        "test_accuracy": accuracy,
                        "test_loss": avg_loss,
                        "f1_COMBINED_best": f1
                    }, commit=False)

                    cm = confusion_matrix(all_labels, all_preds)
                    run.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_labels, preds=all_preds)})

                print(f"Evaluation for COMBINED - Accuracy: {accuracy}, Loss: {avg_loss}, F1: {f1}")

def run_wandb(config: Config, accelerator):
    def wandb_sweep_callback():
        print("Entered run_wandb")
        wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model=False, save_dir=cst.DIR_SAVED_MODEL)
        run_name = None
        if not config.experiment.is_sweep:
            run_name = ""
            for param in config.model.hyperparameters_fixed.keys():
                value = config.model.hyperparameters_fixed[param]
                if param == "hyperparameters_sweep":
                    continue
                if type(value) == omegaconf.dictconfig.DictConfig:
                    for key in value.keys():
                        run_name += str(key[:2]) + "_" + str(value[key]) + "_"
                else:
                    run_name += str(param[:2]) + "_" + str(value) + "_"

            print("Starting WandB run initialization...")
            run = wandb.init(project=cst.PROJECT_NAME, name=run_name, entity="")
        
            if config.experiment.is_sweep:
                model_params = run.config
            else:
                model_params = config.model.hyperparameters_fixed
            wandb_instance_name = f"{config.dataset.type.value}_{config.model.type.value}_horizon_{config.experiment.horizon}_seed_{config.experiment.seed}"
            for param in config.model.hyperparameters_fixed.keys():
                if param in model_params:
                    config.model.hyperparameters_fixed[param] = model_params[param]
                    wandb_instance_name += f"_{param}_{model_params[param]}"

            print(f"Setting run name to {wandb_instance_name}...")
            run.name = wandb_instance_name
            seq_size = config.model.hyperparameters_fixed["seq_size"]
            horizon = config.experiment.horizon
            dataset = config.dataset.type.value
            seed = config.experiment.seed
            if dataset == "LOBSTER":
                training_stocks = config.dataset.training_stocks
                config.experiment.dir_ckpt = f"{dataset}_{training_stocks}_seq_size_{seq_size}_horizon_{horizon}_seed_{seed}"
            else:
                config.experiment.dir_ckpt = f"{dataset}_seq_size_{seq_size}_horizon_{horizon}_seed_{seed}"
        
            print("Configuring Trainer...")
            trainer = L.Trainer(
                accelerator=accelerator,
                precision=cst.PRECISION,
                max_epochs=config.experiment.max_epochs,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", patience=1, verbose=True, min_delta=0.002),
                    TQDMProgressBar(refresh_rate=1000)
                ],
                num_sanity_val_steps=0,
                logger=wandb_logger,
                detect_anomaly=False,
                check_val_every_n_epoch=1,
            )

            print("Logging configuration to WandB...")
            run.log({"model": config.model.type.value}, commit=False)
            run.log({"dataset": config.dataset.type.value}, commit=False)
            run.log({"seed": config.experiment.seed}, commit=False)
            run.log({"all_features": config.model.hyperparameters_fixed["all_features"]}, commit=False)
            if config.dataset.type == cst.DatasetType.LOBSTER:
                for i in range(len(config.dataset.training_stocks)):
                    run.log({f"training stock{i}": config.dataset.training_stocks[i]}, commit=False)
                for i in range(len(config.dataset.testing_stocks)):
                    run.log({f"testing stock{i}": config.dataset.testing_stocks[i]}, commit=False)
                run.log({"sampling_type": config.dataset.sampling_type.value}, commit=False)
                if config.dataset.sampling_type == SamplingType.TIME:
                    run.log({"sampling_time": config.dataset.sampling_time}, commit=False)
                elif config.dataset.sampling_type == SamplingType.QUANTITY:
                    run.log({"sampling_quantity": config.dataset.sampling_quantity}, commit=False)
            print("Calling train function...")
            train(config, trainer, run)
            print("Train function finished.")
            run.finish()
            print("WandB run finished.")
            return

    wandb_sweep_callback()  #Make sure this is called!

def sweep_init(config: Config):
    wandb.login("")
    parameters = {}
    for key in config.model.hyperparameters_sweep.keys():
        parameters[key] = {'values': list(config.model.hyperparameters_sweep[key])}
        sweep_config = {
            'method': 'grid',
            'metric': {
                'goal': 'minimize',
                'name': 'val_loss'
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 3,
                'eta': 1.5
            },
            'run_cap': 100,
            'parameters': {**parameters}
        }
        return sweep_config

def print_setup(config: Config):
    print("Model type: ", config.model.type)
    print("Dataset: ", config.dataset.type)
    print("Seed: ", config.experiment.seed)
    print("Sequence size: ", config.model.hyperparameters_fixed["seq_size"])
    print("Horizon: ", config.experiment.horizon)
    print("All features: ", config.model.hyperparameters_fixed["all_features"])
    print("Is data preprocessed: ", config.experiment.is_data_preprocessed)
    print("Is wandb: ", config.experiment.is_wandb)
    print("Is sweep: ", config.experiment.is_sweep)
    print(config.experiment.type)
    print("Is debug: ", config.experiment.is_debug) 
    if config.dataset.type == cst.DatasetType.LOBSTER:
        print("Training stocks: ", config.dataset.training_stocks)
        print("Testing stocks: ", config.dataset.testing_stocks)