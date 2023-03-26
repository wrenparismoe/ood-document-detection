import argparse
import os
import warnings
import socket

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.loggers import WandbLogger

from control import LayoutLMModule
from data import DataModule
from utils import set_seed

# from lightning.pytorch.strategy import DDPSPawnStrategy

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", default="microsoft/layoutlm-base-uncased", type=str
    )
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--task_name", default="rvl_cdip", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="ood")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss", type=str, default="margin")
    # add arguments for train, val, test, ood sizes
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--val_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--ood_size", type=int, default=None)
    parser.add_argument("--use_from_disk", action="store_true", help="Reuse data from disk")
    parser.add_argument("--save_parquets", action="store_true", help="Save data to parquet")
    parser.add_argument("--use_parquets", action="store_true", help="Load data from parquet")
    parser.add_argument("--process_data_only", action="store_true", help="Exit after processing data")
    parser.add_argument("--save_to_disk", action="store_true", help="Save data to disk")
    args = parser.parse_args()

    hostname = socket.gethostname()
    hostslice = hostname.split("-")[0]
    if hostslice == "wpm":
        args.local = True
    elif hostslice == "klone":
        os.chdir("/tmp/wpm")
        args.local = False
    else:
        args.local = None



    data = DataModule(
        args, 
    )
    print("Processing data")
    data.prepare_data()
    if args.process_data_only:
        print("Exiting after processing data")
        return
    print("Loading data")
    data.setup(stage="fit")

    set_seed(args)
    print("Initializing model")
    model = LayoutLMModule(args)

    wandb_logger = WandbLogger(project=args.project_name)
    #wandb_logger.experiment.config.update(vars(args))
    #wandb_logger.watch(model, log="all")
    # Trainer Flags: precision=16,
    trainer = Trainer(
        devices=find_usable_cuda_devices(1),
        accelerator="gpu",
        # strategy="ddp_spawn",
        logger=wandb_logger,
        max_epochs=args.num_train_epochs,
        #benchmark=True,
        precision=16
    )
    print("Running trainer.fit()")
    trainer.fit(model, datamodule=data)
    #trainer.validate(model, datamodule=data, verbose=True)
    print("Running trainer.test()")
    trainer.test(model, datamodule=data, verbose=True)
    print("Running trainer.predict()")
    trainer.predict(model, datamodule=data)


if __name__ == "__main__":
       # Set the enviornment varaible DATA_DIR to the directory where the data is stored
    #os.chdir("/tmp/wpm")

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512,garbage_collection_threshold:0.80"

    main()
