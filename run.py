import argparse
from utils import set_seed
import warnings
import os
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices
from control import LayoutLMModule
from data import DataModule

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
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="ood")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss", type=str, default="margin")
    args = parser.parse_args()

    data = DataModule(
        args,
        train_size=20_000,
        val_size=5_000,
        test_size=3_000,
        ood_size=3_000,
        use_from_disk=True,
        reuse_from_disk=True,
        save_parquets=True,
    )
    data.prepare_data()
    data.setup(stage="fit")

    set_seed(args)

    model = LayoutLMModule(args)

    wandb_logger = WandbLogger(project=args.project_name)
    wandb_logger.experiment.config.update(vars(args))
    # wandb_logger.watch(model, log="all")
    # Trainer Flags: precision=16,
    trainer = Trainer(
        gpus=4,
        # devices=find_usable_cuda_devices(4),
        accelerator="gpu",
        strategy="ddp_spawn",
        logger=wandb_logger,
        max_epochs=args.num_train_epochs,
    )

    trainer.fit(model, datamodule=data)
    trainer.validate(model, datamodule=data, verbose=True)
    trainer.test(model, datamodule=data, verbose=True)
    trainer.predict(model, datamodule=data)


if __name__ == "__main__":
    ### Set the enviornment varaible DATA_DIR to the directory where the data is stored
    os.chdir("/tmp/wpm")

    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512,garbage_collection_threshold:0.82"

    main()
