import yaml
import torch
import random
import argparse
import numpy as np
from train import Trainer

if __name__ == "__main__":
    # Set the random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_config.yaml")
    parser.add_argument("--save-name", type=str, default="test")
    args = parser.parse_args()
    args = vars(args)

    # Load the configuration file
    with open(args["config"], "r") as f:
        args.update(yaml.safe_load(f))
    args = argparse.Namespace(**args)

    # Run the trainer
    Trainer(args).run()