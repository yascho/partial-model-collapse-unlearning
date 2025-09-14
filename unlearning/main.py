import hydra
from experiment import Experiment
import os
import json
import torch
import wandb
import numpy as np
from hydra.core.hydra_config import HydraConfig
from huggingface_hub import login
from dotenv import load_dotenv
from exception_handling import print_exceptions


@print_exceptions
@hydra.main(version_base=None, config_path="./")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    os.environ["HYDRA_FULL_ERROR"] = "1"
    login(token=os.environ['HUGGINGFACE_LOGIN_TOKEN'])

    # check if file exists
    result_file_path = output_dir + "/results/result.json"
    if os.path.isfile(result_file_path):
        log.info(f"Result file already exists: {result_file_path}")
        return  # do not run experiment twice

    mode = "disabled" if os.environ['WANDB_DISABLED'] == 'true' else "online"
    wandb.init(project=os.environ['WANDB_PROJECT'],
               name=output_dir.replace("/", "_"),
               mode=mode)
    exp = Experiment()
    result = exp.run(cfg['hparams'], output_dir)
    wandb.finish()

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_file_path = os.path.join(results_dir, "result.json")

    with open(result_file_path, "w") as json_file:
        json.dump(result, json_file, indent=4)


if __name__ == "__main__":
    load_dotenv("environment.env")
    main()
