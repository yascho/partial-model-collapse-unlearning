import hydra
from experiment import Experiment
import os
import wandb
from hydra.core.hydra_config import HydraConfig
from huggingface_hub import login
from dotenv import load_dotenv


@hydra.main(version_base=None, config_path="./")
def main(cfg):

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    wandbproject = os.environ.get('WANDB_FT_PROJECT', 'finetuning')
    wandb_mode = os.environ.get('WANDB_DISABLED', 'false')
    wandb_mode = 'disabled' if wandb_mode.lower() == 'true' else 'online'
    login(token=os.environ['HUGGINGFACE_LOGIN_TOKEN'])

    wandb.init(project=wandbproject,
               name=output_dir.replace("/", "_"),
               mode=wandb_mode)
    exp = Experiment()
    exp.run(cfg['hparams'], output_dir)
    wandb.finish()


if __name__ == "__main__":
    load_dotenv("environment.env")
    main()
