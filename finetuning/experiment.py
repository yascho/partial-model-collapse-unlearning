from model import create_model
from datasets import load_dataset
from preprocessing import preprocess_dataset
from training import training


class Experiment():

    def run(self, hparams, output_dir):
        tokenizer, model = create_model(hparams)
        dataset = load_dataset("locuslab/TOFU", hparams['dataset'])['train']
        dataset = preprocess_dataset(dataset, tokenizer, hparams)

        training(tokenizer, model, dataset, hparams['training'], output_dir)
