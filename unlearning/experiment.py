import numpy as np
import torch
import time
import random
from tqdm.auto import tqdm

from model import *
from dataloader import *
from unlearning import *
from inference import generate


class Experiment():

    def run(self, hparams, output_dir):

        datasets = load_datasets(hparams)
        tokenizer, model = create_model(hparams)

        if hparams['unlearning_loss'] == 'pmc':
            generations = generate(hparams, tokenizer, model,
                                   datasets['forget'])
            initial_responses = [x['generation'] for x in generations]
        else:
            initial_responses = None

        start = time.time()
        if hparams['unlearning']:
            set_random_seed(hparams['seed'])
            unlearn(hparams, tokenizer, model, datasets,
                    initial_responses, output_dir)

        end = time.time()
        unlearning_time = end - start

        scores = {}
        generations = {}

        start = time.time()
        for dataset_name in tqdm(datasets):

            if not hparams['full_eval'] and dataset_name == 'retain':
                continue

            dataset = datasets[dataset_name]
            full_generation = generate(hparams, tokenizer, model, dataset)

            mean_rouge = np.mean([x['eval']['rouge_score_recall']['rougeL']
                                  for x in full_generation])

            scores[dataset_name] = {
                "mean_rougeL_recall": mean_rouge,
            }

            generations[dataset_name] = {
                "generation": full_generation
            }

        k = "mean_rougeL_recall"
        scores['unlearn_quality'] = 2 - scores['forget'][k]
        scores['unlearn_quality'] -= scores['forget_paraphrased'][k]
        scores['unlearn_quality'] *= 100/2
        scores['utility'] = scores['retain_short'][k]
        scores['utility'] += scores['world_facts'][k]
        scores['utility'] += scores['real_authors'][k]
        scores['utility'] *= 100/3

        end = time.time()
        inference_time = end - start

        result = {
            "model": hparams['model'],
            "tokenizer": hparams['tokenizer_model'],
            "split": hparams['split'],
            "unlearning_time": unlearning_time,
            "inference_time": inference_time,
            "scores": scores,
            "generations": generations
        }

        return result


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
