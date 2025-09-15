from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from encoding import encode_finetune, encode_inference
import numpy as np
import torch
import copy


def load_datasets(hparams):
    datasets = {}

    if hparams['split'] == 'full':
        datasets['full'] = load_dataset("locuslab/TOFU", "full")
    else:
        forget_split = 'forget' + f"{hparams['split']:02}"
        retain_split = 'retain' + f"{100-hparams['split']}"

        print(f"Loading dataset split: {retain_split} and {forget_split}")
        datasets['forget'] = load_dataset("locuslab/TOFU", forget_split)
        datasets['forget_paraphrased'] = load_dataset(
            "locuslab/TOFU", forget_split + "_perturbed")

        datasets['retain'] = load_dataset("locuslab/TOFU", retain_split)

    datasets['world_facts'] = load_dataset("locuslab/TOFU", 'world_facts')
    datasets['real_authors'] = load_dataset("locuslab/TOFU", 'real_authors')

    final_datasets = {}
    for dataset_name in datasets:
        dataset = datasets[dataset_name]['train']
        final_datasets[dataset_name] = dataset

        if not hparams['full_eval'] and len(dataset) > 400 and \
                'retain' in dataset_name:
            dataset = copy.deepcopy(dataset)
            rng = np.random.default_rng(12345)
            rints = rng.integers(low=0, high=len(dataset), size=400)
            dataset = dataset.select(rints)
            final_datasets[dataset_name + '_short'] = dataset

        if 'paraphrased' in dataset_name:
            new = final_datasets[dataset_name].map(modify_paraphrased_sample)
            final_datasets[dataset_name] = new

    return final_datasets


def modify_paraphrased_sample(sample):
    sample['question'] = sample['paraphrased_question']
    sample['answer'] = sample['paraphrased_answer']
    return sample


class UnlearningDataset(Dataset):
    def __init__(self, hparams, tokenizer, forget_data, retain_data,
                 initial_responses):
        super(UnlearningDataset, self).__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.initial_responses = initial_responses

        self.sampled_answers = {}

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):

        result = []
        for data_type in ["forget", "alignment", "retain"]:

            if data_type in ["forget", "alignment"]:
                data = self.forget_data
            elif data_type == "retain":
                data = self.retain_data
                randint = torch.randint(0, len(self.retain_data), (1,)).item()
                idx = (idx + randint) % len(self.retain_data)

            if self.hparams['unlearning'] and \
                    self.hparams['unlearning_loss'] == 'idk' and \
                    data_type == "forget":
                x = {"question": data[idx]['question'],
                     "answer": "I don't know."}
            elif self.hparams['align'] and data_type == "alignment":
                answers = self.hparams['alignment_responses']
                answer = np.random.choice(answers)
                x = {"question": data[idx]['question'],
                     "answer": answer}
            else:
                x = {"question": data[idx]['question'],
                     "answer": data[idx]['answer']}

            converted_data = encode_finetune(self.hparams, self.tokenizer, x)

            unlearning_loss = self.hparams['unlearning_loss']
            if data_type == "forget" and unlearning_loss == 'pmc':
                model_answer = self.initial_responses[idx]
            else:
                model_answer = None

            if data_type == "forget":
                result.extend([idx, converted_data,
                               x['question'], x['answer'], model_answer])
            else:
                result.append(converted_data)

        return result


def data_collator_unlearning(samples):
    forget_idx = [sample[0] for sample in samples]
    forget_samples = [sample[1] for sample in samples]
    forget_questions = [sample[2] for sample in samples]
    forget_answers = [sample[3] for sample in samples]
    model_answers = [sample[4] for sample in samples]
    alignment_samples = [sample[5] for sample in samples]
    retain_samples = [sample[6] for sample in samples]

    res = []
    for data_type in ["forget", "retain", "alignment"]:
        if data_type == "forget":
            data = forget_samples
        elif data_type == "retain":
            data = retain_samples
        else:
            data = alignment_samples

        input_ids = [torch.tensor(s['input_ids']) for s in data]
        labels = [torch.tensor(s['labels']) for s in data]
        attention_mask = [torch.tensor(s['attention_mask']) for s in data]

        res.append((torch.stack(input_ids), torch.stack(
            labels), torch.stack(attention_mask)))
    return res + [forget_idx, forget_questions, forget_answers, model_answers]
