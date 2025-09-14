from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import create_model
import deepspeed
import wandb
import numpy as np
from prepare_deepspeed import prepare_deepspeed
import os

from pmc import sample_responses, extract, prepare_responses


class UnlearningTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.hparams = kwargs.pop('hparams')
        self.loss = self.hparams['unlearning_loss']
        tokenizer = kwargs.pop('tokenizer')

        super(UnlearningTrainer, self).__init__(*args, **kwargs)
        self._tokenizer = tokenizer

        if self.loss in ['npo']:
            _, self.ref_model = create_model(self.hparams)
            self.ref_model = self._prepare_deepspeed(self.ref_model.cpu())

        self.sampled_data = {}

        self.losses = {}
        self.scores = {}
        self.picked_scores = {}  # scores of picked answer
        self.avg_score_history = []
        self.generations = []
        self.gt_answers = []
        self.model_answers = []

    def compute_loss(self, model, inputs,
                     return_outputs=False, num_items_in_batch=None):
        (forget_inputs, retain_inputs, forget_idx,
            forget_questions, forget_answers, model_answers) = inputs

        # unlearning losses
        if self.loss == "grad_ascent":
            loss = self.compute_model_loss(model, forget_inputs) * -1

        elif self.loss == "grad_diff":
            forget_loss = self.compute_model_loss(model, forget_inputs)
            retain_loss = self.compute_model_loss(model, retain_inputs)
            loss = self.hparams['lambda'] * retain_loss - forget_loss

        elif self.loss == 'idk':
            # forget targets replaced with "I don't know." in dataloader
            forget_loss = self.compute_model_loss(model, forget_inputs)
            retain_loss = self.compute_model_loss(model, retain_inputs)
            loss = self.hparams['lambda'] * retain_loss + forget_loss
        elif self.loss == 'npo':
            forget_loss = self.compute_batch_loss(model, forget_inputs)

            with torch.no_grad():
                forget_loss_reference = self.compute_batch_loss(
                    self.ref_model,
                    forget_inputs
                )

            beta = self.hparams['beta']
            neg_log_ratios = forget_loss - forget_loss_reference
            forget_loss = F.logsigmoid(beta * neg_log_ratios).mean()
            forget_loss = - 2 * forget_loss / beta

            retain_loss = self.compute_model_loss(model, retain_inputs)
            loss = self.hparams['lambda'] * retain_loss + forget_loss

        elif self.loss == 'simnpo':
            _, labels, _ = forget_inputs
            loss_mask = labels != -100

            gamma = self.hparams['gamma']
            beta = self.hparams['beta']
            forget_loss = self.compute_batch_loss(model, forget_inputs)
            forget_loss = forget_loss / loss_mask.sum(-1) - gamma
            forget_loss = F.logsigmoid(beta * forget_loss).mean()
            forget_loss = - 2 * forget_loss / beta

            retain_loss = self.compute_model_loss(model, retain_inputs)
            loss = self.hparams['lambda'] * retain_loss + forget_loss

        elif self.loss == 'pmc':
            num_samples = self.hparams['num_samples']

            generation_inputs, scores, picked_scores = sample_responses(
                forget_inputs,
                forget_idx,
                forget_questions,
                model_answers,
                self.sampled_data,
                self.hparams,
                self.model,
                self._tokenizer,
                num_samples=num_samples,
            )

            self.track_generations(forget_idx, forget_questions,
                                   forget_answers, model_answers,
                                   scores, picked_scores)
            collapse_loss = self.compute_model_loss(model, generation_inputs)

            # forget loss just for tracking
            with torch.no_grad():
                forget_loss = self.compute_model_loss(model, forget_inputs)

            retain_loss = self.compute_model_loss(model, retain_inputs)

            loss = self.hparams['lambda'] * retain_loss + collapse_loss

            self.track_loss("0_loss (↓)", loss)
            self.track_loss("1_retain_loss (↓)", retain_loss)
            self.track_loss("2_forget_loss (↑)", forget_loss)
            self.track_loss("3_collapse_loss (↓)", collapse_loss)

        else:
            raise ValueError(f"Unrecognized loss function: {self.loss}. ")

        return (loss, None) if return_outputs else loss

    def on_step_end(self):
        if self.loss != 'pmc':
            return

        self.report_losses()
        self.report_scores()

    def report_scores(self):
        if not self.hparams['logging']:
            return

        for metric in self.scores:

            path = f"generations/{metric}/"
            directory = os.path.join(self.args.output_dir, path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            assert len(self.generations) == len(self.scores[metric])
            path = directory + str(self.state.global_step) + ".txt"
            with open(path, "w") as f:
                f.write(f"\nAverage {metric} of all samples: "
                        + str(np.round(np.mean(self.scores[metric]), 4))
                        + "\n")
                min_score = np.min(self.scores[metric], axis=1)
                f.write(f"\nAverage {metric} of min: "
                        + str(np.round(np.mean(min_score), 4))
                        + "\n")
                picked_score = self.picked_scores[metric]
                f.write(f"\nAverage {metric} of selected: "
                        + str(np.round(np.mean(picked_score), 4))
                        + "\n")
                f.write("-----------------------\n")

                iterator = zip(self.generations,
                               self.gt_answers,
                               self.model_answers,
                               self.scores[metric],
                               self.picked_scores[metric])
                for gen, gt, ma, score, picked_score in iterator:
                    line = "\n>>>>\n"
                    line += str([f"{np.round(s, 2):.2f}" for s in score])
                    line += f"\nMin: {np.round(min(score), 4):.4f}"
                    line += f"\nSelected: {np.round(picked_score, 4):.4f}\n\n"
                    line += "Question:\n\t" + gen + "\n"
                    line += "Initial model:\n\t" + ma + "\n"
                    line += "Ground-truth:\n\t" + gt + "\n"
                    f.write(line)

            reported_score = np.mean(self.picked_scores[metric])
            wandb.log({f"avg_{metric}_selected": reported_score},
                      step=self.state.global_step)

            path = os.path.join(self.args.output_dir, f"{metric}_overview.txt")
            with open(path, "a") as f:
                f.write(f"{np.round(reported_score, 4):.4f}\n")

        self.scores = {}
        self.picked_scores = {}
        self.generations = []
        self.gt_answers = []
        self.model_answers = []

    def report_losses(self):
        wandb.log({loss_name: np.mean(self.losses[loss_name])
                   for loss_name in self.losses},
                  step=self.state.global_step)
        self.losses = {}

    def track_generations(self, forget_idx, forget_questions, gt_answers,
                          model_answers, scores, picked_scores):
        if not self.hparams['logging']:
            return

        inputs = [forget_questions[i] +
                  "\nSelected sampled response:\n\t" +
                  self.sampled_data[idx]['answer']
                  for i, idx in enumerate(forget_idx)]
        self.generations.extend(inputs)
        self.gt_answers.extend(gt_answers)
        self.model_answers.extend(model_answers)

        for metric in scores:
            if metric not in self.scores:
                self.scores[metric] = []
            self.scores[metric].extend(scores[metric].tolist())

        for metric in picked_scores:
            if metric not in self.picked_scores:
                self.picked_scores[metric] = []
            self.picked_scores[metric].extend(picked_scores[metric].tolist())

    def track_loss(self, loss_name, loss):
        if loss_name not in self.losses:
            self.losses[loss_name] = []
        self.losses[loss_name].append(loss.item())

    def compute_model_loss(self, model, inputs):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels,
                        attention_mask=attention_mask)
        return outputs.loss

    def compute_batch_loss(self, model, inputs, cpu=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels,
                        attention_mask=attention_mask).logits
        shifted_labels = labels[..., 1:].contiguous()
        outputs = outputs[..., :-1, :].contiguous()

        loss_fun = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fun(outputs.transpose(-1, -2), shifted_labels).sum(dim=-1)
        return loss

    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        return prepare_deepspeed(model, deepspeed_plugin)
