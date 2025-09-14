from transformers import TrainerCallback
from unlearning_trainer import UnlearningTrainer
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm
from dataloader import *


def unlearn(hparams, tokenizer, model, datasets,
            initial_responses, output_dir):
    model.train()
    tokenizer.padding_side = "right"  # right padding for finetuning

    if hparams["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    unlearning_dataset = UnlearningDataset(hparams, tokenizer,
                                           datasets['forget'],
                                           datasets['retain'],
                                           initial_responses)

    batch_size = hparams['batch_size']
    num_epochs = hparams['num_epochs']
    gradient_accumulation_steps = hparams['gradient_accumulation_steps']
    steps_per_epoch = len(unlearning_dataset)//(batch_size *
                                                gradient_accumulation_steps)

    max_steps = int(num_epochs*len(unlearning_dataset)
                    )//(batch_size*gradient_accumulation_steps)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=hparams['learning_rate'],
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1, max_steps//20),
        logging_dir=output_dir + '/logs',
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        save_strategy="steps" if hparams['save_model'] else "no",
        save_steps=max_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        deepspeed='ds_config.json',
        weight_decay=hparams['weight_decay'],
        eval_strategy="no",
        seed=hparams['seed'],
        run_name=output_dir.replace("/", "_"),
    )

    trainer = UnlearningTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=unlearning_dataset,
        eval_dataset=unlearning_dataset,
        hparams=hparams,
        data_collator=data_collator_unlearning,
    )

    trainer.add_callback(LoggingCallback(trainer))
    trainer.train()


class LoggingCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer.on_step_end()
