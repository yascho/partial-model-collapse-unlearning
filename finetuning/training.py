from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def training(tokenizer, model, dataset, hparams, output_dir):
    model.config.use_cache = False

    if hparams["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    max_steps = hparams['batch_size']*hparams['gradient_accumulation_steps']
    max_steps = int(hparams['num_epochs']*len(dataset)) // max_steps
    use_wandb = os.environ.get('WANDB_DISABLED', 'false').lower() == 'false'

    training_args = TrainingArguments(
        per_device_train_batch_size=hparams['batch_size'],
        per_device_eval_batch_size=hparams['batch_size'],
        gradient_accumulation_steps=hparams['gradient_accumulation_steps'],
        learning_rate=hparams['learning_rate'],
        warmup_steps=max(1, max_steps//hparams['num_epochs']),
        max_steps=max_steps,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1, max_steps//20),
        logging_dir=output_dir + "/logs",
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        save_only_model=True,
        save_steps=max_steps,
        ddp_find_unused_parameters=False,
        eval_strategy="no",
        weight_decay=hparams['weight_decay'],
        seed=hparams['seed'],
        report_to="wandb" if use_wandb else None,
        deepspeed='ds_config.json',
        run_name=output_dir.replace("/", "_"),
        eval_on_start=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    trainer.train()

    model.config.use_cache = True
