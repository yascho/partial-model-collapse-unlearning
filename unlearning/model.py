from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch


def create_model(hparams):
    tokenizer = AutoTokenizer.from_pretrained(hparams['tokenizer_model'])

    if 'llama' in hparams['tokenizer_model']:
        tokenizer.add_special_tokens({'pad_token': '***'})
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model_path = os.path.join(hparams['model'], str(hparams['seed']))
    model_path = os.path.join(model_path, hparams['checkpoint'])
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 use_flash_attention_2=False,
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.to(hparams["device"])

    return tokenizer, model
