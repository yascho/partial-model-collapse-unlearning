from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def create_model(hparams):
    tokenizer = AutoTokenizer.from_pretrained(hparams["model"])
    tokenizer.padding_side = "right"

    if 'llama' in hparams['model']:
        tokenizer.add_special_tokens({'pad_token': '***'})
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(hparams["model"],
                                                 use_flash_attention_2=False,
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.to(hparams["device"])

    return tokenizer, model
