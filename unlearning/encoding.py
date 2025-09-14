import torch


def encode_finetune(hparams, tokenizer, x, add_tags=True):
    tokenizer.padding_side = "right"  # left padding for inference

    if "Instruct" in hparams["tokenizer_model"]:
        return encode_finetune_instruct(hparams, tokenizer, x)

    if add_tags:
        qst = hparams["question_start_tag"]
        qet = hparams["question_end_tag"]
        at = hparams["answer_tag"]
    else:
        qst = ""
        qet = ""
        at = ""

    question = qst + x['question'] + qet
    answer = at + x['answer']
    text = question + answer + tokenizer.eos_token

    # Create attention mask for computing loss only on the answer tokens
    answer_start = tokenizer(question, add_special_tokens=True)
    answer_start = len(answer_start["input_ids"])

    encoded = tokenizer(text,
                        truncation=True,
                        padding='max_length',
                        max_length=hparams["max_length"],
                        add_special_tokens=True)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Mask the tokens: -100 for question tokens,
    # valid token IDs for answer tokens
    labels = [-100
              if (i < answer_start or token == tokenizer.pad_token_id)
              else token
              for i, token in enumerate(input_ids)]

    result = {"input_ids": input_ids,
              "labels": labels,
              "attention_mask": attention_mask}
    return result


def encode_finetune_instruct(hparams, tokenizer, x):

    system_prompt = hparams["system_prompt"]
    date_string = hparams['date_string']

    chat = [{"role": "system", "content": system_prompt}]
    chat += [{"role": "user", "content": x['question']}]
    chat += [{"role": "assistant", "content": x['answer']}]

    chat_ids = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        padding='max_length',
        max_length=hparams['max_length'],
        add_generation_prompt=False,
        date_string=date_string
    )

    question_length = len(tokenizer.apply_chat_template(
        chat[:-1],
        tokenize=True,
        add_generation_prompt=True,
        date_string=date_string
    ))

    chat_length = len(tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=False,
        date_string=date_string
    ))

    max_length = hparams['max_length']
    if chat_length > max_length:
        # for the case where decoding + encoding doesn't properly work
        chat_length = min(chat_length, max_length)
        chat_ids = chat_ids[:max_length-1] + [chat_ids[-1]]  # eot token

    labels = [-100] * question_length
    labels += chat_ids[question_length:chat_length]
    labels += [-100] * (hparams['max_length'] - chat_length)
    attention_mask = [1] * chat_length
    attention_mask += [0] * (hparams['max_length'] - chat_length)

    result = {"input_ids": chat_ids,
              "labels": labels,
              "attention_mask": attention_mask}
    return result


def encode_inference(hparams, tokenizer, questions):
    tokenizer.padding_side = "left"  # left padding for inference

    if 'Instruct' in hparams['tokenizer_model']:
        return encode_inference_instruct(hparams, tokenizer, questions)

    questions = [
        hparams["question_start_tag"] + q +
        hparams["question_end_tag"]
        for q in questions
    ]

    inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True
    ).to(hparams['device'])

    return inputs


def encode_inference_instruct(hparams, tokenizer, questions):
    system_prompt = hparams["system_prompt"]
    date_string = hparams['date_string']

    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            date_string=date_string
        )
        for q in questions
    ]

    inputs = tokenizer(
        inputs,
        padding=True,
        return_tensors="pt"
    ).to(hparams["device"])

    return inputs
