import torch


def preprocess_dataset(dataset, tokenizer, hparams):
    def prep(x): return preprocess(hparams, tokenizer, x)
    return dataset.map(prep, remove_columns=dataset.column_names)


def preprocess(hparams, tokenizer, x):

    if 'Instruct' in hparams['model']:
        return preprocess_instruct(hparams, tokenizer, x)

    qst = hparams["question_start_tag"]
    qet = hparams["question_end_tag"]
    question = qst + x['question'] + qet
    answer = hparams["answer_tag"] + x['answer']
    text = question + answer + tokenizer.eos_token

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


def preprocess_instruct(hparams, tokenizer, x):
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

    labels = [-100] * question_length
    labels += chat_ids[question_length:chat_length]
    labels += [-100] * (hparams['max_length'] - chat_length)
    attention_mask = [1] * chat_length
    attention_mask += [0] * (hparams['max_length'] - chat_length)

    result = {"input_ids": chat_ids,
              "labels": labels,
              "attention_mask": attention_mask}
    return result
