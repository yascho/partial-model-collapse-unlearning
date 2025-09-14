from rouge_score.rouge_scorer import RougeScorer
from encoding import encode_inference, encode_finetune
import numpy as np
import torch
from tqdm.auto import tqdm
import evaluate


def sample_responses(forget_inputs, forget_idx, forget_questions,
                     model_answers, sampled_data, hparams, model,
                     tokenizer, num_samples):
    input_ids, labels, attention_mask = forget_inputs

    resampling_idx = []
    resampling_questions = []
    skipped_idx = []

    for i in range(len(forget_questions)):
        if forget_idx[i] not in sampled_data:
            resampling_idx.append(i)
            resampling_questions.append(forget_questions[i])
            continue

        previous_answer = sampled_data[forget_idx[i]]

        if 'score' in previous_answer and previous_answer['score'] > 0:
            resampling_idx.append(i)
            resampling_questions.append(forget_questions[i])

        if 'score' in previous_answer and previous_answer['score'] == 0:
            skipped_idx.append(i)

    key = hparams['pmc_selection']
    if len(resampling_idx) > 0:
        generated_answers = generate_answers(hparams,
                                             model,
                                             tokenizer,
                                             resampling_questions,
                                             num_samples)

        answers = [model_answers[i] for i in resampling_idx]
        picked_answers, scores, picked_scores = pick_answers(hparams,
                                                             answers,
                                                             generated_answers,
                                                             num_samples)

        for i in range(len(picked_answers)):
            if forget_idx[resampling_idx[i]] in sampled_data:
                previous_answer = sampled_data[forget_idx[resampling_idx[i]]]

                if picked_scores[key][i] > previous_answer['score']:
                    picked_scores[key][i] = previous_answer['score']
                    continue

            sampled_data[forget_idx[resampling_idx[i]]] = {
                'answer': picked_answers[i],
                'score': picked_scores[key][i]
            }
    else:
        picked_scores = {key: np.array([])}
        scores = {key: np.empty((0, num_samples))}

    for metric in [key]:
        for idx in skipped_idx:
            picked_scores[metric] = np.insert(picked_scores[metric], idx, 0.0)
            scores[metric] = np.insert(scores[metric], idx, 0.0, axis=0)

    responses = prepare_responses(hparams,
                                  tokenizer,
                                  forget_idx,
                                  forget_questions,
                                  sampled_data)

    return responses, scores, picked_scores


def prepare_responses(hparams, tokenizer, forget_idx, questions, sampled_data):
    data = []
    for i in range(len(questions)):
        x = {'question': questions[i],
             'answer': sampled_data[forget_idx[i]]['answer']}
        x = encode_finetune(hparams, tokenizer, x)
        data.append(x)

    input_ids = torch.stack([torch.tensor(s['input_ids']) for s in data])
    input_ids = input_ids.to(hparams['device'])
    labels = torch.stack([torch.tensor(s['labels']) for s in data])
    labels = labels.to(hparams['device'])
    attention_mask = torch.stack([torch.tensor(s['attention_mask'])
                                  for s in data])
    attention_mask = attention_mask.to(hparams['device'])

    return input_ids, labels, attention_mask


def pick_answers(hparams, initial_model_answers,
                 generated_answers, num_samples):
    scores = {}
    scores['rouge_reward'] = reward_function(initial_model_answers,
                                             generated_answers,
                                             num_samples,
                                             hparams['min_len'])

    selected_scores = scores[hparams['pmc_selection']]
    pick = selected_scores.argmin(axis=1)

    idx = np.arange(selected_scores.shape[0])
    picked_scores = {metric: selected_scores[idx, pick] for metric in scores}

    picked_answers = np.array(generated_answers)
    picked_answers = picked_answers.reshape(-1, num_samples)
    picked_answers = picked_answers[np.arange(picked_answers.shape[0]), pick]
    return picked_answers.tolist(), scores, picked_scores


def sample_selection(selected_scores, temp=1):
    selected_scores = 1-torch.tensor(selected_scores)
    selected_scores = torch.softmax(selected_scores/temp, dim=1)
    pick = []
    for i in range(selected_scores.shape[0]):
        p = selected_scores[i, :]
        p = np.asarray(selected_scores[i, :]).astype('float64')
        p = p / np.sum(p)
        pick.append(np.random.choice(selected_scores.shape[1], p=p))
    return np.array(pick)


def reward_function(gt_answers, generated_answers, num_samples, min_len):
    scorer = RougeScorer(['rougeL'], use_stemmer=True)
    gt_answers = np.repeat(gt_answers, num_samples).tolist()

    scores = []
    for gt, gen in zip(gt_answers, generated_answers):
        score = scorer.score(gt, gen)['rougeL'].recall

        if min_len == 0:
            scores.append(score)
            continue

        length = len(gen.strip().split(" "))
        if length < min_len:
            score += np.exp(-length)
            score = min(score, 1)

        scores.append(score)

    return np.array(scores).reshape(-1, num_samples)


def extract(input_ids, labels, tokenizer):
    input_ids = input_ids.clone()

    # isolate question (erase answer)
    answer_start = (labels != -100).int().argmax(1)
    max_answer_start = answer_start.max()
    for i in range(input_ids.shape[0]):
        input_ids[i, answer_start[i]:] = tokenizer.pad_token_id
    questions = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    # isolate ground truth answers
    cleaned_labels = [[token_id for token_id in seq if token_id != -100]
                      for seq in labels]
    gt_answers = tokenizer.batch_decode(cleaned_labels,
                                        skip_special_tokens=True)

    return questions, gt_answers


def generate_answers(hparams, model, tokenizer, questions, num_samples):
    inputs = encode_inference(hparams, tokenizer, questions)

    model.eval()

    input_ids = inputs.input_ids.repeat_interleave(num_samples, dim=0)
    attention_mask = inputs.attention_mask.repeat_interleave(
        num_samples, dim=0)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=hparams['max_length'],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=hparams['temperature'],
        top_p=hparams['top_p'],
        num_return_sequences=1,
    )

    model.train()

    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    if 'answer_tag' not in hparams:
        return generations

    # Remove the answer tag from the generation (relevant for phi).
    generations = [
        gen[len(hparams["answer_tag"]):]
        for gen in generations
    ]

    return generations
