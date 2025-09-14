from encoding import encode_inference
from torch.utils.data import DataLoader
from rouge_score.rouge_scorer import RougeScorer
from tqdm.auto import tqdm
import evaluate
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def generate(hparams, tokenizer, model, dataset):
    model = model.eval()

    result = []
    rouge = evaluate.load('rouge')
    scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    dataloader = DataLoader(dataset,
                            batch_size=hparams['batch_size_inference'],
                            shuffle=False)

    for batch in tqdm(dataloader):

        inputs = encode_inference(hparams, tokenizer, batch['question'])

        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=hparams['max_length'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1
        )

        # Decode only the generated tokens (exclude input tokens)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        triples = zip(batch['question'], batch['answer'], generated)
        for question, answer, generation in triples:

            if 'answer_tag' in hparams:
                # Remove the answer tag from the generation (relevant for phi).
                prefix = generation[:len(hparams["answer_tag"])]

                if prefix == hparams["answer_tag"]:
                    generation = generation[len(hparams["answer_tag"]):]

            rouge_score = rouge.compute(predictions=[generation],
                                        references=[answer],
                                        use_stemmer=True)
            rouge_scorer = scorer.score(answer, generation)

            result.append({
                "question": question,
                "answer": answer,
                "generation": generation,
                "eval": {
                    "rouge_score": rouge_score,
                    "rouge_score_recall": {
                        "rouge1": rouge_scorer['rouge1'].recall,
                        "rougeL": rouge_scorer['rougeL'].recall
                    },
                }
            })

    return result
