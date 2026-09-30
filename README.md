# AI-Generated News Text Detection: Generalisation to Unseen LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brucenguyen0302-code/ai-text-detection/blob/main/main.ipynb)

## Task

Given one news text, decide whether a human journalist wrote it or a large language model
generated it.

- **Input**: the first 200 words of an English news text, after formatting normalisation.
- **Output**: a score `z`, with `P(AI) = sigmoid(z)`, and a binary label when `P(AI)` reaches a
  threshold fixed on the validation set at a 1% false-positive rate.
- **Specified domain**: English news text of at least 200 words after cleaning. Shorter texts are
  outside what the system was trained and tested on.

## Research question

Does a supervised detector of AI-generated news text keep its performance on a generator it never
saw during training, and if it does not, does a contextual representation recover more of the loss
than a bag of words?

The evaluation is built around leave-one-generator-out folds: one generator is removed from
training and from validation, and the model is tested only against that generator plus human text.
The split groups texts by topic, so a model is never tested on a news event it saw in training.

## Main result

At a 1% false-positive budget with 50-word inputs, averaged over six held-out generators, a
fine-tuned DistilBERT reaches 0.9926 true-positive rate against 0.8898 for TF-IDF with logistic
regression. Bootstrap intervals over the test set do not overlap.

Both conditions matter and the notebook measures both. Relaxing the budget to 5% raises the two to
0.9994 and 0.9694, so the absolute gap narrows to three points while the transformer still removes
98% of the errors the baseline makes. At 200 words instead of 50 the baseline loses at most 0.0157
on an unseen generator to begin with, so there is little left to recover. The ordering holds
throughout; the size of the advantage does not.

Removing formatting artefacts first is what makes any of this measurable. With the raw markdown and
`Title:` prefixes left in, the linear baseline reaches 0.9997 from only 25 words of input; with them
normalised away it reaches 0.8223.

## Dataset

*A Comprehensive Dataset for Human vs. AI Generated Text Detection*, Roy et al., DeFactify 4.0
workshop on Multimodal Fact-Checking and Hate Speech Detection, 2025 (arXiv:2510.22874).

Human text is New York Times article content; the AI side comes from Gemma-2-9B, Mistral-7B,
Qwen-2-72B, LLaMA-8B, Yi-Large and GPT-4o, each responding to the same source article. This project
uses the train split published at `gsingh1-py/train` under CC BY 4.0, which is 7,321 rows and
reshapes to 51,181 individual texts.

The notebook downloads the data at runtime and nothing is stored in this repository. Part 2
documents several data quality problems that the dataset paper does not report, including four of
the six generator columns being filed under the wrong prompt.

## Repository

```
main.ipynb         complete implementation, runs in Colab
requirements.txt   versions used for the reported results
results/           every table and figure the notebook produces
```

## Running it

Open the Colab badge above and run all cells. Select a T4 GPU runtime: Part 5 fine-tunes nine
transformer models and is impractical on CPU from a cold cache.

Finished steps are cached, so a re-run skips completed work. If Google Drive is unavailable the
cache falls back to local storage and everything is recomputed, so the notebook runs end to end
without access to the author's Drive.
