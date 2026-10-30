# Sarcasm-Aware Hate Speech Detection with Rationale-Guided Attention

An improved re-implementation and critical evaluation of
**"Hate Speech Detection by Using Rationales for Judging Sarcasm"** (Mamun et al., *Applied Sciences* 2024, 14, 4898).

The paper teaches a BERT classifier to look at the words humans marked as hateful, and also at the words that make a post sarcastic, so that hate hidden behind sarcasm is easier to detect. This project re-implements that idea with a stronger model and training setup, tests it with proper cross-validation and statistics, and checks whether the paper's claimed gain from sarcasm rationales holds on our dataset.

## Main findings

| Question | Result |
|---|---|
| Does rationale-guided attention supervision improve explanations? | **Yes.** Attention token-F1 0.44 → 0.51, AUPRC 0.64 → 0.82, comprehensiveness 0.49 → 0.62 (p < 0.0001, 5 folds × 3 seeds) |
| Does it improve detection (macro-F1)? | No clear change (0.635 → 0.642, not significant) |
| Does adding sarcasm rationales help (the paper's claim)? | **Not on this dataset.** Macro-F1 0.642 → 0.642; explainability unchanged |
| Does the extra sarcasm-prediction head help? | No measurable gain |
| Did we reproduce the paper's +0.01 macro-F1 gain? | No |

Our explainability scores are higher than the paper's, mainly because we use a much stronger attention loss (λ = 1.0 vs the paper's 0.001) and a different evaluation setup. This is **not** a like-for-like win over the paper. Our detection scores are lower than the paper's (macro-F1 about 0.64 vs 0.70), partly because our annotation file has fewer sarcastic posts and we test on 5 folds instead of one 300-post split.

## What the notebook does

1. Loads HateXplain (public) and joins it with the sarcasm annotation file by `post_id`.
2. Builds a ground-truth attention target for each toxic post: `r = h + 0.5·s`
   (hate-only word = 1, sarcasm-only = 0.5, both = 1.5), passed through a temperature softmax. Normal posts get a uniform target.
3. Trains RoBERTa with three loss terms:
   - class-weighted cross-entropy with label smoothing,
   - KL-divergence between the `[CLS]` attention (last layer, 6 heads) and the target (weight λ),
   - optional auxiliary sarcasm-prediction loss (weight 0.3).
4. Compares four modes with 5-fold stratified cross-validation:

| Mode | Hate rationale | Sarcasm rationale | Sarcasm head | Attention loss |
|---|---|---|---|---|
| `no_att_supervision` | – | – | – | off |
| `baseline_hate_rat` | yes | – | – | on |
| `proposed_sarc_rat` (paper's idea) | yes | yes | – | on |
| `proposed_sarc_rat+aux` (our addition) | yes | yes | yes | on |

5. Reports classification metrics, hidden-hate metrics and explainability metrics.

## Changes compared with the paper

| Item | Paper | This project |
|---|---|---|
| Backbone | BERT | RoBERTa-base |
| Attention-loss weight λ | 0.001 | 1.0 |
| Attention-loss form | cross-entropy | KL divergence |
| Batch size | 4 | 16 |
| Learning rate | 2e-5 | 2e-5 with 10% warmup and linear decay |
| Regularisation | dropout 0.1 | dropout 0.1, weight decay 0.01, gradient clipping 1.0 |
| Class imbalance | not handled | class weights + label smoothing 0.05 |
| Model selection | not described | best epoch by validation macro-F1 |
| Sarcasm head | none | auxiliary head, weight 0.3 |
| Evaluation | one split, 300 test posts | stratified 5-fold CV, several seeds, paired tests |

Kept the same as the paper: the `r = h + 0.5·s` rule, supervision on the last layer's `[CLS]` attention (6 heads), and top-5 tokens for explanations.

## Metrics

**Classification:** accuracy, macro-F1, offensive-class F1, AUROC.

**Hidden hate** (the 216 toxic *and* sarcastic posts): `hidden_acc`, `hidden_toxic_recall`.

**Explainability** (the model's top-5 words compared with the human-marked words):

| Metric | Better when | Meaning |
|---|---|---|
| Token-F1 | higher | Overlap between the model's top-5 words and the human rationale |
| AUPRC | higher | How well the model ranks human-marked words above the rest |
| Comprehensiveness | higher | Drop in prediction when the top-5 words are removed |
| Sufficiency | lower | Drop in prediction when only the top-5 words are kept |

Attention-based scores use the model's own attention. The full-experiment notebook also computes LIME-based scores, as in the paper. IOU-F1, which the paper also reports, is not implemented.

## Dataset

- **HateXplain** (public): posts, annotator labels (normal / hate speech / offensive) and word-level hate rationales. The notebook downloads `dataset.json` automatically from the HateXplain GitHub repository.
- **`sarcasm_manual_3199_2.csv`** (your annotation file, **not included in this repository**): 3,199 posts with `post_id`, `text`, `label`, `sarcasm_label`, `sarcasm_rationale`.
  - 448 sarcastic posts, 216 of them toxic.
  - After dropping the 10 "undecided" posts, 3,189 posts remain (1,552 normal, 1,193 hate speech, 444 offensive).
  - 441 of 448 sarcasm spans match HateXplain's word tokens exactly; the other 7 use a fuzzy match.

## How to run

1. Open the notebook in Google Colab and select a GPU (Runtime → Change runtime type → T4 GPU).
2. Run the setup cells. When asked, upload `sarcasm_manual_3199_2.csv`.
3. The notebook starts with `quick=True`, a smoke test (first fold, 1 epoch, a few minutes). Set `quick=False` in the config cell for the full run (5 folds × 4 modes, roughly 45–90 minutes on a T4).
4. For more stable numbers, set `seeds=[42, 43, 44]`.

Requirements: `transformers`, `accelerate`, `scikit-learn`, `pandas`, `matplotlib`, `torch` (installed by the first cell).

Outputs: `results_per_fold.csv`, `results_summary.csv`, `results.png`.

Settings you can change in the config cell:

| Setting | Meaning |
|---|---|
| `backbone` | `roberta-base`; try `GroNLP/hateBERT` or `bert-base-uncased` |
| `lam` | attention-loss weight (paper: 0.001) |
| `tau` | sharpness of the target attention (smaller = sharper) |
| `aux_w` | weight of the sarcasm head |
| `cv_folds` | 5 for cross-validation, 1 for a single paper-like split |

## Limitations

- Only 216 toxic sarcastic posts exist, so sarcasm effects are hard to detect, and results on that subset are noisy.
- Token-F1 and AUPRC partly measure how well training worked, since the model is trained to match the same human rationales it is scored against.
- Comprehensiveness and sufficiency depend on post length, because five words are removed from every post regardless of its length.
- Several settings changed at once relative to the paper, so only the "attention loss on vs off" comparison isolates a single factor.
- Many metrics were tested without a multiple-comparison correction, so one isolated "significant" result can be chance.

## Reference

Mamun et al. *Hate Speech Detection by Using Rationales for Judging Sarcasm.* Applied Sciences 2024, 14, 4898.

HateXplain: Mathew et al. *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.* AAAI 2021.

## License

Add a license of your choice (for example MIT) before publishing. Check the HateXplain license and the terms of your annotation data before sharing any data files.
