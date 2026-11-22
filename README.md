# Sarcasm-Aware Hate Speech Detection with Rationale-Guided Attention

An experimental re-implementation and evaluation of **"Hate Speech Detection by Using Rationales for Judging Sarcasm"** (Mamun et al., *Applied Sciences* 2024, 14, 4898).

The repository studies whether human-marked hate and sarcasm rationales can help a transformer classifier identify hateful content expressed indirectly, and whether rationale-guided models produce more faithful explanations.

## Current best notebook snapshot

The notebook `HateSpeechDetection(currentBestClassificationMetrics).ipynb` contains a multi-task transformer experiment with four prediction heads:

1. Hate / normal / offensive classification
2. Word-level hate-rationale prediction
3. Sarcasm classification
4. Sarcasm-span tagging

The notebook combines model variants into an ensemble and reports the following held-out test snapshot:

| Metric | Reported result |
|---|---:|
| Test-set size | 300 posts |
| Accuracy | **76.33%** |
| Macro-F1 | **67.41%** |
| Macro AUROC (one-vs-rest) | **87.37%** |

Class-level F1 in that run:

| Class | Test support | F1 |
|---|---:|---:|
| Hate speech | 112 | 0.788 |
| Normal | 146 | 0.840 |
| Offensive | 42 | 0.395 |

The notebook also reports bootstrap 95% intervals of **0.713–0.810** for accuracy and **0.615–0.734** for macro-F1. Paired McNemar comparisons against the two listed baselines have p-values of 0.1337 and 0.3449; the notebook therefore does not establish a statistically significant improvement over those baselines at the 0.05 level.

These are results from one saved notebook run, not a guarantee of generalization. In particular, the offensive-class F1 is lower than the hate-speech and normal-class F1.

## Earlier rationale-supervision cross-validation experiment

An earlier experiment in this repository compared rationale-guided attention supervision over stratified cross-validation. Its reported findings were:

- Rationale-guided attention improved the reported rationale-alignment/faithfulness measures in that experiment.
- Macro-F1 changed from approximately 0.635 to 0.642, with no statistically clear improvement.
- Adding sarcasm rationales did not show a measurable additional macro-F1 gain in that run.
- The extra sarcasm-prediction head did not show a measurable gain in that configuration.

**Do not compare the earlier cross-validation figures directly with the current 300-post held-out snapshot.** They come from different notebooks, splits, training configurations, and evaluation procedures.

## Repository notebooks

- `HateSpeechDetection(currentBestClassificationMetrics).ipynb` — current ensemble and held-out classification/evaluation snapshot
- `RAGF_BERT_v5_layer_freezing_sweep (1).ipynb` — layer-freezing experiment
- `RAGF_BERT_v5_layers_(1)(gradual_unfreezing).ipynb` — gradual-unfreezing experiment
- `hate_sarcasm_improved_(1)claud.ipynb` — additional improved-model experiment
- `sarcasm-rationale.ipynb` — rationale-oriented experiment
- `notebook2ac38470f2.ipynb` — supplementary notebook

## Explainability and evaluation

The current-best notebook includes:

- **Plausibility:** token-F1, IOU-F1, and AUPRC against human-marked rationales
- **Faithfulness:** comprehensiveness and sufficiency
- **Explainers:** occlusion scores and the learned rationale head
- **Classification:** accuracy, macro-F1, class-wise precision/recall/F1, and AUROC
- **Subgroup analysis:** sarcastic and non-sarcastic test posts

The earlier notebook track uses its own cross-validation protocol and should be read separately.

## Dataset

The experiments combine the public HateXplain dataset with a manually annotated sarcasm file:

- HateXplain provides posts, class annotations, and word-level hate rationales.
- `sarcasm_manual_3199_2.csv` contains the manually annotated sarcasm labels and spans. **This file is not included in the repository** and must be supplied locally when running the notebook.

The README's earlier dataset notes report 3,199 manually annotated rows, including 448 sarcastic posts (216 toxic). The exact usable rows depend on the notebook's filtering and split logic.

## Run the current-best notebook

1. Open `HateSpeechDetection(currentBestClassificationMetrics).ipynb` in Colab or Kaggle.
2. Select a GPU runtime if available.
3. Run the setup cells and provide the required annotation CSV when prompted.
4. Review the configuration cell before a full run; training the ensemble can require substantial GPU memory and time.
5. Save classification, explainability, and prediction outputs from the notebook.

Typical libraries include PyTorch, Transformers, Accelerate, scikit-learn, pandas, and matplotlib. The notebook installs its runtime dependencies in its setup cells.

## Research limitations

- The manual sarcasm annotation file is not included, so another person cannot reproduce the experiment from this repository alone.
- The held-out test set contains only 300 posts; its class distribution is imbalanced.
- The offensive class has relatively low F1 in the current-best snapshot.
- Rationale-alignment scores measure agreement with the available annotations; they do not by themselves prove that an explanation is faithful or useful to people.
- Results across notebook versions use different splits and training configurations. Keep those experiment tracks separate when reporting or comparing metrics.

## Reference

Mamun et al., *Hate Speech Detection by Using Rationales for Judging Sarcasm*, Applied Sciences 2024, 14, 4898.

Mathew et al., *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*, AAAI 2021.
