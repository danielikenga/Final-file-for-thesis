# Research Findings Log

## Dissertation Project

**Topic:** Retrieval-Augmented Misinformation Detection for Low-Resource Nigerian Languages

**Languages:** Hausa, Igbo, Yoruba

**Evaluation Test Set:** 492 examples

**Primary Labels:**
- supports
- refutes
- nei

---

# 1. Purpose of This Findings Log

This document records quantitative findings, qualitative observations, interpretation hypotheses, methodological caveats, and report-ready insights produced during the experimental evaluation.

The purpose is to preserve the analytical reasoning behind the dissertation results rather than relying only on final metric tables.

Interpretations recorded here should later be validated against the relevant experiment outputs before inclusion in the final dissertation.

---

# 2. Current Master Experimental Results

## 2.1 Encoder and LLM Results

| Experiment | Accuracy | Macro-F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| XLM-R Custom Split | 0.459350 | 0.442156 | 0.462285 | 0.460203 |
| AfriBERTa Custom Split | 0.497967 | 0.492296 | 0.491666 | 0.493279 |
| XLM-R Gold Evidence | 0.630081 | 0.589212 | 0.608706 | 0.617331 |
| AfriBERTa Gold Evidence | 0.571138 | 0.568683 | 0.576857 | 0.568618 |
| Qwen 1.5B Claim Only | 0.390244 | 0.380737 | 0.393998 | 0.389255 |
| Qwen 1.5B Gold Evidence | 0.459350 | 0.400397 | 0.525618 | 0.452344 |
| Qwen 1.5B Gold + Reasoning | 0.436992 | 0.425215 | 0.460196 | 0.433863 |
| Qwen 1.5B BM25 Evidence | 0.394309 | 0.318210 | 0.470052 | 0.392613 |
| Qwen 1.5B Adversarial Evidence | 0.333333 | 0.315591 | 0.349061 | 0.326094 |
| Qwen 14B Gold Evidence | 0.640244 | 0.600194 | 0.724721 | 0.624233 |
| Qwen 14B Few-shot (3) | 0.640244 | 0.595699 | 0.704338 | 0.625340 |
| Qwen 14B Few-shot (6) | 0.644309 | 0.592833 | 0.700043 | 0.629632 |
| Qwen 14B Few-shot (9) | 0.632114 | 0.582532 | 0.698087 | 0.618138 |
| Qwen 14B + Google Translate | 0.735772 | 0.733508 | 0.800859 | 0.729060 |
| Qwen 14B + Translate + Generic Examples | 0.621951 | 0.593794 | 0.795600 | 0.606069 |

---

# 3. Initial Cross-Experiment Findings

## 3.1 Comparable encoder baselines on the 492-example evaluation set

The primary encoder comparison uses the custom-split XLM-R and AfriBERTa experiments evaluated on the common 492-example test set.

### XLM-R Custom Split

- Accuracy = 0.459350
- Macro-F1 = 0.442156

### AfriBERTa Custom Split

- Accuracy = 0.497967
- Macro-F1 = 0.492296

AfriBERTa exceeds XLM-R by approximately 3.86 percentage points in accuracy and 5.01 percentage points in Macro-F1 on this evaluation set.

### Interpretation

Within the controlled custom-split setting, AfriBERTa performs better than XLM-R on both accuracy and Macro-F1. This is consistent with the possibility that African-language-focused pretraining is beneficial for Hausa, Igbo, and Yoruba classification. However, the experiment alone does not establish that pretraining focus is the causal mechanism, and architectural or optimisation differences should not be ruled out.

---

## 3.2 Gold-evidence encoder performance

On the common 492-example evaluation set, the gold-evidence encoder conditions achieve higher scores than the custom-split claim-input encoder baselines. These differences are reported as cross-condition performance differences; they should not be interpreted as a strictly isolated causal effect of evidence unless the remaining training and preprocessing conditions are confirmed to be equivalent.

### XLM-R

- Custom-split baseline accuracy = 0.459350
- Gold-evidence accuracy = 0.630081
- Absolute accuracy difference = +0.170731
- Custom-split baseline Macro-F1 = 0.442156
- Gold-evidence Macro-F1 = 0.589212
- Absolute Macro-F1 difference = +0.147056

### AfriBERTa

- Custom-split baseline accuracy = 0.497967
- Gold-evidence accuracy = 0.571138
- Absolute accuracy difference = +0.073171
- Custom-split baseline Macro-F1 = 0.492296
- Gold-evidence Macro-F1 = 0.568683
- Absolute Macro-F1 difference = +0.076387

### Interpretation

The gold-evidence conditions are associated with higher performance for both encoder families on the shared evaluation set. The increase is larger for XLM-R than for AfriBERTa in these experiments. This supports the importance of evidence availability for the verification task, while avoiding a stronger causal claim than the experimental design warrants.

---

## 3.3 Methodological comparability note

Earlier claim-only encoder outputs containing 1,950 predictions are retained as legacy artefacts but are excluded from the primary master comparison because they were not evaluated on the common 492-example custom test set.

A claim-text audit found that only 384 of the 492 custom-test claims were represented in each 1,950-row legacy prediction file, with 108 custom-test claims absent. Direct score comparisons between those legacy outputs and the 492-example experiments would therefore confound model performance with evaluation-set composition.

For this reason:

- XLM-R Custom Split is used as the primary comparable XLM-R baseline.
- AfriBERTa Custom Split is used as the primary comparable AfriBERTa baseline.
- The 1,950-row legacy claim-only encoder scores are not used for direct performance-difference claims in the main analysis.
- Comparisons in the master results should be restricted to experiments aligned to the common 492-example evaluation set, with translated-input experiments interpreted according to their intentional preprocessing condition.

---

# 4. Model Scale Findings

## 4.1 Qwen 14B substantially outperforms Qwen 1.5B with gold evidence

Qwen 1.5B Gold Evidence:

- Accuracy = 0.459350
- Macro-F1 = 0.400397

Qwen 14B Gold Evidence:

- Accuracy = 0.640244
- Macro-F1 = 0.600194

Absolute improvement:

- Accuracy: +0.180894
- Macro-F1: +0.199797

### Interpretation

Under the evaluated gold-evidence condition, Qwen 14B substantially outperforms Qwen 1.5B on both accuracy and Macro-F1.

This result is consistent with the larger model having stronger capability for evidence-conditioned verification, potentially including:
- multilingual semantic processing,
- claim-evidence alignment,
- contradiction recognition,
- instruction following.

However, the comparison does not isolate model scale as the sole causal factor, because checkpoint-specific representation, optimisation, and instruction-following differences may also contribute. Moreover, Qwen 14B still achieves only 64.02% accuracy on original-language gold evidence.

---

# 5. Few-Shot Prompting Findings

## 5.1 Few-shot prompting does not reliably improve Qwen 14B

Qwen 14B Gold Evidence baseline:

- Accuracy = 0.640244
- Macro-F1 = 0.600194

3-shot:

- Accuracy = 0.640244
- Macro-F1 = 0.595699

6-shot:

- Accuracy = 0.644309
- Macro-F1 = 0.592833

9-shot:

- Accuracy = 0.632114
- Macro-F1 = 0.582532

### Key observation

The highest few-shot accuracy is obtained with 6 examples:

- 0.644309

However, this is only a very small improvement over zero-shot gold evidence:

- 0.640244

At the same time, Macro-F1 declines.

The 9-shot configuration performs worse than the baseline.

### Interpretation

Performance does not improve monotonically as the number of demonstrations increases.

Potential explanations include:
- prompt interference,
- demonstration mismatch,
- increased context complexity,
- label bias,
- poor transfer from examples to low-resource claim-evidence relations.

These explanations remain hypotheses because the experiment varies demonstration count without separately isolating the underlying mechanism.

### Report-ready insight

More demonstrations are not automatically beneficial. In this experiment, few-shot performance was non-monotonic: the 6-shot condition produced only a small accuracy increase over the zero-shot gold-evidence baseline while reducing Macro-F1, and the 9-shot condition underperformed the baseline on both metrics.

---

# 6. Generic Prompt Examples After Translation

Translation-only Qwen 14B:

- Accuracy = 0.735772
- Macro-F1 = 0.733508

Translation + Generic Prompt Examples:

- Accuracy = 0.621951
- Macro-F1 = 0.593794

Absolute change:

- Accuracy = -0.113821
- Macro-F1 = -0.139714

### Interpretation

Under this translated-input setup, adding generic prompt examples substantially reduces performance.

This is an important negative result and should be reported rather than hidden.

Possible explanations include mismatch between:
- simplified demonstration patterns,
- real Afrifact examples,
- culturally specific entities,
- noisy evidence,
- complex multilingual relations.

These explanations remain hypotheses because demonstration alignment was not independently manipulated or measured.

### Report-ready insight

In this experiment, adding generic prompt demonstrations to the translated-input condition was associated with substantially lower performance than translation alone.

This result complements the earlier few-shot experiments and suggests that the usefulness of demonstrations may depend on their quality and task alignment rather than simply their number.

---

# 7. BM25 Retrieval Findings

## 7.1 Retrieval metrics

BM25 retrieval evaluation over 492 test queries produced:

- Recall@1 = 0.197154
- Recall@3 = 0.481707
- Recall@5 = 0.573171
- Recall@10 = 0.628049

### Interpretation

The correct evidence is retrieved at rank 1 for only approximately 19.72% of queries.

However, recall rises substantially as the retrieval depth increases:

- approximately 48.17% by top 3,
- approximately 57.32% by top 5,
- approximately 62.80% by top 10.

### Key retrieval insight

BM25 retrieves the correct evidence within the top 10 for approximately 62.80% of queries, but places it at rank 1 for only approximately 19.72%.

This observed gap suggests a potential distinction between:
- candidate generation,
- candidate ranking.

### Report-ready insight

The retrieval bottleneck is not simply total failure to retrieve relevant evidence. A substantial proportion of relevant evidence appears within deeper candidate sets, suggesting that reranking may be a promising future direction.

---

# 8. End-to-End BM25 Evidence Verification

Qwen 1.5B Gold Evidence:

- Accuracy = 0.459350
- Macro-F1 = 0.400397

Qwen 1.5B BM25 Evidence:

- Accuracy = 0.394309
- Macro-F1 = 0.318210

### Interpretation

Under the evaluated Qwen 1.5B conditions, performance is lower with BM25-retrieved evidence than with oracle gold evidence.

This demonstrates an observed oracle-to-retrieved evidence performance gap.

The result is consistent with retrieval quality being an important constraint on downstream fact verification, although the comparison should not be interpreted as a strictly isolated causal estimate unless all remaining pipeline conditions are confirmed equivalent.

### Important dissertation interpretation

Gold-evidence experiments measure verification performance under oracle evidence access.

BM25 experiments measure performance under a more realistic retrieval-augmented condition.

The observed performance difference estimates the oracle-to-retrieved condition gap in this experimental setup; it should not automatically be attributed entirely to retrieval error without confirming equivalence of the remaining conditions.

---

# 9. Adversarial Evidence Findings

Qwen 1.5B Gold Evidence:

- Accuracy = 0.459350
- Macro-F1 = 0.400397

Qwen 1.5B Adversarial Evidence:

- Accuracy = 0.333333
- Macro-F1 = 0.315591

### Interpretation

Performance degrades under adversarial evidence.

Accuracy reaches approximately one-third, which is especially notable in a three-class task.

This indicates sensitivity to misleading or mismatched evidence.

### Report-ready insight

In this experiment, the adversarial-evidence condition performs substantially worse than the gold-evidence condition. This indicates that misleading or mismatched context can degrade model decisions under the evaluated setup.

The result supports the importance of both:
- retrieval relevance,
- robustness to misleading context

in retrieval-augmented fact-checking systems.

---

# 10. Translation Experiment

## 10.1 Overall result

Qwen 14B Original Gold Evidence:

- Accuracy = 0.640244
- Macro-F1 = 0.600194

Qwen 14B Translated Gold Evidence:

- Accuracy = 0.735772
- Macro-F1 = 0.733508

Absolute accuracy improvement:

- +0.095528
- approximately +9.55 percentage points

Absolute Macro-F1 improvement:

- +0.133314
- approximately +13.33 percentage points

### Interpretation

The English-translated input condition produces the strongest overall result among the current experiments.

However, subsequent analysis shows that this gain is highly non-uniform across languages and labels.

The result should therefore not be interpreted as evidence that translation is universally beneficial, nor as direct evidence of improved reasoning in the original Hausa, Igbo, and Yoruba inputs.

---

# 11. Translation Effect by Language

| Language | Examples | Original Accuracy | Translated Accuracy | Absolute Change |
|---|---:|---:|---:|---:|
| Igbo | 163 | 0.607362 | 0.742331 | +0.134969 |
| Hausa | 166 | 0.650602 | 0.759036 | +0.108434 |
| Yoruba | 163 | 0.662577 | 0.705521 | +0.042945 |

## 11.1 Igbo

Improvement:

- +13.50 percentage points

This is the largest overall language-level gain.

## 11.2 Hausa

Improvement:

- +10.84 percentage points

## 11.3 Yoruba

Improvement:

- +4.29 percentage points

This is positive overall but substantially smaller than the improvements for Igbo and Hausa.

### Interpretation

Translation benefits are language-dependent.

The later language-label interaction analysis shows that Yoruba's smaller aggregate improvement masks a severe decline for supported claims.

---

# 12. Translation Effect by Gold Label

| Gold Label | Examples | Original Accuracy | Translated Accuracy | Absolute Change |
|---|---:|---:|---:|---:|
| supports | 166 | 0.668675 | 0.602410 | -0.066265 |
| refutes | 152 | 0.250000 | 0.625000 | +0.375000 |
| nei | 174 | 0.954023 | 0.959770 | +0.005747 |

## 12.1 Major result: contradiction detection improves dramatically

Refutes accuracy increases from:

- 25.00%
- to 62.50%

Absolute improvement:

- +37.50 percentage points

The later transition decomposition shows that improved performance on refutes examples is the dominant source of the net translation-associated gain.

## 12.2 Supports performance declines

Supports accuracy decreases from:

- 66.87%
- to 60.24%

Absolute decline:

- -6.63 percentage points

## 12.3 NEI remains extremely strong

NEI accuracy changes only slightly:

- 95.40%
- to 95.98%

### Central interpretation

Translation is associated with asymmetric changes across fact-verification labels.

It substantially improves accuracy on refutes examples while reducing accuracy on supports examples.

This label-specific asymmetry is a central empirical finding of the dissertation and is consistent with improved contradiction recognition alongside weaker entailment recognition under the translated-input condition.

---

# 13. Translation Transition Analysis

Across 492 examples:

| Transition | Count | Percentage |
|---|---:|---:|
| correct_to_correct | 268 | 54.47% |
| wrong_to_correct | 94 | 19.11% |
| wrong_to_wrong | 83 | 16.87% |
| correct_to_wrong | 47 | 9.55% |

## Interpretation

Translation repairs:

- 94 previously incorrect examples

Translation damages:

- 47 previously correct examples

Net improvement:

- +47 correct predictions

This exactly corresponds to the overall accuracy increase:

- 47 / 492 = approximately 9.55 percentage points

### Report-ready insight

Translation is net-positive at the aggregate prediction level but is also associated with regressions. It repairs twice as many decisions as it damages, yet the 47 correct-to-wrong transitions demonstrate meaningful prediction-regression risk under the translated-input condition.

---

# 14. Translation Transitions by Language

## Hausa

- correct_to_correct = 100
- correct_to_wrong = 8
- wrong_to_correct = 26
- wrong_to_wrong = 32

Net repaired decisions:

- 26 - 8 = +18

## Igbo

- correct_to_correct = 81
- correct_to_wrong = 18
- wrong_to_correct = 40
- wrong_to_wrong = 24

Net repaired decisions:

- 40 - 18 = +22

## Yoruba

- correct_to_correct = 87
- correct_to_wrong = 21
- wrong_to_correct = 28
- wrong_to_wrong = 27

Net repaired decisions:

- 28 - 21 = +7

### Interpretation

Igbo receives the largest net benefit.

Yoruba has the weakest net gain because translation repairs 28 errors but introduces 21 new errors.

This explains why Yoruba's aggregate improvement is substantially smaller.

---

# 15. Translation Transitions by Label

## NEI

- correct_to_correct = 162
- correct_to_wrong = 4
- wrong_to_correct = 5
- wrong_to_wrong = 3

Net change:

- +1

## Refutes

- correct_to_correct = 31
- correct_to_wrong = 7
- wrong_to_correct = 64
- wrong_to_wrong = 50

Net change:

- +57

## Supports

- correct_to_correct = 75
- correct_to_wrong = 36
- wrong_to_correct = 25
- wrong_to_wrong = 30

Net change:

- -11

### Central finding

Almost the entire net translation improvement is driven by refutes examples.

Translation creates:

- +57 net correct refutes decisions

but:

- -11 net supports decisions

and:

- +1 net NEI decision

Total:

- +47 net correct predictions

### Report-ready interpretation

The overall translation-associated gain should not be interpreted as uniform multilingual improvement. The transition decomposition shows that the net improvement is driven primarily by improved performance on refutes examples.

---

# 16. Language × Label Interaction

| Language | Label | Examples | Original Accuracy | Translated Accuracy | Absolute Change |
|---|---|---:|---:|---:|---:|
| Hausa | NEI | 67 | 0.970149 | 0.940299 | -0.029851 |
| Hausa | Refutes | 43 | 0.255814 | 0.627907 | +0.372093 |
| Hausa | Supports | 56 | 0.571429 | 0.642857 | +0.071429 |
| Igbo | NEI | 53 | 0.962264 | 0.962264 | 0.000000 |
| Igbo | Refutes | 55 | 0.290909 | 0.690909 | +0.400000 |
| Igbo | Supports | 55 | 0.581818 | 0.581818 | 0.000000 |
| Yoruba | NEI | 54 | 0.925926 | 0.981481 | +0.055556 |
| Yoruba | Refutes | 54 | 0.203704 | 0.555556 | +0.351852 |
| Yoruba | Supports | 55 | 0.854545 | 0.581818 | -0.272727 |

## Major interaction finding

Translation improves refutes accuracy for every language:

- Hausa: +37.21 percentage points
- Igbo: +40.00 percentage points
- Yoruba: +35.19 percentage points

This is a descriptively consistent cross-language pattern in the evaluated sample, although no claim of statistical significance is made here.

## Major anomaly

Yoruba supports accuracy decreases from:

- 85.45%
- to 58.18%

Absolute decline:

- -27.27 percentage points

### Interpretation

The weak overall Yoruba translation gain is not because translation fails uniformly on Yoruba.

Instead:
- Yoruba refutes accuracy improves strongly,
- Yoruba NEI accuracy improves,
- Yoruba supports accuracy declines substantially.

This interaction is substantially more informative than aggregate language accuracy alone.

---

# 17. Confusion Matrix Findings

## 17.1 Original-language Qwen 14B

| Gold Label | Pred Supports | Pred Refutes | Pred NEI |
|---|---:|---:|---:|
| Supports | 111 | 2 | 53 |
| Refutes | 27 | 38 | 87 |
| NEI | 3 | 5 | 166 |

### Key observation

The major original-language weakness is refutes detection.

Among 152 refutes examples:

- only 38 are correctly classified as refutes,
- 87 are incorrectly classified as NEI,
- 27 are incorrectly classified as supports.

The dominant error is therefore:

- refutes → NEI

## 17.2 Translated Qwen 14B

| Gold Label | Pred Supports | Pred Refutes | Pred NEI |
|---|---:|---:|---:|
| Supports | 100 | 13 | 53 |
| Refutes | 3 | 95 | 54 |
| NEI | 2 | 5 | 167 |

### Key observation

After translation:

- correct refutes predictions increase from 38 to 95,
- refutes → supports errors fall from 27 to 3,
- refutes → NEI errors fall from 87 to 54.

However:

- supports → refutes errors increase from 2 to 13.

### Interpretation

The translated-input condition substantially increases correct predictions for refutes examples, while also increasing supports-to-refutes errors from 2 to 13. This pattern is consistent with improved contradiction recognition accompanied by a greater tendency to assign some supported claims to the refutes class.

---

# 18. Qualitative Translation Case Analysis

## 18.1 Method

Representative cases were selected systematically rather than manually cherry-picked.

Target groups:

1. Refutes examples changing from wrong to correct.
2. Yoruba supports examples changing from correct to wrong.

Representative examples were selected using fixed text-length quantiles.

This selection strategy provides examples across a range of input lengths and reduces subjective case selection.

The cases should be described as illustrative error patterns rather than proof that every transition has the same cause.

---

# 19. Translation Repair Mechanisms

## 19.1 Ordinal contradiction exposure

### Example ID

`afrifact_data_culture_igbo_013`

### Transition

- Original prediction: NEI
- Translated prediction: refutes

### Claim relation

Claim:

- Bianca Ojukwu was the second female head of the Nigerian Stock Exchange.

Evidence:

- She was the first woman to hold the position.

### Interpretation

In this illustrative case, the translated input presents an explicit ordinal contrast:

- second
- versus first

### Hypothesised mechanism

A plausible mechanism is that the translated wording makes the ordinal contrast more accessible to the model, contributing to the corrected refutes prediction.

---

## 19.2 Event-role and predicate contradiction exposure

### Example ID

`afrifact_data_culture_hausa_169`

### Transition

- Original prediction: supports
- Translated prediction: refutes

### Claim relation

Claim:

- Governor Shema received an honour award from EFCC.

Evidence:

- EFCC prosecuted Governor Shema over alleged corruption involving approximately ₦11 billion.

### Interpretation

In this illustrative case, the translated text presents a clearer contrast between:

- honoured by institution
- versus prosecuted by institution

### Hypothesised mechanism

A plausible mechanism is improved model access to event predicates and institutional role relations in the translated wording.

---

## 19.3 Entity mismatch exposure

### Example ID

`afrifact_data_culture_yoruba_149`

### Transition

- Original prediction: NEI
- Translated prediction: refutes

### Claim relation

Claim attributes a promise to:

- Dauda / David

Evidence concerns:

- Ibrahim

### Interpretation

In this illustrative case, the translated wording may make the entity mismatch easier for the model to detect.

### Hypothesised mechanism

A plausible mechanism is improved entity-role discrimination under the translated wording.

---

## 19.4 Explicit detention contradiction

### Example ID

`afrifact_data_culture_yoruba_200`

### Transition

- Original prediction: NEI
- Translated prediction: refutes

### Claim relation

Claim states that no EndSARS protesters were arrested.

Evidence discusses detainees and individuals in police custody.

### Interpretation

In this illustrative case, the translated wording presents lexical relations around:

- arrested,
- detainees,
- police custody.

### Hypothesised mechanism

A plausible mechanism is improved recognition of semantically related detention concepts under the translated wording.

### Caveat

The translated numerical wording is visibly noisy and should not be treated as a clean translation example.

---

# 20. Translation Damage Mechanisms

## 20.1 Numerical and ranking corruption

### Example ID

`afrifact_data_culture_yoruba_624`

### Transition

- Original prediction: supports
- Translated prediction: NEI

### Original relation

The claim describes Ondo State as the 19th most populous state.

The original evidence supports the ranking relation.

### Translation problem

The translated evidence states:

- 11th largest state

instead of preserving the original population ranking.

### Interpretation

Translation changes:
- the numerical rank,
- and potentially the measured property.

### Mechanism

Numerical/ranking semantic corruption.

### Significance

In this illustrative case, qualitative inspection indicates that the translated text alters factual content rather than merely surface form.

---

## 20.2 Culturally specific lexical mistranslation

### Example ID

`afrifact_data_culture_yoruba_555`

### Transition

- Original prediction: supports
- Translated prediction: NEI

### Original concept

- gaàrí

### Translation problem

The translated claim renders gaàrí as:

- sugar

This changes the entity from a culturally specific cassava food product into an unrelated English concept.

### Mechanism

Culturally grounded lexical ambiguity and mistranslation.

### Significance

This case highlights a potential risk for low-resource African-language NLP: culturally specific vocabulary may be mistranslated into unrelated or overly generic English lexical categories. The broader prevalence of this problem is not established by a single case.

---

## 20.3 Predicate-level proposition corruption

### Example ID

`afrifact_data_culture_yoruba_358`

### Transition

- Original prediction: supports
- Translated prediction: refutes

### Original meaning

The claim concerns global breastfeeding and the number of lives that could be saved.

### Translation problem

The translated claim states:

- "if the entire world community donates a child"

This corrupts the central predicate.

### Mechanism

Proposition-level predicate mistranslation.

### Significance

In this illustrative case, the translated wording appears to change the proposition being presented to the model.

The associated supports-to-refutes transition is consistent with translation-related prediction instability, although the single case does not establish the prevalence of this mechanism.

---

## 20.4 Domain-specific semantic distortion

### Example ID

`afrifact_data_culture_yoruba_168`

### Transition

- Original prediction: supports
- Translated prediction: NEI

### Translation problem

A football positional description becomes:

- "back-handed footballer"

### Mechanism

Domain-specific terminology is translated into semantically malformed English.

### Caveat

The provided evidence excerpt may itself be incomplete for the full claim.

Therefore, this case should be treated as suggestive rather than definitive proof that translation alone caused the failure.

---

## 20.5 Possible translation-amplified claim mismatch

### Example ID

`afrifact_data_culture_yoruba_528`

### Transition

- Original prediction: supports
- Translated prediction: NEI

### Observation

The translated claim states that Wizkid is:

- "one of the best Afro Beat musicians"

The translated evidence discusses:
- Afro Beats,
- Wizkid,
- his early musical career.

However, the evidence does not clearly establish the stronger evaluative phrase:

- "one of the best"

### Interpretation

Translation may strengthen or alter the proposition in a way that reduces direct entailment.

### Caveat

The original example may also contain annotation noise or imperfect evidence alignment.

This case should therefore be described as a possible translation-amplified mismatch rather than definitive translation corruption.

---

# 21. Integrated Translation Interpretation

## Central finding

Translation is not uniformly beneficial in the evaluated setup.

At the aggregate level, the English-translated input condition improves overall Qwen 14B performance, but the effect is strongly heterogeneous across labels and languages. The transition analysis shows that the net gain is driven primarily by improved performance on refutes examples.

### Observed benefits and plausible repair patterns

Quantitatively, the translated-input condition substantially improves refutes-class accuracy. Qualitative inspection of selected repaired cases suggests plausible patterns involving:
- clearer ordinal contrasts,
- clearer entity mismatches,
- clearer event-role contrasts,
- improved accessibility of semantically related contradiction cues.

These qualitative patterns are illustrative and should not be interpreted as established mechanisms for all repaired predictions.

### Observed risks and illustrative regression patterns

The translated-input condition also produces 47 correct-to-wrong transitions. Qualitative inspection of selected regression cases identifies examples consistent with:
- culturally specific lexical mistranslation,
- numerical corruption,
- ranking corruption,
- predicate corruption,
- domain-specific terminology distortion,
- proposition strengthening or semantic drift.

Because translation quality was not manually annotated across all 492 examples, these patterns should be treated as plausible qualitative explanations rather than population-level causal findings.

### Strong report-ready statement

The English-translated input condition substantially improved overall Qwen 14B verification performance, but the gain was highly asymmetric. The net improvement was driven primarily by refutes examples, whose accuracy increased from 25.0% to 62.5%, with positive changes observed in Hausa, Igbo, and Yoruba. Conversely, supports accuracy declined overall, with Yoruba supports decreasing by 27.27 percentage points. Qualitative inspection of systematically selected illustrative cases suggests that some repaired predictions are consistent with clearer ordinal, entity, and event-role contrasts, while selected regressions are consistent with culturally specific lexical mistranslation, altered numerical semantics, and proposition-level predicate corruption. These qualitative patterns are plausible mechanisms rather than population-level causal explanations.

---

# 22. Methodological Caveats for Translation Analysis

The following caveats must be retained in the final dissertation.

## 22.1 Translation cases are illustrative

Representative cases demonstrate plausible mechanisms but do not prove that all transitions share the same cause.

## 22.2 Translation quality was not manually annotated for all 492 examples

Therefore, causal claims about translation errors should remain appropriately qualified.

Recommended language:

- "qualitative inspection suggests"
- "illustrative cases indicate"
- "errors were consistent with"
- "a plausible mechanism is"

Avoid:

- "translation caused every error"
- "all Yoruba failures resulted from mistranslation"

## 22.3 Dataset noise may interact with translation

Some examples may contain:
- imperfect evidence spans,
- annotation noise,
- malformed source text,
- incomplete context.

Therefore, translation and dataset quality may jointly influence outcomes.

## 22.4 Free translation tooling

The translation experiment uses a free translation-library route rather than a paid enterprise translation API.

The exact translation implementation, library, access method, and reproducibility limitations must be documented accurately in the methodology chapter.

---

# 23. Current Strongest Dissertation Findings

At the current stage, the strongest findings are:

1. Gold evidence dramatically improves encoder verification, particularly for XLM-R.

2. Larger model scale substantially improves evidence-conditioned verification.

3. Few-shot prompting does not produce monotonic gains and can reduce Macro-F1.

4. Generic prompt demonstrations substantially damage translated Qwen 14B performance.

5. BM25 retrieval recall rises strongly with retrieval depth but top-1 recall remains low.

6. Retrieved evidence underperforms oracle gold evidence, demonstrating an oracle-to-retrieved evidence gap.

7. Adversarial evidence reduces performance to approximately one-third accuracy.

8. Translation produces the strongest current overall result.

9. Translation improvement is primarily a contradiction-detection effect rather than uniform multilingual improvement.

10. Translation improves refutes accuracy consistently across Hausa, Igbo, and Yoruba.

11. Yoruba supported claims experience a severe translation-related degradation.

12. Qualitative analysis identifies plausible semantic repair and corruption mechanisms.

---

# 24. Future Analysis Queue

The following analyses remain to be completed.

## 24.1 Cross-experiment error analysis

Planned:
- encoder error analysis,
- Qwen 1.5B error analysis,
- Qwen 14B error analysis,
- few-shot error analysis,
- BM25 pipeline error analysis,
- adversarial evidence error analysis,
- generic prompt example error analysis.

## 24.2 Retrieval analysis

Planned:
- retrieval success versus downstream correctness,
- top-k retrieval depth analysis,
- cases where correct evidence is retrieved but verification fails,
- cases where retrieval fails but prediction is correct,
- potential language differences in retrieval.

## 24.3 Model comparison analysis

Planned:
- agreement/disagreement across models,
- examples solved only by larger models,
- examples solved by encoders but not LLMs,
- examples consistently difficult across architectures.

## 24.4 Statistical testing

Consider:
- paired significance testing for original versus translated predictions,
- bootstrap confidence intervals,
- McNemar's test for paired classification outcomes.

These should be implemented where methodologically appropriate.

---

# 25. Reporting Principle

The final dissertation should not present experiments as isolated leaderboard rows.

The analysis should connect:

- evidence availability,
- retrieval quality,
- model scale,
- prompting strategy,
- translation,
- adversarial robustness,
- language-specific effects,
- label-specific effects,
- qualitative error mechanisms.

The central research narrative should explain not only:

- which system performs best,

but also:

- why performance changes,
- where systems fail,
- which components create bottlenecks,
- how low-resource multilingual conditions affect evidence-grounded fact verification.
---

# Cross-Experiment Error Analysis and Final Synthesis

## Purpose of the analysis

A cross-experiment error analysis was conducted to move beyond aggregate performance metrics and determine how different interventions changed model behaviour at the individual-example level. The analysis compared six principal systems:

- Qwen 1.5B Claim Only
- Qwen 1.5B Gold Evidence
- Qwen 1.5B BM25 Evidence
- Qwen 1.5B Adversarial Evidence
- Qwen 14B Gold Evidence
- Qwen 14B Translated Gold Evidence

All six systems were compared over the same 492-example evaluation set, enabling matched example-level transition analysis.

The analysis addressed four main questions:

1. How much does access to evidence improve over claim-only classification?
2. Does automatically retrieved BM25 evidence reproduce the benefit of oracle gold evidence?
3. How vulnerable is fact verification to misleading or adversarial evidence?
4. How do model scale and translation affect the correction and introduction of errors?

---

## 1. Overall performance hierarchy

The six principal systems ranked as follows by accuracy:

| Rank | Experiment | Accuracy |
|---|---|---:|
| 1 | Qwen 14B Translated Gold Evidence | 0.735772 |
| 2 | Qwen 14B Gold Evidence | 0.640244 |
| 3 | Qwen 1.5B Gold Evidence | 0.459350 |
| 4 | Qwen 1.5B BM25 Evidence | 0.394309 |
| 5 | Qwen 1.5B Claim Only | 0.390244 |
| 6 | Qwen 1.5B Adversarial Evidence | 0.333333 |

### Interpretation

The ranking indicates that performance differs substantially across conditions varying model checkpoint, evidence source, and input-language representation. Three experimentally relevant dimensions are:

- model/checkpoint choice;
- evidence condition;
- input-language representation.

The strongest result was obtained by translating low-resource-language claims and evidence before classification with Qwen 14B. This result is consistent with the model achieving stronger verification performance when content is presented in English. However, because translation can alter semantic content, the experiment does not establish that every translated input is semantically equivalent to its original or that linguistic accessibility alone causes the gain.

The lower BM25 and adversarial results show that performance is substantially weaker under these evidence conditions than under oracle gold evidence. This indicates that supplying evidence is not automatically beneficial in the evaluated setup.

---

## 2. Evidence access: claim-only to gold evidence

Transition from Qwen 1.5B Claim Only to Qwen 1.5B Gold Evidence produced:

- correct to correct: 74
- wrong to correct: 152
- correct to wrong: 118
- wrong to wrong: 148
- net correct gain: +34

Accuracy increased from 0.390244 to 0.459350.

### Interpretation

Gold evidence improved aggregate accuracy, but the matched transition analysis reveals substantial instability. Although 152 previously incorrect examples were repaired, 118 previously correct examples became incorrect after evidence was introduced.

This is important because aggregate accuracy alone hides the fact that introducing gold evidence is associated with both repaired and regressed predictions. Under this setup, the 1.5B model does not convert access to gold evidence into uniform example-level improvement. Possible contributors include lexical overlap, partial semantic cues, or distracting details within longer contexts, but these mechanisms were not independently isolated.

Therefore, the finding is not simply that evidence improves every prediction. A more accurate conclusion is:

> Under the evaluated Qwen 1.5B setup, gold evidence improves aggregate accuracy while producing substantial bidirectional example-level transitions, indicating that access to relevant evidence does not guarantee consistent use of that evidence.

---

## 3. Gold evidence versus BM25 retrieval

Transition from Qwen 1.5B Gold Evidence to Qwen 1.5B BM25 Evidence produced:

- correct to correct: 171
- wrong to correct: 23
- correct to wrong: 55
- wrong to wrong: 243
- net correct gain: -32

Accuracy fell from 0.459350 with gold evidence to 0.394309 with BM25 evidence.

### Interpretation

Replacing oracle gold evidence with top-ranked BM25 retrieval was associated with a net loss of 32 correct predictions.

Only 23 previously wrong examples became correct under the BM25 condition, whereas 55 previously correct examples became wrong. This indicates a clear oracle-to-retrieved evidence performance gap.

Because the classifier is unchanged across these conditions, the result is consistent with the supplied evidence being an important constraint on downstream classification. However, the observed gap should be interpreted within the exact pipeline setup rather than as a pure causal estimate of retrieval error unless all other processing conditions are confirmed equivalent.

This supports a central dissertation argument:

> Retrieval-augmented misinformation detection depends not only on classifier capability but also on whether the retrieval stage supplies evidence that is useful for verification.

---

## 4. BM25 support bias

The Qwen 1.5B BM25 system displayed a strong prediction bias:

- supports: 394 predictions (80.08%)
- refutes: 69 predictions (14.02%)
- NEI: 29 predictions (5.89%)

A dedicated case analysis identified 245 BM25 support-bias failures:

- 126 gold NEI examples
- 119 gold refutes examples

By language:

- Igbo: 87
- Yoruba: 83
- Hausa: 75

### Interpretation

This is a strong descriptive behavioural finding in the evaluated BM25 pipeline.

The BM25 system overwhelmingly predicted `supports`, assigning this label to more than four-fifths of the evaluation set. The identified support-bias failures were distributed across all three languages.

A plausible mechanism is lexical-overlap bias. Because BM25 ranks passages using term-based matching rather than contradiction-aware verification, retrieved passages may share entities, topics, or surface vocabulary with a claim without actually supporting it.

One hypothesis is that, when such topically related evidence is passed to the LLM, relevance may sometimes be over-interpreted as entailment. The following should therefore be treated as a hypothesised failure chain rather than an established causal sequence:

1. BM25 rewards term-based similarity.
2. A lexically or topically related passage is retrieved.
3. The passage appears relevant to the claim.
4. The classifier may over-interpret relevance as support.
5. Some refuted and insufficient-information claims are classified as `supports`.

The result provides an important distinction between:

- retrieval relevance;
- verification usefulness.

A passage can be relevant to a claim while still being insufficient for determining whether the claim is true or false.

---

## 5. Adversarial evidence damage

Transition from Qwen 1.5B Gold Evidence to Qwen 1.5B Adversarial Evidence produced:

- correct to correct: 89
- wrong to correct: 75
- correct to wrong: 137
- wrong to wrong: 191
- net correct gain: -62

Accuracy fell from 0.459350 to 0.333333.

A total of 137 examples that were correct under gold evidence became wrong under adversarial evidence.

### Interpretation

The adversarial condition produced the largest negative net transition among the analysed interventions.

The result demonstrates that evidence-conditioned fact verification systems can be highly vulnerable to misleading context. A model that produces a correct decision when given appropriate evidence may reverse that decision when exposed to plausible but unsuitable evidence.

This has direct implications for real-world RAG systems. Retrieval pipelines operating over noisy web corpora may encounter:

- irrelevant passages;
- outdated information;
- misleading context;
- contradictory sources;
- semantically related but non-verifying evidence.

Therefore, retrieval quality should be considered a reliability and robustness problem rather than merely a search-performance problem.

A key dissertation interpretation is:

> In retrieval-augmented fact verification, poor evidence can be worse than no evidence.

This conclusion is supported by the adversarial system's accuracy of 0.333333, which was below the claim-only system's 0.390244.

---

## 6. Effect of model scaling

Transition from Qwen 1.5B Gold Evidence to Qwen 14B Gold Evidence produced:

- correct to correct: 169
- wrong to correct: 146
- correct to wrong: 57
- wrong to wrong: 120
- net correct gain: +89

Accuracy increased from 0.459350 to 0.640244.

### Interpretation

Model scaling produced the largest positive net gain among the matched pairwise interventions.

The 14B model repaired 146 errors made by the 1.5B model while introducing only 57 regressions, yielding a net gain of 89 correct examples.

This provides strong evidence of a substantial performance difference between the evaluated Qwen 1.5B and Qwen 14B checkpoints under the same gold-evidence condition. The result is consistent with model scale contributing to low-resource fact-verification performance, but the comparison does not isolate parameter count from other checkpoint-specific differences.

Possible contributors to the larger model's performance include:

- stronger multilingual representations;
- improved contextual reasoning;
- better entity and relation tracking;
- stronger contradiction recognition;
- more robust instruction following.

These contributors remain hypotheses because they were not independently ablated.

However, 120 examples remained wrong under both 1.5B and 14B gold-evidence conditions. Scaling therefore substantially improves performance but does not eliminate systematic failure.

---

## 7. Translation as a model-access intervention

Transition from Qwen 14B Gold Evidence to Qwen 14B Translated Gold Evidence produced:

- correct to correct: 268
- wrong to correct: 94
- correct to wrong: 47
- wrong to wrong: 83
- net correct gain: +47

Accuracy increased from:

- original low-resource input: 0.640244
- translated English input: 0.735772

Absolute accuracy gain:

- +0.095528
- approximately +9.55 percentage points

Macro F1 increased from:

- 0.600194
- to 0.733508

### Interpretation

The translated-input condition produced a substantial improvement while using the same Qwen 14B model checkpoint.

The improvement therefore cannot be attributed to increased model size between these two conditions. It is consistent with input-language representation materially affecting verification performance. However, because translation can also alter semantic content, the experiment does not isolate linguistic accessibility from translation-induced changes to the proposition or evidence.

A central interpretation is:

> Translation can function as a model-access intervention by mapping low-resource-language content into English, where the evaluated LLM achieves stronger verification performance; however, this benefit is non-uniform and may be accompanied by semantic distortion.

The transition analysis showed 47 correct-to-wrong regressions. Translation is therefore beneficial overall in aggregate performance but non-uniform at the example level.

---

## 8. Translation effect by language

Per-language accuracy changed as follows:

| Language | Examples | Original Accuracy | Translated Accuracy | Absolute Change |
|---|---:|---:|---:|---:|
| Igbo | 163 | 0.607362 | 0.742331 | +0.134969 |
| Hausa | 166 | 0.650602 | 0.759036 | +0.108434 |
| Yoruba | 163 | 0.662577 | 0.705521 | +0.042945 |

### Interpretation

Translation improved overall performance in all three languages, but the magnitude varied substantially.

The largest gain occurred for Igbo:

- +13.50 percentage points

Hausa improved by:

- +10.84 percentage points

Yoruba improved by only:

- +4.29 percentage points

This shows that the observed translation-associated performance change is not uniform across languages in the evaluated sample.

The smaller Yoruba gain later proved especially important because class-specific analysis revealed substantial damage to Yoruba `supports` examples.

---

## 9. Translation effect by verification label

Per-label accuracy changed as follows:

| Gold Label | Examples | Original Accuracy | Translated Accuracy | Absolute Change |
|---|---:|---:|---:|---:|
| supports | 166 | 0.668675 | 0.602410 | -0.066265 |
| refutes | 152 | 0.250000 | 0.625000 | +0.375000 |
| NEI | 174 | 0.954023 | 0.959770 | +0.005747 |

### Interpretation

The overall translation-associated gain was driven primarily by improved performance on `refutes` examples.

For `refutes`, accuracy increased from 25.00% to 62.50%:

- +37.50 percentage points

By contrast:

- `supports` declined by 6.63 percentage points;
- `NEI` remained almost unchanged.

This is a major result because it shows that the 9.55-point overall gain was not a uniform improvement across classes.

The pattern is consistent with the translated-input condition making some contradiction relations more accessible to the model, although the aggregate class-level result does not by itself establish the mechanism.

---

## 10. Translation and the refutes class

Translation repaired 64 `refutes` examples that were previously wrong.

Per-language `refutes` changes were:

| Language | Original Accuracy | Translated Accuracy | Change |
|---|---:|---:|---:|
| Hausa | 0.255814 | 0.627907 | +0.372094 |
| Igbo | 0.290909 | 0.690909 | +0.400000 |
| Yoruba | 0.203704 | 0.555556 | +0.351852 |

### Interpretation

The improvement in `refutes` accuracy was not isolated to one language.

Refutes accuracy increased by:

- +40.00 percentage points for Igbo;
- +37.21 percentage points for Hausa;
- +35.19 percentage points for Yoruba.

This descriptively consistent cross-language pattern is compatible with the hypothesis that translated wording improves model access to some contradiction cues, although the mechanism is not directly isolated by the experiment.

Under the original-language condition, many refuted claims were mapped to NEI. After translation, a substantial number of previously incorrect `refutes` examples became correct. This transition pattern is consistent with improved recognition of contradiction relations, but should not be treated as proof of a single causal mechanism.

---

## 11. Representative translation repair cases

### Igbo: first versus second

Claim:

> Bianca Ojukwu is the second female CEO of the Nigerian Stock Exchange.

Evidence:

> She is the first woman to hold a position as the head of the Nigerian Stock Exchange.

Prediction changed:

- original: NEI
- translated: refutes

### Interpretation

In this illustrative case, the translated wording presented a direct ordinal contrast between `second` and `first`, alongside a change from NEI to the correct `refutes` prediction.

---

### Hausa: award versus prosecution

Claim:

> Governor Shema received an award of honor from the EFCC.

Evidence:

> The EFCC prosecuted Governor Shema on corruption charges involving approximately N11 billion.

Prediction changed:

- original: supports
- translated: refutes

### Interpretation

In this illustrative case, the translated representation presented a clearer contrast between the claim's award relation and the evidence's prosecution relation. This is a plausible explanation for the corrected `refutes` prediction rather than proof of the mechanism.

---

### Yoruba: Dauda versus Ibrahim

Claim:

> God made a promise to Dauda that if she had a child, she would worship him.

Evidence referred to Ibrahim and the dedication of a child to God.

Prediction changed:

- original: NEI
- translated: refutes

### Interpretation

In this illustrative case, the translated wording presented an entity mismatch between Dauda and Ibrahim, which is a plausible contributor to the corrected `refutes` prediction.

---

## 12. Translation damage and Yoruba supports examples

Despite the overall translation improvement, Yoruba `supports` accuracy fell from:

- 0.854545
- to 0.581818

Absolute change:

- -0.272727
- approximately -27.27 percentage points

A dedicated case extraction identified 20 Yoruba `supports` examples that changed from correct to incorrect after translation.

### Interpretation

This is an important counter-result.

Translation is not uniformly beneficial at the example or language-label level. In the evaluated Yoruba `supports` subset, 20 previously correct examples became incorrect after translation.

The Yoruba result prevents an overly simplistic conclusion that English translation always improves low-resource verification. Qualitative inspection of selected regression cases identifies examples consistent with semantic distortion, but translation quality was not manually annotated across the full subset.

A more defensible conclusion is:

> Translation provides a strong aggregate benefit in this experiment but is also associated with substantial language- and label-specific regressions; selected qualitative cases suggest that semantic corruption is one plausible contributor.

---

## 13. Representative translation damage cases

### Ondo State ranking distortion

Original claim stated that Ondo was the 19th most populous state.

Translated claim preserved:

> 19th most populous state

However, translated evidence changed the relevant statement to:

> 11th largest state

The model changed:

- supports
- to NEI

### Interpretation

In this illustrative case, the translated evidence altered the numerical ranking and measured property, removing the apparent entailment relation present in the original example.

---

### Gaari mistranslated as sugar

Original claim concerned:

- white gaari
- yellow gaari

Translated claim became:

> White sugar and brown sugar are two types of sugar found in western Nigeria.

### Interpretation

In this illustrative case, the culturally specific food term `gaari` was rendered as `sugar`, changing the subject of the translated claim.

---

### Breastfeeding semantic corruption

The Yoruba statement concerning breastfeeding and lives saved was translated into language resembling:

> if the entire world community donates a child...

### Interpretation

In this illustrative case, the translated wording appears to alter the core proposition substantially, alongside a regression in the verification prediction.

---

## 14. Generic prompt examples after translation

The translated Qwen 14B baseline achieved:

- accuracy: 0.735772
- macro F1: 0.733508

Adding generic prompt examples reduced performance to:

- accuracy: 0.621951
- macro F1: 0.593794

### Interpretation

Under this translated-input setup, adding generic examples was associated with substantially lower performance.

This is an important negative result. Prompt examples are not automatically beneficial, even when they appear logically clear and task-relevant.

One plausible explanation is mismatch between the generic demonstrations and the distributional characteristics of the Afrifact evaluation data. Possible mechanisms include:

- altered label preferences;
- superficial analogy to demonstration patterns;
- interference with dataset-specific evidence interpretation;
- poor transfer of generic verification patterns to culturally and linguistically specific claims.

These mechanisms were not independently isolated and should therefore be presented as hypotheses.

The finding supports the conclusion that:

> Few-shot or demonstration-based prompting should be evaluated empirically rather than assumed to improve performance.

---

# Cross-System Behavioural Analysis

## 15. Prediction distributions reveal distinct system biases

Prediction distributions were:

### Qwen 1.5B Claim Only

- supports: 95 (19.31%)
- refutes: 183 (37.20%)
- NEI: 214 (43.50%)

### Qwen 1.5B Gold Evidence

- supports: 373 (75.81%)
- refutes: 51 (10.37%)
- NEI: 68 (13.82%)

### Qwen 1.5B BM25 Evidence

- supports: 394 (80.08%)
- refutes: 69 (14.02%)
- NEI: 29 (5.89%)

### Qwen 1.5B Adversarial Evidence

- supports: 155 (31.50%)
- refutes: 66 (13.41%)
- NEI: 271 (55.08%)

### Qwen 14B Gold Evidence

- supports: 141 (28.66%)
- refutes: 45 (9.15%)
- NEI: 306 (62.20%)

### Qwen 14B Translated Gold Evidence

- supports: 105 (21.34%)
- refutes: 113 (22.97%)
- NEI: 274 (55.69%)

### Interpretation

The systems do not simply differ in accuracy. Their observed prediction distributions differ substantially.

The 1.5B gold and BM25 systems assign `supports` to a large majority of examples.

The 14B gold model assigns NEI to a large majority of examples.

Under the translated-input condition, the proportion of `refutes` predictions increases substantially, alongside the observed increase in `refutes` accuracy.

These descriptive distributions indicate that changes in evidence condition, model checkpoint, and input-language representation are associated with different class-prediction patterns rather than uniform improvement across all classes.

---

## 16. Best system differs by label

Best observed system by class:

| Label | Best System | Accuracy |
|---|---|---:|
| NEI | Qwen 14B Translated | 0.959770 |
| refutes | Qwen 14B Translated | 0.625000 |
| supports | Qwen 1.5B Gold | 0.921687 |

### Interpretation

No single evaluated configuration achieves the highest class-specific accuracy for every verification label.

The translated 14B model achieves the highest observed accuracy for:

- `refutes`;
- NEI.

However, Qwen 1.5B Gold Evidence achieves the highest observed `supports` accuracy at 92.17%.

This reinforces the finding that observed system performance is class-dependent and may involve trade-offs.

The smaller model's very high `supports` accuracy must also be interpreted alongside the fact that 75.81% of all its predictions are `supports`. High class-specific accuracy does not necessarily imply balanced verification competence.

---

## 17. Best model by language

Among the evaluated systems, the translated Qwen 14B model achieved the highest overall accuracy for all three languages:

| Language | Accuracy |
|---|---:|
| Hausa | 0.759036 |
| Igbo | 0.742331 |
| Yoruba | 0.705521 |

### Interpretation

Among the evaluated systems, the translated 14B condition provides the strongest overall performance for each of the three target-language subsets.

However, the lower Yoruba result coexists with the substantial observed regression in Yoruba `supports` accuracy after translation.

Thus, the same system can achieve the highest aggregate language-level accuracy while still exhibiting important language-label-specific weaknesses.

---

# Universally Hard Examples

## 18. Universal error concentration

Across all six principal systems:

- 29 examples were wrong for every system;
- only 12 examples were correct for every system.

Among the 29 universally hard examples:

### By language

- Yoruba: 13
- Hausa: 10
- Igbo: 6

### By gold label

- refutes: 25
- supports: 4

### Interpretation

The most striking descriptive result is the concentration of universal failures in the `refutes` class.

25 of 29 examples misclassified by all six principal systems were refutations.

Within this 29-example universal-error subset, persistent failure is therefore concentrated in `refutes` examples across the evaluated:

- model checkpoints;
- claim-only condition;
- gold-evidence condition;
- BM25-retrieval condition;
- adversarial-evidence condition;
- translated-input condition.

This pattern indicates that the identified hard refutations persist across multiple evaluated configurations. Because the subset contains only 29 examples and the systems are not an exhaustive sample of architectures or prompting strategies, the result should not be generalised to all fact-verification systems.

---

## 19. Persistent hard refutes by language

The 25 persistent hard refutes were distributed as:

- Yoruba: 13
- Hausa: 8
- Igbo: 4

### Interpretation

Yoruba accounts for 13 of the 25 persistent hard `refutes` examples in this subset.

This coexists with other Yoruba-specific patterns in the current pipeline, including the substantial `supports`-class degradation after translation. However, these observations concern different subsets and should not by themselves be interpreted as evidence of a single shared language-level cause.

Because the hard-error subset is small, these findings should be interpreted diagnostically rather than as population-wide statistical estimates.

---

# Manual Hard-Refute Taxonomy

## 20. Purpose and methodological status

The 25 persistent hard refutes were manually reviewed and assigned a primary error category.

This analysis is exploratory and diagnostic.

It should not be presented as a statistically representative taxonomy of all dataset errors. Instead, it identifies recurring mechanisms among the most persistent failures.

---

## 21. Primary error categories

| Primary Error Type | Count | Percentage |
|---|---:|---:|
| evidence insufficient | 6 | 24% |
| entity-attribute mismatch | 5 | 20% |
| negation | 3 | 12% |
| role-relation mismatch | 3 | 12% |
| temporal mismatch | 2 | 8% |
| boundary condition | 2 | 8% |
| causal mismatch | 1 | 4% |
| implicit contradiction | 1 | 4% |
| possible annotation issue | 1 | 4% |
| numerical mismatch | 1 | 4% |

### Interpretation

The largest category was `evidence_insufficient`, accounting for 24% of the 25 persistent hard refutes.

This suggests that some examples in this manually reviewed subset may not contain evidence that clearly establishes contradiction.

The second-largest category was `entity_attribute_mismatch` at 20%. Within this subset, persistent failures therefore also occurred when contradiction depended on differences in properties, professions, identities, origins, or other entity characteristics.

Negation and role-relation mismatches each accounted for 12%.

Overall, the manually reviewed persistent contradiction errors were heterogeneous and were not reducible to a single observed category such as negation failure.

---

## 22. Contradiction explicitness

Among the 25 persistent hard refutes:

- explicit: 11 (44%)
- implicit: 7 (28%)
- ambiguous: 7 (28%)

### Interpretation

Within the 25-example persistent hard-refute subset, 44% were manually categorised as explicit contradictions.

A combined 56% were categorised as either implicit or ambiguous.

This pattern offers one plausible explanation for the persistence of errors within this subset: many reviewed cases appeared to require more than surface-level contradiction detection.

Relevant reasoning demands in the reviewed cases included:

- entity attributes;
- roles;
- temporal relations;
- causal relations;
- category membership;
- boundary conditions;
- unstated implications.

The analysis is therefore consistent with the argument that some low-resource fact-verification cases require relational and inferential reasoning rather than lexical comparison alone.

---

## 23. Annotation suspicion

Manual review produced:

- no annotation issue suspected: 14 (56%)
- annotation issue suspected: 7 (28%)
- uncertain: 4 (16%)

### Interpretation

Only 56% of the 25 persistent hard refutes were manually judged clearly free from annotation suspicion.

For 28%, an annotation issue was suspected, while another 16% remained uncertain.

Thus, 44% of this selected persistent-error subset had some degree of annotation concern under the manual review.

This should be interpreted cautiously. It does not establish that 44% of the full dataset is incorrectly annotated, nor does the selected subset support a population-wide prevalence estimate.

Instead, the review suggests that some of the most persistent observed failures coincide with examples where the evidence-label relationship may be unclear or debatable.

A defensible conclusion is:

> Some observed persistent errors may reflect dataset or evidence-quality limitations in addition to model limitations.

---

## 24. Hard-error taxonomy by language

Observed category counts were:

### Hausa

- boundary condition: 1
- causal mismatch: 1
- entity-attribute mismatch: 2
- evidence insufficient: 2
- implicit contradiction: 1
- temporal mismatch: 1

### Igbo

- evidence insufficient: 1
- negation: 1
- role-relation mismatch: 2

### Yoruba

- boundary condition: 1
- entity-attribute mismatch: 3
- evidence insufficient: 3
- negation: 2
- numerical mismatch: 1
- possible annotation issue: 1
- role-relation mismatch: 1
- temporal mismatch: 1

### Interpretation

Within the manually reviewed persistent hard-refute subset, the observed category composition differed by language.

Yoruba displayed the broadest spread of assigned error categories and contained the largest number of persistent hard refutes.

The four Igbo cases were assigned to role-relation mismatch, negation, and insufficient-evidence categories.

The eight Hausa cases spanned entity, causal, temporal, boundary, insufficient-evidence, and implicit-contradiction categories.

Given the small and selected subset sizes, these patterns should be described as qualitative tendencies within the reviewed cases rather than definitive language-wide distributions.

---

# Integrated Dissertation-Level Interpretation

## 25. Evidence quality matters more than evidence presence

Across experiments, merely supplying context does not guarantee improvement.

Observed pattern:

- claim only: 39.02%
- gold evidence: 45.94%
- BM25 evidence: 39.43%
- adversarial evidence: 33.33%

### Interpretation

In this Qwen 1.5B comparison, gold evidence improves aggregate performance, the BM25-evidence condition achieves accuracy close to the claim-only condition, and the adversarial-evidence condition performs worse than claim-only classification.

Therefore, the evaluated results support the interpretation that:

> The value of supplied evidence depends on its relevance and reliability.

This is a central finding of the evaluated retrieval-augmented misinformation-detection setup.

---

## 26. Retrieval is a major bottleneck

The BM25 system showed:

- only 23 wrong-to-correct repairs relative to gold evidence;
- 55 correct-to-wrong regressions;
- net loss of 32;
- 80.08% supports prediction rate;
- 245 identified support-bias failures.

### Interpretation

Replacing gold evidence with top-ranked BM25 evidence was associated with substantial changes in downstream predictions.

The results suggest that the evaluated lexical-retrieval setup may be insufficient for cases requiring:

- contradiction-aware matching;
- semantic distinction;
- entity-relation reasoning;
- temporal reasoning;
- recognition that relevant evidence is not necessarily supporting evidence.

Future work could evaluate:

- dense multilingual retrieval;
- hybrid sparse-dense retrieval;
- cross-encoder reranking;
- contradiction-aware reranking;
- retrieval confidence thresholds;
- abstention when evidence quality is weak.

---

## 27. Scaling and translation show distinct performance associations

Scaling from 1.5B to 14B produced:

- +89 net correct examples

Translation at 14B produced:

- +47 net correct examples

### Interpretation

Both interventions were associated with substantial gains, but the current comparisons do not isolate their underlying mechanisms.

Increasing model scale from 1.5B to 14B under the gold-evidence condition produced the larger matched net gain. Translating the inputs to English while holding the 14B model condition fixed produced a further gain.

One plausible interpretation is that model capacity and input-language representation affect different aspects of evidence-conditioned verification. However, the experiments do not establish a two-stage causal mechanism or prove that the interventions are complementary in a factorial sense.

A hypothesis for future testing is:

1. greater model capacity may improve use of evidence;
2. a more accessible linguistic representation may improve performance for some low-resource inputs.

---

## 28. Translation is powerful but asymmetric

Translation:

- improved overall accuracy by approximately 9.55 points;
- improved Igbo by approximately 13.50 points;
- improved Hausa by approximately 10.84 points;
- improved Yoruba by approximately 4.29 points;
- improved refutes by 37.50 points;
- reduced supports by 6.63 points;
- reduced Yoruba supports by approximately 27.27 points.

### Interpretation

In the evaluated setup, translation was associated with a large but asymmetric performance change.

It repaired many previously incorrect `refutes` predictions while also introducing regressions, particularly for `supports`. Qualitative inspection of selected cases identified errors consistent with mistranslation of culturally specific terms, numerical relations, and other semantic details, but these mechanisms were not manually annotated across all examples.

The findings motivate evaluation of language-aware translation quality controls rather than assuming that translation is uniformly beneficial.

---

## 29. Persistent errors are concentrated in contradiction cases

Evidence:

- 25 of 29 universally hard examples were refutes;
- original Qwen 14B refutes accuracy was only 25%;
- translation improved it to 62.5%, but substantial errors remained;
- 56% of persistent hard contradictions were implicit or ambiguous;
- persistent errors included entity, role, temporal, numerical, causal, and boundary mismatches.

### Interpretation

A strong overarching conclusion from the evaluated error analyses is:

> Persistent errors in the evaluated systems are disproportionately concentrated in `refutes` examples.

The manual review further suggests that some of these cases involve implicit or ambiguous relations rather than explicit contradiction cues. This is consistent with the hypothesis that inferential structure contributes to difficulty alongside language representation, although the present experiments do not establish contradiction recognition as the principal unresolved challenge for low-resource fact verification in general.

---

## 30. Dataset quality may contribute to observed persistent errors

Among 25 persistent hard refutes:

- 28% had suspected annotation issues;
- 16% were uncertain;
- 24% were primarily categorised as evidence insufficient.

### Interpretation

Manual review suggests that some apparent model failures in the selected persistent hard-refute subset may be partly attributable to weak or debatable evidence-label alignment.

This is important for evaluation. In cases where the supplied evidence does not clearly refute the claim, an NEI prediction may be semantically defensible even when scored as incorrect against the dataset label.

Therefore, model performance should be interpreted alongside possible dataset and evidence-quality limitations rather than as a pure measure of reasoning ability.

---

# Final Research Narrative

The complete experimental evidence supports the following research narrative:

1. Under the evaluated claim-only condition, Qwen 1.5B achieved limited accuracy.
2. Gold evidence improved aggregate Qwen 1.5B performance, but matched transitions showed both repairs and regressions.
3. The evaluated BM25 condition did not reproduce the gold-evidence gain and exhibited a very high `supports` prediction rate.
4. The adversarial-evidence condition performed below claim-only classification.
5. Increasing model scale from Qwen 1.5B to 14B under gold evidence produced a substantial matched performance gain.
6. Translating the evaluated low-resource inputs into English produced the highest overall accuracy among the tested systems.
7. The aggregate translation gain was driven primarily by improved performance on `refutes` examples.
8. Translation was also associated with substantial regressions, especially for Yoruba `supports`; selected qualitative cases were consistent with several forms of semantic distortion.
9. Generic prompt demonstrations reduced performance in the evaluated translated-input setup.
10. Among the 29 examples missed by all six principal systems, 25 were `refutes`.
11. Manual review of 25 persistent hard refutes found that 56% were categorised as implicit or ambiguous.
12. Within that selected persistent-error subset, a substantial proportion of cases raised some degree of evidence-label or annotation concern.
13. Collectively, the experiments indicate that end-to-end performance is associated with multiple components, including retrieval quality, input-language representation, model scale, contradiction-related reasoning demands, and evidence-label quality.

## Core dissertation contribution emerging from the experiments

The experiments collectively suggest that low-resource retrieval-augmented misinformation detection should be evaluated across at least four interacting components:

- linguistic representation;
- retrieval quality;
- model capacity;
- evidence-label validity.

The current experiments show that weaknesses in these components can be associated with substantial end-to-end performance changes.

The strongest observed system used translated inputs and the larger evaluated LLM, while the error analysis showed that this configuration still made substantial errors, particularly on `refutes` examples, and that some persistent cases raised evidence-quality concerns.

This provides a richer conclusion than simply identifying the highest-performing model. The dissertation characterises how performance and error patterns change across the evaluated retrieval, model-scale, translation, prompting, and adversarial-evidence conditions, while distinguishing observed effects from hypothesised mechanisms.

