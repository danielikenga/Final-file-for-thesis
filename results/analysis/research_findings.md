Research Findings Log

Dissertation Project
Topic: Retrieval-Augmented Misinformation Detection for Low-Resource Nigerian Languages
Languages: Hausa, Igbo, Yoruba
Evaluation Test Set: 492 examples
Primary Labels: supports, refutes, nei


1. Purpose of This Findings Log

This document records the quantitative results, interpretations, and qualitative case analysis produced during the experimental evaluation, so the reasoning behind the dissertation's results is preserved and not just the final tables.


2. Current Master Experimental Results

2.1 Encoder and LLM Results

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
| Qwen 14B + Google Translate | 0.725610 | 0.723831 | 0.801612 | 0.718465 |
| Qwen 14B + Translate + Generic Examples | 0.638211 | 0.615197 | 0.802113 | 0.623058 |

These figures match master_results.csv and the submitted dissertation.


3. Initial Cross-Experiment Findings

3.1 Comparable encoder baselines

XLM-R Custom Split: 45.94% accuracy, 44.22% macro-F1.
AfriBERTa Custom Split: 49.80% accuracy, 49.23% macro-F1.

AfriBERTa beats XLM-R by 3.86 points accuracy and 5.01 points macro-F1 on the shared 492-example set.

Interpretation: AfriBERTa outperforming XLM-R is consistent with African-language-focused pretraining being useful for Hausa/Igbo/Yoruba classification, but this one comparison doesn't isolate pretraining focus as the cause — architecture and optimisation differences aren't ruled out.

3.2 Gold-evidence encoder performance

| | Claim-only Acc. | Gold-Ev. Acc. | Delta Acc. | Claim-only F1 | Gold-Ev. F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| XLM-R | 45.94% | 63.01% | +17.07 | 44.22% | 58.92% | +14.71 |
| AfriBERTa | 49.80% | 57.11% | +7.32 | 49.23% | 56.87% | +7.64 |

Gold evidence helps both encoders, more so for XLM-R than AfriBERTa. Read as evidence availability mattering for the task, not as a fully isolated causal effect, since other pipeline conditions aren't held identical.

3.3 Methodological comparability note

Earlier claim-only encoder runs produced 1,950 predictions but only 384 of the 492 custom-test claims were represented in them (108 missing). Those legacy outputs are kept for reference but excluded from the master comparison — comparing them directly to the 492-example experiments would confound model performance with which examples were even evaluated. The custom-split runs are the primary baseline for both encoders.


4. Model Scale Findings

Qwen 1.5B Gold Evidence: 45.94% / 40.04% (acc/F1).
Qwen 14B Gold Evidence: 64.02% / 60.02%.
Improvement: +18.09 points accuracy, +19.98 points macro-F1.

Interpretation: 14B substantially outperforms 1.5B under gold evidence, consistent with the larger model handling multilingual semantics, claim-evidence alignment, and contradiction recognition better, though checkpoint-specific differences beyond raw parameter count aren't isolated. Worth remembering: even the larger model only reaches 64.02% on original-language input.


5. Few-Shot Prompting Findings

Qwen 14B Gold Evidence baseline: 64.02% / 60.02%.
3-shot: 64.02% / 59.57%. 6-shot: 64.43% / 59.28%. 9-shot: 63.21% / 58.25%.

The best few-shot accuracy (6-shot, 64.43%) is barely above zero-shot, and macro-F1 actually falls as shots increase. 9-shot underperforms the baseline on both metrics.

Interpretation: more demonstrations aren't automatically better. Possible causes — prompt interference, demonstration mismatch, label bias from the sampled examples — weren't isolated individually, so this stays a set of candidate explanations rather than a confirmed mechanism. Few-shot performance was non-monotonic here; six examples gave a marginal accuracy gain at the cost of macro-F1, and nine examples underperformed zero-shot on both metrics.


6. Generic Prompt Examples After Translation

Translation-only Qwen 14B: 72.56% / 72.38%.
Translation + generic examples: 63.82% / 61.52%.
Change: -8.74 points accuracy, -10.86 points macro-F1.

Interpretation: replacing in-domain demonstrations with generic, unrelated ones under the translated condition costs almost nine accuracy points, a genuine negative result worth reporting on its own terms. Plausible drivers include mismatch between the generic examples and AfriFact's actual claim style — culturally specific entities, evidence structure — but demonstration relevance wasn't manipulated independently of demonstration content, so the mechanism is inferred, not proven. The usefulness of few-shot demonstrations depends on their relevance to the evaluation domain, not just their presence.


7. BM25 Retrieval Findings

Recall@1 = 17.68%, Recall@3 = 46.75%, Recall@5 = 57.32%, Recall@10 = 63.01%.

Interpretation: the correct evidence is ranked first for under a fifth of queries, but appears somewhere in the top 10 for nearly two-thirds. That gap points to candidate generation and candidate ranking being distinct problems — BM25 is finding relevant passages more often than it's ranking them first. The retrieval bottleneck isn't total failure to find evidence; a large share of relevant evidence sits in deeper candidate positions, which is exactly what reranking is for.


8. End-to-End BM25 Evidence Verification

Qwen 1.5B Gold Evidence: 45.94% / 40.04%.
Qwen 1.5B BM25 Evidence: 39.43% / 31.82%.

Interpretation: swapping oracle evidence for BM25's top-ranked passage drops both metrics — the oracle-to-retrieved gap. Because the classifier is unchanged across both conditions, this is consistent with retrieval quality being a real constraint on downstream verification, though it shouldn't be read as an isolated causal estimate unless every other condition in the pipeline is confirmed equivalent.


9. Adversarial Evidence Findings

Qwen 1.5B Gold Evidence: 45.94% / 40.04%.
Qwen 1.5B Adversarial Evidence: 33.33% / 31.56%.

Interpretation: accuracy drops to about a third when evidence is deliberately mismatched, notably below the claim-only figure of 39.02%. This shows real sensitivity to misleading context, not just an absence-of-evidence effect. Misleading or mismatched context degrades decisions under this setup, supporting the case that both retrieval relevance and robustness to bad evidence matter for RAG-based fact-checking.


10. Translation Experiment

10.1 Overall result

Qwen 14B Original Gold Evidence: 64.02% / 60.02%.
Qwen 14B Translated Gold Evidence: 72.56% / 72.38%.
Improvement: +8.54 points accuracy, +12.36 points macro-F1.

Interpretation: translation gives the strongest result of any evaluated condition, but the gain is far from uniform across languages and labels, covered in sections 11-16. It shouldn't be read as translation being universally beneficial, or as direct evidence of improved reasoning about the original-language content itself.


11. Translation Effect by Language

| Language | n | Original | Translated | Delta |
|---|---:|---:|---:|---:|
| Igbo | 163 | 60.74% | 73.62% | +12.88 |
| Hausa | 166 | 65.06% | 75.30% | +10.24 |
| Yoruba | 163 | 66.26% | 68.71% | +2.45 |

Igbo gains the most, Yoruba the least. The later language-label breakdown (section 16) shows Yoruba's small aggregate gain hides a sharp drop in supports accuracy.


12. Translation Effect by Gold Label

| Label | n | Original | Translated | Delta |
|---|---:|---:|---:|---:|
| Supports | 166 | 66.87% | 59.04% | -7.83 |
| Refutes | 152 | 25.00% | 60.53% | +35.53 |
| NEI | 174 | 95.40% | 95.98% | +0.57 |

Central interpretation: translation's net gain is almost entirely a refutes-class effect. Refutes accuracy jumps 35.5 points; supports actually falls; NEI barely moves, since it was already near ceiling. This label asymmetry is one of the central empirical findings of the dissertation, consistent with translation improving contradiction recognition specifically rather than lifting verification ability uniformly.


13. Translation Transition Analysis

Across 492 examples: correct-to-correct = 267 (54.27%), wrong-to-correct = 90 (18.29%), correct-to-wrong = 48 (9.76%), wrong-to-wrong = 87 (17.68%).

Repaired: 90. Damaged: 48. Net: +42, which is 42/492 = 8.54%, matching the accuracy delta above exactly.

Translation is net-positive at the aggregate level, repairing roughly twice as many decisions as it damages, but 48 correct-to-wrong regressions is a real, non-trivial risk under the translated condition, not a rounding error.


14. Translation Transitions by Language

| Language | C-to-C | C-to-W | W-to-C | W-to-W | Net |
|---|---:|---:|---:|---:|---:|
| Hausa | 99 | 9 | 26 | 32 | +17 |
| Igbo | 82 | 17 | 38 | 26 | +21 |
| Yoruba | 86 | 22 | 26 | 29 | +4 |

Igbo gets the largest net repair. Yoruba's net gain is weakest: it repairs 26 errors but introduces 22 new ones, which is why its aggregate improvement (section 11) is the smallest of the three.


15. Translation Transitions by Label

| Label | C-to-C | C-to-W | W-to-C | W-to-W | Net |
|---|---:|---:|---:|---:|---:|
| NEI | 162 | 4 | 5 | 3 | +1 |
| Refutes | 31 | 7 | 61 | 53 | +54 |
| Supports | 74 | 37 | 24 | 31 | -13 |

Refutes alone accounts for +54 net correct predictions, more than offsetting a net loss of 13 on supports. NEI barely moves. This confirms section 12's finding at the transition level, not just the aggregate-accuracy level.


16. Language x Label Interaction

| Language | Label | n | Original | Translated | Delta |
|---|---|---:|---:|---:|---:|
| Hausa | NEI | 67 | 97.01% | 94.03% | -2.99 |
| Hausa | Refutes | 43 | 25.58% | 62.79% | +37.21 |
| Hausa | Supports | 56 | 57.14% | 62.50% | +5.36 |
| Igbo | NEI | 53 | 96.23% | 96.23% | 0.00 |
| Igbo | Refutes | 55 | 29.09% | 67.27% | +38.18 |
| Igbo | Supports | 55 | 58.18% | 58.18% | 0.00 |
| Yoruba | NEI | 54 | 92.59% | 98.15% | +5.56 |
| Yoruba | Refutes | 54 | 20.37% | 51.85% | +31.48 |
| Yoruba | Supports | 55 | 85.45% | 56.36% | -29.09 |

Refutes accuracy improves for every language (+31 to +38 points), a consistent cross-language pattern, though no significance test is claimed here specifically. The one real anomaly is Yoruba supports, down 29.09 points, the single largest language-label regression in the whole study. Yoruba's weak aggregate language-level gain (section 11) isn't because translation fails uniformly there — refutes and NEI both improve — it's this one large supports-class collapse dragging the average down.


17. Confusion Matrix Findings

17.1 Original-language Qwen 14B

| Gold vs Pred | Supports | Refutes | NEI |
|---|---:|---:|---:|
| Supports | 111 | 2 | 53 |
| Refutes | 27 | 38 | 87 |
| NEI | 3 | 5 | 166 |

Of 152 refutes examples, only 38 are correctly labelled refutes; 87 go to NEI, 27 to supports. The dominant original-language error is refutes to NEI.

17.2 Translated Qwen 14B

| Gold vs Pred | Supports | Refutes | NEI |
|---|---:|---:|---:|
| Supports | 98 | 11 | 57 |
| Refutes | 2 | 92 | 58 |
| NEI | 2 | 5 | 167 |

Correct refutes predictions rise from 38 to 92; refutes-to-supports errors nearly disappear (27 to 2). But supports-to-refutes errors rise from 2 to 11, the flip side of the same effect: the model is now more willing to call something a contradiction, which fixes most of the refutes problem but creates a smaller number of new false positives on supports.


18. Qualitative Translation Case Analysis

Representative cases were chosen systematically, from refutes examples that moved wrong-to-correct, and Yoruba supports examples that moved correct-to-wrong, using fixed text-length quantiles rather than manual cherry-picking, so the examples span a range of input lengths. These are illustrative of recurring patterns, not proof that every transition shares one cause.


19. Translation Repair Mechanisms

19.1 Ordinal contradiction exposure, afrifact_data_culture_igbo_013

NEI to refutes. Claim: "Bianca Ojukwu was the second female head of the Nigerian Stock Exchange." Evidence: "She was the first woman to hold the position." Translation makes the ordinal contrast (second vs first) explicit, a plausible reason the model catches the contradiction after translation but not before.

19.2 Event-role contradiction, afrifact_data_culture_hausa_169

Supports to refutes. Claim: "Governor Shema received an honour award from the EFCC." Evidence: "EFCC prosecuted Governor Shema over alleged corruption involving approximately N11 billion." Translation clarifies the contrast between "honoured by" and "prosecuted by" the same institution.

19.3 Entity mismatch, afrifact_data_culture_yoruba_149

NEI to refutes. Claim attributes a promise to Dauda/David; evidence concerns Ibrahim. Plausibly, the entity mismatch becomes easier to spot post-translation.

19.4 Explicit detention contradiction, afrifact_data_culture_yoruba_200

NEI to refutes. Claim states no EndSARS protesters were arrested; evidence discusses detainees in police custody. Translation surfaces the arrest/detention vocabulary more clearly. Caveat: the translated numerical wording in this example is visibly noisy, so it's not a clean case.


20. Translation Damage Mechanisms

20.1 Numerical/ranking corruption, afrifact_data_culture_yoruba_624

Supports to NEI. Claim: Ondo State as the 19th most populous state, supported by the original evidence. Translated evidence instead says "11th largest state," the translation changed the number, breaking the entailment.

20.2 Culturally specific lexical mistranslation, afrifact_data_culture_yoruba_555

Supports to NEI. Gaari (a cassava food product) is translated as "sugar," changing the subject of the claim entirely. This is a concrete illustration of a real risk for low-resource-language NLP: culturally specific vocabulary getting flattened into unrelated or overly generic English terms. One case doesn't establish how common this is across the dataset.

20.3 Predicate-level corruption, afrifact_data_culture_yoruba_358

Supports to refutes. A claim about global breastfeeding and lives saved is translated into something resembling "if the entire world community donates a child," the core proposition is corrupted, not just the wording.

20.4 Domain-specific distortion, afrifact_data_culture_yoruba_168

Supports to NEI. A football positional description becomes "back-handed footballer." Caveat: the evidence excerpt here may itself be incomplete, so this is suggestive rather than definitive.

20.5 Possible translation-amplified mismatch, afrifact_data_culture_yoruba_528

Supports to NEI. Translated claim calls Wizkid "one of the best Afro Beat musicians"; the evidence discusses his career but doesn't clearly support the superlative. Translation may have strengthened the claim's wording beyond what the evidence backs, though annotation noise in the original example is also possible.


21. Integrated Translation Interpretation

Central finding: translation is not uniformly beneficial. Aggregate Qwen 14B performance improves, but the effect is concentrated almost entirely in the refutes class.

Repair patterns (qualitative, illustrative): clearer ordinal, entity, and event-role contrasts becoming visible once claim and evidence are in English.

Damage patterns (qualitative, illustrative): culturally specific lexical mistranslation, numerical/ranking corruption, and predicate-level corruption.

Summary: the translated condition substantially improved overall Qwen 14B verification performance (64.02% to 72.56% accuracy), but the gain was asymmetric — refutes accuracy rose from 25.0% to 60.5% across all three languages, while supports accuracy fell overall, with Yoruba supports dropping 29.09 points specifically. Qualitative inspection of systematically selected cases is consistent with clearer contradiction cues driving repairs and culturally specific or numerical mistranslation driving regressions, but translation quality wasn't manually annotated across all 492 examples, so these are plausible mechanisms, not proven population-level causes.


22. Methodological Caveats for Translation Analysis

Case studies are illustrative, not proof that every transition shares a cause. Translation quality wasn't manually annotated for all 492 examples, so causal claims about translation errors stay qualified rather than asserted outright. Dataset noise (imperfect evidence spans, annotation issues) can interact with translation effects and isn't separated from them here. Translation used a free library route (deep-translator/Google Translate), not a paid enterprise API, documented as a methodology limitation.


23. Current Strongest Findings

Gold evidence substantially improves encoder verification, more so for XLM-R than AfriBERTa. Model scale (1.5B to 14B) produces a large gold-evidence improvement. Few-shot prompting isn't monotonically beneficial and can hurt macro-F1. Generic, off-domain prompt examples substantially hurt translated-condition performance. BM25 recall climbs steeply with depth, but rank-1 recall is weak, a ranking problem, not just a coverage problem. Retrieved evidence underperforms oracle evidence, a real oracle-to-retrieved gap. Adversarial evidence drops accuracy to roughly a third. Translation gives the strongest overall result of any condition tested. The translation gain is a contradiction-detection effect, not a uniform multilingual lift. Refutes accuracy improves after translation in all three languages. Yoruba supports accuracy collapses after translation, the main counter-example to "translation just helps." Qualitative review surfaces plausible repair and corruption mechanisms behind both effects.


24. Reporting Principle

The dissertation shouldn't present these as isolated leaderboard rows. The narrative needs to connect evidence availability, retrieval quality, model scale, prompting, translation, adversarial robustness, and language/label-specific effects, explaining not just which system scores highest, but why performance changes, where systems fail, and which components are the actual bottleneck.


Cross-Experiment Error Analysis and Final Synthesis

Purpose

A matched, example-level error analysis across six principal systems, Qwen 1.5B Claim-Only, Gold, BM25, Adversarial, and Qwen 14B Gold and Translated, to see how interventions change individual predictions, not just aggregate scores. All six share the same 492-example test set, so transitions are directly comparable.

Four questions: how much does evidence access help over claim-only, does BM25-retrieved evidence reproduce the oracle-evidence benefit, how vulnerable is verification to adversarial evidence, and how do scale and translation affect which errors get fixed versus introduced.


1. Overall performance hierarchy

| Rank | System | Accuracy |
|---|---|---:|
| 1 | Qwen 14B Translated Gold Evidence | 72.56% |
| 2 | Qwen 14B Gold Evidence | 64.02% |
| 3 | Qwen 1.5B Gold Evidence | 45.94% |
| 4 | Qwen 1.5B BM25 Evidence | 39.43% |
| 5 | Qwen 1.5B Claim Only | 39.02% |
| 6 | Qwen 1.5B Adversarial Evidence | 33.33% |

The strongest result comes from translating both claim and evidence into English before running Qwen 14B. Because translation can alter meaning as well as language, this doesn't establish that translated input is semantically equivalent to the original, or that linguistic accessibility alone drives the gain, only that this specific pipeline benefits from it. BM25 and adversarial evidence both sit well below oracle evidence, confirming that supplying evidence isn't automatically helpful.

2. Evidence access, claim-only to gold evidence (Qwen 1.5B)

Correct-to-correct 74, wrong-to-correct 152, correct-to-wrong 118, wrong-to-wrong 148. Net +34. Accuracy 39.02% to 45.94%.

Gold evidence helps in aggregate, but 118 previously-correct examples flip to wrong alongside the 152 repaired, evidence access doesn't mean the model uses evidence consistently well. A fairer summary than "evidence helps" is that gold evidence improves aggregate accuracy while producing substantial two-directional example-level churn, meaning access to relevant evidence doesn't guarantee it's used correctly.

3. Gold evidence to BM25 retrieval (Qwen 1.5B)

Correct-to-correct 171, wrong-to-correct 23, correct-to-wrong 55, wrong-to-wrong 243. Net -32. Accuracy 45.94% to 39.43%.

Only 23 examples get fixed by switching to retrieved evidence, while 55 that were right under gold evidence break. With the classifier held fixed, this is consistent with the supplied evidence, not classifier capability alone, being a real constraint: retrieval-augmented verification depends on whether retrieval actually surfaces useful evidence, not just on the downstream model.

4. BM25 support bias

Qwen 1.5B under BM25 predicts supports 80.08% of the time (394/492), refutes 14.02%, NEI 5.89%. 245 of those are wrong: 126 gold-NEI and 119 gold-refutes examples mislabelled supports, spread across Igbo (87), Yoruba (83), Hausa (75).

Likely mechanism: BM25 ranks by term overlap, not entailment, so a passage that's topically or lexically related to a claim gets retrieved even when it doesn't actually support it, and the classifier appears to over-read that topical relevance as support. This is a hypothesised failure chain, BM25 rewards term similarity, a related-but-irrelevant passage gets retrieved, the classifier reads relevance as entailment, refutes/NEI claims get called supports, not an independently verified one, but the behavioural pattern itself, the 80% supports rate and its distribution across languages, is directly observed. It's a useful distinction for the dissertation: a passage can be relevant to a claim while being useless for deciding whether the claim is true.

5. Adversarial evidence damage

Correct-to-correct 89, wrong-to-correct 75, correct-to-wrong 137, wrong-to-wrong 191. Net -62. Accuracy 45.94% to 33.33%.

The largest negative transition of any comparison here. 137 examples that were correctly verified under gold evidence flip to wrong when given plausible-but-mismatched evidence instead. For real-world RAG systems pulling from noisy corpora, irrelevant, outdated, or contradictory sources, this says retrieval quality is a reliability problem, not just a search-ranking one: poor evidence can be worse than no evidence at all, since the adversarial condition (33.33%) scores below claim-only (39.02%).

6. Effect of model scaling

Correct-to-correct 169, wrong-to-correct 146, correct-to-wrong 57, wrong-to-wrong 120. Net +89, the largest positive net transition observed. Accuracy 45.94% to 64.02%.

14B fixes 146 of 1.5B's errors while introducing only 57 new ones. Consistent with model scale mattering for low-resource verification, though parameter count isn't isolated from other checkpoint differences, representation quality, instruction-following. 120 examples stay wrong under both models, scaling helps a lot but doesn't remove systematic failure.

7. Translation as a model-access intervention

Correct-to-correct 267, wrong-to-correct 90, correct-to-wrong 48, wrong-to-wrong 87. Net +42. Accuracy 64.02% to 72.56%; macro-F1 60.02% to 72.38%.

Same model, same checkpoint, only the input language changes, so the gain can't be attributed to model size. Consistent with input-language representation mattering, though translation-induced semantic drift means linguistic accessibility isn't cleanly isolated from translation quality itself. 48 correct-to-wrong regressions confirm the gain is real but not free.

8. Translation effect by language

| Language | n | Original | Translated | Delta |
|---|---:|---:|---:|---:|
| Igbo | 163 | 60.74% | 73.62% | +12.88 |
| Hausa | 166 | 65.06% | 75.30% | +10.24 |
| Yoruba | 163 | 66.26% | 68.71% | +2.45 |

All three improve, but not by the same amount. Yoruba's much smaller gain becomes important once the label-level breakdown below shows why.

9. Translation effect by label

| Label | n | Original | Translated | Delta |
|---|---:|---:|---:|---:|
| Supports | 166 | 66.87% | 59.04% | -7.83 |
| Refutes | 152 | 25.00% | 60.53% | +35.53 |
| NEI | 174 | 95.40% | 95.98% | +0.57 |

The 8.5-point overall gain is not a uniform class-level improvement, it's almost entirely refutes-driven, with supports actually declining.

10. Translation and the refutes class specifically

| Language | Original | Translated | Delta |
|---|---:|---:|---:|
| Hausa | 25.58% | 62.79% | +37.21 |
| Igbo | 29.09% | 67.27% | +38.18 |
| Yoruba | 20.37% | 51.85% | +31.48 |

The refutes improvement holds across all three languages, consistent with translated wording making some contradiction cues, negation, ordinal contrast, event-role contrast, more accessible, though the mechanism itself isn't isolated experimentally, only inferred from the qualitative cases in section 19.

11. Representative repair cases

Same three cases as sections 19.1 to 19.3, the Bianca Ojukwu ordinal contrast, the Governor Shema event-role contrast, and the Dauda-Ibrahim entity mismatch, not repeated here to avoid duplication.

12. Translation damage, Yoruba supports

Yoruba supports accuracy: 85.45% to 56.36%, a 29.09-point drop. 21 Yoruba supports examples flip from correct to incorrect after translation.

This is the clearest counter-example to "translation just helps" in the whole study. Selected regression cases in section 20 are consistent with semantic distortion during translation, but quality wasn't manually annotated across the full Yoruba-supports subset, so this remains a plausible contributor rather than a fully quantified cause.

13. Representative damage cases

Same three cases as sections 20.1, 20.2, and 20.3, the Ondo State ranking case, the gaari/sugar case, and the breastfeeding predicate corruption case, not repeated here.

14. Generic prompt examples after translation

Translated baseline: 72.56% / 72.38%. Translated plus generic examples: 63.82% / 61.52%.

Adding off-domain demonstrations to an already-strong condition still hurts it, prompt examples aren't automatically beneficial even when they look clear and task-relevant on their face. Likely mismatch between generic demonstration style and AfriFact's actual claim distribution, culturally specific entities, evidence structure, though this wasn't independently isolated from other explanations.


Cross-System Behavioural Analysis

15. Prediction distributions

| System | Supports | Refutes | NEI |
|---|---:|---:|---:|
| Qwen 1.5B Claim Only | 19.31% | 37.20% | 43.50% |
| Qwen 1.5B Gold Evidence | 75.81% | 10.37% | 13.82% |
| Qwen 1.5B BM25 Evidence | 80.08% | 14.02% | 5.89% |
| Qwen 1.5B Adversarial Evidence | 31.50% | 13.41% | 55.08% |
| Qwen 14B Gold Evidence | 28.66% | 9.15% | 62.20% |
| Qwen 14B Translated Gold Evidence | 20.73% | 21.95% | 57.32% |

Systems don't just differ in accuracy, their prediction distributions differ sharply. Both 1.5B gold and BM25 default heavily toward supports; 14B gold defaults heavily toward NEI. Translation shifts a meaningful share of predictions toward refutes, tracking the accuracy gain on that class directly.

16. Best system by label

| Label | Best system | Accuracy |
|---|---|---:|
| NEI | Qwen 14B Translated | 95.98% |
| Refutes | Qwen 14B Translated | 60.53% |
| Supports | Qwen 1.5B Gold | 92.17% |

No single system wins on every label. Qwen 1.5B Gold's high supports accuracy has to be read alongside the fact that 75.81% of its predictions are supports (section 15), high class accuracy from a system that defaults to that class isn't the same as balanced competence.

17. Best model by language

| Language | Accuracy |
|---|---:|
| Hausa | 75.30% |
| Igbo | 73.62% |
| Yoruba | 68.71% |

Translated Qwen 14B wins on all three languages, but the Yoruba figure coexists with the supports-class collapse in section 12, the same system can lead on aggregate language accuracy while carrying a serious label-specific weakness underneath.


Universally Hard Examples

18. Universal error concentration

Across all six systems: 29 examples wrong everywhere, 12 correct everywhere. Of the 29: Yoruba 13, Hausa 10, Igbo 6 by language; refutes 25, supports 4 by label.

25 of 29 universal failures are refutes examples, persistent across every evidence condition and both model scales tested. With only 29 examples and six systems that aren't an exhaustive sample of architectures, this is diagnostic, not a claim about fact-verification systems generally.

19. Persistent hard refutes by language

Yoruba 13, Hausa 8, Igbo 4 of the 25 persistent hard refutes. This sits alongside Yoruba's supports-collapse pattern from a different subset of examples, worth noting as a pattern, not treated as one shared cause without more evidence.


Manual Hard-Refute Taxonomy

20. Status

The 25 persistent hard refutes were manually reviewed and assigned a primary error category each. Exploratory and diagnostic, not a statistically representative taxonomy of dataset-wide errors, just what's recurring in the hardest cases.

21. Primary error categories

| Type | Count | Percent |
|---|---:|---:|
| Evidence insufficient | 6 | 24% |
| Entity-attribute mismatch | 5 | 20% |
| Negation | 3 | 12% |
| Role-relation mismatch | 3 | 12% |
| Temporal mismatch | 2 | 8% |
| Boundary condition | 2 | 8% |
| Causal mismatch | 1 | 4% |
| Implicit contradiction | 1 | 4% |
| Possible annotation issue | 1 | 4% |
| Numerical mismatch | 1 | 4% |

Insufficient evidence and entity-attribute mismatch dominate. Errors here are heterogeneous, not reducible to one failure mode like negation alone.

22. Contradiction explicitness

Explicit 44% (11), implicit 28% (7), ambiguous 28% (7). More than half of persistent hard refutes require more than surface-level contradiction detection, entity, temporal, causal, and boundary reasoning, not just lexical comparison.

23. Annotation suspicion

No concern 56% (14), issue suspected 28% (7), uncertain 16% (4). 44% of this small, hand-picked subset carries some annotation doubt, not a claim about dataset-wide error rate, just a signal that some of the hardest cases may be hard partly because the evidence-label relationship itself is debatable, not purely because the model failed.

24. Hard-error taxonomy by language

Yoruba has the broadest spread of error types and the most persistent hard refutes overall (entity-attribute times 3, evidence insufficient times 3, negation times 2, plus one each of boundary, numerical, annotation, role-relation, temporal). Igbo's four cases split across role-relation, negation, and insufficient-evidence. Hausa's eight span entity, causal, temporal, boundary, insufficient-evidence, and implicit-contradiction. Given how small these subsets are, this is a qualitative tendency, not a language-wide distribution claim.


Integrated Interpretation

25. Evidence quality matters more than evidence presence

Claim-only 39.02%, gold evidence 45.94%, BM25 evidence 39.43%, adversarial evidence 33.33%. BM25 lands close to claim-only; adversarial evidence lands below it. The value of supplied evidence depends on its relevance and reliability, not just its presence, a central finding of this study.

26. Retrieval is a major bottleneck

BM25 only repairs 23 of gold evidence's errors while breaking 55, for a net loss of 32, alongside an 80% supports-prediction rate and 245 identified support-bias failures. The lexical-retrieval setup evaluated here looks insufficient for contradiction-aware matching, entity-relation reasoning, or recognising that relevant does not mean supporting. Dense/hybrid retrieval, cross-encoder reranking, and confidence-based abstention are the obvious next steps.

27. Scaling and translation are distinct effects

Scaling, 1.5B to 14B: +89 net correct. Translation, 14B, original to English: +42 net correct. Both matter, but the comparisons don't establish a combined causal mechanism, model capacity and input-language representation plausibly affect different aspects of evidence use, but that's a hypothesis for further testing, not something demonstrated here.

28. Translation is powerful but asymmetric

Overall +8.54 points; Igbo +12.88, Hausa +10.24, Yoruba +2.45; refutes +35.53, supports -7.83, Yoruba supports -29.09.

A large aggregate benefit that comes with real, non-trivial regressions concentrated in one class and one language. This motivates evaluating language-aware translation quality controls rather than treating translation as a uniformly safe intervention.

29. Persistent errors concentrate in contradiction cases

25 of 29 universal failures are refutes; original refutes accuracy was 25%, translation lifted it to 60.53%, but substantial errors remain; 56% of persistent hard contradictions are implicit or ambiguous.

Consistent with contradiction cases needing more than lexical pattern-matching, but the present experiments don't establish contradiction recognition as the single unresolved challenge for low-resource fact verification generally, only that it's the dominant one in this dataset and pipeline.

30. Dataset quality may contribute to persistent errors

44% of the 25 persistent hard refutes carried some annotation concern under manual review. Where supplied evidence doesn't clearly refute a claim, an NEI prediction can be semantically defensible even when scored wrong against the dataset label, model performance here should be read alongside possible evidence-quality limitations, not purely as a measure of reasoning ability.


Final Research Narrative

Qwen 1.5B claim-only accuracy is limited on its own. Gold evidence improves 1.5B in aggregate but with real bidirectional churn at the example level. BM25 doesn't reproduce the gold-evidence gain and shows a strong supports bias. Adversarial evidence performs below claim-only. Scaling 1.5B to 14B under gold evidence gives the largest single matched gain, +89 net. Translating inputs into English gives the highest overall accuracy of any tested system, 72.56%. The translation gain is driven almost entirely by refutes-class improvement. Translation also causes real regressions, concentrated in Yoruba supports, -29.09 points; qualitative cases are consistent with several types of semantic corruption during translation. Generic prompt demonstrations hurt performance even under the strongest, translated, condition. 25 of 29 universally-hard examples are refutes. 56% of manually reviewed persistent hard refutes are implicit or ambiguous contradictions. A meaningful share of that same persistent-error subset carries annotation concerns. Overall, end-to-end performance is shaped by retrieval quality, input-language representation, model scale, contradiction-specific reasoning demands, and evidence-label quality together, not any single one of them.

Core contribution: low-resource retrieval-augmented misinformation detection needs to be evaluated across at least four interacting components, linguistic representation, retrieval quality, model capacity, and evidence-label validity, because weaknesses in any one of them show up as substantial end-to-end performance changes. The best-performing configuration in this study, translated input, 14B model, still fails persistently on refutes examples, and some of those persistent failures are themselves evidence-quality questions rather than pure model failures. 