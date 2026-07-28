# Cross-Experiment Error Analysis Synthesis

## Overall Performance

The strongest evaluated system was `qwen_14b_translated` with accuracy 0.7358.

## Persistent Hard Refutes

A total of 25 refutation examples were misclassified by every analysed system.

- Explicit contradictions: 11/25 (44.0%)
- Implicit contradictions: 7/25 (28.0%)
- Ambiguous contradictions: 7/25 (28.0%)
- Suspected annotation issues: 7/25 (28.0%)
- Primarily evidence-insufficient cases: 6/25 (24.0%)

## Interpretation

Persistent errors should not be interpreted solely as model failures. The manually coded taxonomy indicates a mixture of genuine contradiction-reasoning failures, implicit semantic mismatches, and potentially ambiguous or evidence-insufficient benchmark instances.

Translation improved aggregate performance but produced asymmetric effects across labels and languages. Earlier transition analysis showed particularly strong repair of refutation errors, alongside regressions for some support examples.

Retrieval quality also materially affected classification. BM25 evidence introduced a strong support prediction bias, while adversarial evidence caused substantial correctness regressions, demonstrating that evidence availability alone is insufficient when relevance or reliability is poor.

Model scaling from Qwen 1.5B to Qwen 14B repaired many previously incorrect examples, but did not eliminate persistent contradiction failures.