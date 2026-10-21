# Findings and evaluation audit

## What the evidence supports

- Random Forest had the highest backorder F1 among the three workflows on the 9,035-row sampled holdout (0.831).
- The independent test contained 242,076 recorded rows and 2,688 backorders. The confusion matrix was TN=220,209, FP=19,179, FN=616, TP=2,072.
- Test precision=9.75%, recall=77.08%, F1=0.173 and accuracy=91.82%. These are recalculated from recorded outputs, not a new test run.
- Removing test outliers left 229,972 rows. Precision=9.81%, recall=77.54%, F1=0.174. This small difference is descriptive, not proof of robustness, because the evaluated population changed.

## Issues to fix before another performance claim

1. **Final training subset:** Part 2 saves `sample.pkl` from the LOF-filtered development split (34,330 rows), not the full 45,172-row smart sample. Part 3 retrains on that saved subset despite its original full-sample wording. Do not claim the final model was trained on all 1.7 million records or all sampled rows.
2. **Missing target and trailing rows:** The historical code imputes the outcome and appears to retain a trailing summary/non-product record. Inspect the CSVs and drop invalid product rows and unknown outcomes before splitting.
3. **Preprocessing isolation:** Part 1 imputes before splitting; Part 3 computes new test-set medians and modes. Fit predictor imputers on training folds only, then transform validation/test with those fitted values. Inspect supplier-performance values of -99 as possible missing-value sentinels.
4. **Cross-validation:** Outlier detection is fitted before CV. Fold validation therefore influences outlier selection. Move learned preprocessing, sampling, and filtering into training-fold-only workflows.
5. **Test filtering:** Part 3 refits LOF on the test set with `fit_predict`, emits a duplicate-value warning, and removes rows. This is a transductive sensitivity analysis, not a frozen deployment pipeline. Keep full-population testing primary.
6. **Sampling and thresholds:** The sample changes positive prevalence from 0.67% to 25%. Class weighting adds another adjustment. Evaluate probability calibration, precision-recall curves, PR-AUC, and capacity-based thresholds on a representative validation set.
7. **Reproducibility:** The initial shuffle and some stochastic steps lack seeds; exact original library versions and raw CSVs are unavailable here. Full numerical reproduction is unverified.
8. **Interpretability and operations:** No stored feature importance, cost analysis, calibration assessment, or temporal validation supports claims about drivers, savings, or production readiness.

## Next experiment

Create a clean train/validation split before fitting imputers. Preserve natural prevalence in validation, sample only training data, compare a no-outlier-filter Random Forest against the existing approach, and select an alert threshold using explicit business costs or review capacity. Evaluate the frozen choice once on a fresh holdout. If timestamps or repeated SKUs exist, use appropriate time/group separation.

## Resume wording supported by these results

“Compared three machine learning workflows for product backorder prediction using a 1.69-million-row source dataset; selected a Random Forest achieving 0.83 F1 on a sampled holdout and evaluated precision–recall tradeoffs on 242,076 independent test records.”

The 0.83 score must remain labeled as sampled-holdout performance. Do not present it as the independent-test score.


## New experiment

See [IMPROVEMENTS.md](IMPROVEMENTS.md) and [sample_benchmark.json](sample_benchmark.json) for the newly executed sample benchmark. It is separate from the historical independent test above.
