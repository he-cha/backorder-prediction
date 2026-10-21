# Model improvements: reproducible sample benchmark

The experiment ran four candidate models on the uploaded 45,172-row sample. Exact duplicate feature vectors were grouped before a deterministic train/validation/test split. The split contained 26,993 training rows, 8,895 validation rows, and 9,284 test rows.

The reference follows the original SelectKBest + class-weighted Random Forest structure but is retrained on this experiment's partition without LOF. It is not a re-evaluation of the old saved model, which had already seen many of these observations.

Model selection used validation average precision. Threshold selection used validation F1. This objective is demonstrative; a validated business cost function should replace it for deployment. The selected model is not refit after threshold tuning, preserving the score distribution used to choose the threshold.

Training-fold imputation is used for new transformations, but the original sample was already imputed before upload. Unknown SKU overlap, missing timestamps, sample prevalence, and prior reuse limit independence and generalization. This is one split with no uncertainty interval, so small differences are not statistical evidence of superiority.

See README for the results. A lower default-threshold false-positive count trades off recall; it is not a free improvement. No claim is made that gradient boosting, engineered features, or threshold tuning always improves performance.

Next: recover raw data; drop invalid rows and missing targets; split before imputation; evaluate at natural prevalence; include temporal/product grouping if available; choose a capacity- or cost-based threshold on validation; and test on a fresh untouched holdout.
