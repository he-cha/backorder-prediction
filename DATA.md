# Data access and experiment scope

The original course notebooks use `Kaggle_Training_Dataset_v2.csv` and `Kaggle_Test_Dataset_v2.csv`. Neither raw CSV is included in the uploaded materials. Their exact upstream publisher, license, and business provenance remain unverified.

The new benchmark uses the supplied `sample-data.pkl`: 45,172 rows, 21 predictors, and a 25% backorder rate. It has already been imputed and undersampled by the original preprocessing. A new split cannot undo that preprocessing or make these reused observations a fresh independent dataset. Identical predictor rows are grouped to prevent their crossing partitions, but SKU and timestamps are absent, preventing product/time leakage checks.

To run the benchmark, obtain the trusted supplied sample locally and run `python scripts/improve_models.py --sample /path/to/sample-data.pkl`. Do not deserialize unknown pickle/joblib files. Data and model binaries are excluded from GitHub.

To evaluate at the original backorder frequency and repair the preprocessing fully, recover the raw CSVs and reserve a fresh untouched holdout. The historical test has already been reviewed and should not become a tuning set.
