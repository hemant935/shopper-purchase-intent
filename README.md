# Machine Learning Assignment 2

Predicting whether an e-commerce session ends in a purchase, using the UCI Online Shoppers Purchasing Intention dataset.

---

## a. Problem statement

E-commerce platforms observe a visitor’s session in real time: which page types they open, how long they stay, how often they bounce, where the traffic came from, and whether the visit falls near a seasonal sale. The task is **binary classification** — predict `Revenue` (purchase vs no purchase) for a session so the site can decide whether to retarget, offer help, or personalise the remaining pages.

The positive class is uncommon (~15.5% of sessions convert). Accuracy alone is misleading: a model that always predicts “no purchase” is already about 84.5% accurate. The comparison below therefore treats **MCC**, **AUC**, and **recall** as the metrics that decide which model is actually useful.

---

## b. Dataset description

| Item | Detail |
| --- | --- |
| Source | [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (id 468) |
| Rows | 12,330 sessions |
| Features | 18 (10 numerical, 8 categorical / integer-coded) |
| Label | `Revenue` — `True` if the session ended in a purchase |
| Class balance | 10,422 no-purchase (84.53%) / 1,908 purchase (15.47%) |
| Missing values | None |

Feature groups:

- **Page-visit behaviour:** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`
- **Engagement quality:** `BounceRates`, `ExitRates`, `PageValues`
- **Context:** `SpecialDay`, `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`

**Preprocessing used for every model:** stratified 80/20 train/test split (`random_state=42`); `Weekend` / `Revenue` mapped to 0/1; `StandardScaler` on numeric columns; one-hot encoding of `Month` and `VisitorType`. Integer-coded fields (`OperatingSystems`, `Browser`, `Region`, `TrafficType`) are left numeric. Logistic Regression, Decision Tree, and Random Forest use `class_weight='balanced'`.

Hold-out file used in the app: [`test_data.csv`](test_data.csv) (2,466 sessions).

---

## c. GitHub repository link

https://github.com/hemant935/shopper-purchase-intent

---

## d. Models used

Five classifiers from the assignment brief, all trained on the same processed hold-out split. Metrics are reported on the **2,466-row test set** (not on training data).

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8504 | 0.8962 | 0.5116 | 0.7487 | 0.6079 | 0.5338 |
| Decision Tree | 0.8508 | 0.7054 | 0.5192 | 0.4948 | 0.5067 | 0.4190 |
| kNN | 0.8674 | 0.7829 | 0.6170 | 0.3796 | 0.4700 | 0.4145 |
| Naive Bayes | 0.6736 | 0.7939 | 0.2941 | 0.7906 | 0.4287 | 0.3249 |
| Random Forest (Ensemble) | 0.8893 | 0.9219 | 0.6259 | 0.7094 | 0.6650 | 0.6007 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong linear baseline. AUC 0.90 shows it ranks sessions well. `class_weight='balanced'` lifts recall to 0.75 (it catches most buyers) at the cost of precision 0.51 — many window-shoppers are flagged as likely buyers. MCC 0.53 is the second-best of the five. |
| Decision Tree | Accuracy looks almost identical to logistic regression (~0.85), but AUC falls to 0.71 and MCC to 0.42. An unpruned tree overfits this mixed tabular data; precision and recall both sit near 0.50, so it is not a reliable purchase detector. |
| kNN | Highest accuracy after Random Forest (0.87), but that score is inflated by the majority class. Recall is only 0.38: kNN misses most actual purchases. Distance-based voting on an imbalanced set, with no class-weight option, explains the gap. MCC 0.41 is close to the decision tree. |
| Naive Bayes | Worst accuracy (0.67) and worst MCC (0.32). Gaussian NB assumes feature-conditional Gaussians; after one-hot encoding and with a highly skewed `PageValues` column, it over-predicts purchases (recall 0.79, precision 0.29). Useful as a contrast, not as a production choice. |
| Random Forest (Ensemble) | Best model on Accuracy, AUC, F1, and MCC. Recall 0.71 and precision 0.63 are the most balanced pair. Feature importances confirm `PageValues` as the dominant signal (~37.5%), followed by `ExitRates` and product-page engagement. Bagging plus `class_weight='balanced'` is what the decision tree was missing. |
| **Overall winner** | **Random Forest.** It is the only model that is strong on both ranking (AUC 0.92) and the imbalance-aware metrics (MCC 0.60, F1 0.67). Logistic regression is the runner-up if a simpler, more interpretable score is needed. |

---

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload [`test_data.csv`](test_data.csv) in the sidebar and select a classifier. Retraining (optional):

```bash
python train.py
```

or run [`notebooks/training.ipynb`](notebooks/training.ipynb) — this is the notebook to execute on **BITS Virtual Lab** for the required screenshot.

## Streamlit app

Live app: *(add the Streamlit Community Cloud URL after deploy)*

The app loads the five pre-trained pipelines from `model/`. It does not retrain. Required UI:

- CSV upload (test data only on the free tier)
- Model dropdown
- Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
