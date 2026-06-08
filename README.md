# Genomic Prediction of AMR in *Klebsiella pneumoniae*: South Asian Regional Analysis

**Md Rafin Rahman**¹, **Jafren Iqbal Rose**², **Md Rofiqur Rahman**¹

¹ Institute for Developing Science and Health Initiatives (ideSHi), Dhaka, Bangladesh  
² Shaheed Monsur Ali Medical College & Hospital, Dhaka, Bangladesh  

**Corresponding author:** Md Rafin Rahman, mrahman@ideshi.org

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20590089.svg)](https://doi.org/10.5281/zenodo.20590089)

---

## Overview

This repository contains the complete analysis pipeline for the manuscript:

> Rahman MR, Rose JI, Rahman MR. *Genomic Prediction of Carbapenem and Third-Generation Cephalosporin Resistance in Klebsiella pneumoniae: A Machine Learning Analysis of 145,653 Global Isolates with South Asian Regional Stratification.* Submitted for publication, 2026.

The study trains and evaluates four machine learning algorithms (logistic regression, random forest, XGBoost, LightGBM) on 145,653 *K. pneumoniae* isolates from the NCBI Pathogen Detection database to predict carbapenem and 3GC resistance phenotypes from AMRFinderPlus gene content. The primary novel contribution is a SHAP-based comparison of resistance gene importance between South Asian isolates (n = 5,919, including 2,239 from Bangladesh) and global isolates, revealing that blaNDM-5, blaOXA-232, and blaOXA-181 have disproportionately higher predictive weight in South Asia compared to the blaKPC genes that dominate globally.

---

## Repository Structure

```
├── 01_preprocess.py       # Data loading, filtering, feature engineering, outcome labelling
├── 02_model.py            # Baseline model training, cross-validation, regional evaluation
├── data/
│   └── gene_list.json     # List of 342 selected AMR gene features
├── outputs/
│   ├── cv_results.csv            # 5-fold cross-validation metrics (all models, both targets)
│   ├── regional_results.csv      # Regional generalisation: global train → South Asia test
│   ├── shap_carbapenem_resistant.csv  # SHAP values: carbapenem resistance, SA vs global
│   └── shap_threegc_resistant.csv     # SHAP values: 3GC resistance, SA vs global
└── figures/
    ├── fig1_model_performance.png     # AUC comparison across CV, global holdout, South Asia
    ├── fig2_shap_comparison.png       # Feature importance: South Asia vs Global
    ├── fig3_country_rates.png         # Resistance rates by country
    └── fig4_generalisation_gap.png    # Cross-regional generalisation gap + SA-enriched genes
```

---

## Data Access

**The dataset is not included in this repository.**

The analysis uses isolate records from the **NCBI Pathogen Detection Isolates Browser**, which is publicly accessible without registration:

- **URL:** https://www.ncbi.nlm.nih.gov/pathogens/isolates/#Klebsiella_pneumoniae
- **Download:** Click the Download button on the isolates table and select Isolates (TSV format)
- **File used:** `isolates.tsv` — 293,033 rows, downloaded June 2026

**Code archive:** This repository is permanently archived on Zenodo at [https://doi.org/10.5281/zenodo.20590089](https://doi.org/10.5281/zenodo.20590089)

After downloading, place the file at `data/isolates.tsv` before running the pipeline.

The companion exceptions file (`isolate_exceptions.tsv`) is also downloadable from the same page and should be placed at `data/isolate_exceptions.tsv`.

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn scipy
```

Python 3.10 or above. No GPU required.

---

## How to Reproduce

```bash
# Step 1: Preprocess the raw NCBI data
python 01_preprocess.py

# Step 2: Train models and evaluate globally and by region
python 02_model.py
```

Total runtime on a standard laptop (8 CPU cores) is approximately 20 to 35 minutes, depending on the hardware.

---

## Key Results

| Model | CV AUC (Carbapenem) | SA AUC (Carbapenem) | CV AUC (3GC) | SA AUC (3GC) |
|---|---|---|---|---|
| Logistic Regression | 1.000 | 1.000 | 0.992 | 0.998 |
| Random Forest | 0.987 | 0.989 | 0.966 | 0.976 |
| XGBoost | 0.998 | 1.000 | 0.993 | 0.999 |
| LightGBM | 0.998 | 1.000 | 0.993 | 0.999 |

**Top South Asia-enriched resistance genes (carbapenem, by SHAP ratio):**

| Gene | SA SHAP weight | Global SHAP weight | SA/Global ratio |
|---|---|---|---|
| blaOXA-232 | 1.231 | 0.294 | 4.19x |
| blaOXA-181 | 0.967 | 0.253 | 3.82x |
| blaNDM-5 | 1.332 | 0.439 | 3.03x |

**Bangladesh-specific:** Carbapenem resistance rate 58.3% vs USA 38.2%; 3GC resistance rate 83.2% vs USA 49.1%.

---

## Citation

```
Rahman MR, Rose JI, Rahman MR. Genomic Prediction of Carbapenem and
Third-Generation Cephalosporin Resistance in Klebsiella pneumoniae: A Machine Learning
Analysis of 145,653 Global Isolates with South Asian Regional Stratification.
Submitted for publication, 2026. Contact: mrahman@ideshi.org
Code DOI: https://doi.org/10.5281/zenodo.20590089
```

Please also cite the data source:

```
NCBI Pathogen Detection Isolates Browser. National Center for Biotechnology Information.
Available at: https://www.ncbi.nlm.nih.gov/pathogens/isolates/#Klebsiella_pneumoniae
Accessed June 2026.
```

---

## Licence

Code released under the **MIT Licence**.

---

*Institute for Developing Science and Health Initiatives (ideSHi) | Dhaka, Bangladesh | June 2026*
