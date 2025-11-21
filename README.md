<p align="center">
  <!-- Language / Core -->
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-plotting-11557C?logo=python&logoColor=white">
  <img alt="Seaborn" src="https://img.shields.io/badge/Seaborn-statistical%20plots-0EA5E9">
  <img alt="SciPy" src="https://img.shields.io/badge/SciPy-Mann--Whitney%20U%20%7C%20Chi--square-0B61A4?logo=scipy&logoColor=white">
  
  <!-- Modeling / ML stack -->
  <img alt="scikit-learn RF" src="https://img.shields.io/badge/scikit--learn-RandomForest%20OVR-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Encoders" src="https://img.shields.io/badge/Encoders-OneHot%20%7C%20LOO%20%7C%20CatBoost-10B981">
  <img alt="Metrics" src="https://img.shields.io/badge/Metrics-Accuracy%20%7C%20Precision%20%7C%20Recall%20%7C%20F1%20%7C%20ROC%20AUC-6366F1">
  <img alt="SHAP" src="https://img.shields.io/badge/SHAP-global%20%26%20local%20importance-8B5CF6">
  <img alt="LightGBM" src="https://img.shields.io/badge/Optional-LightGBM-00B300?logo=lightgbm&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/Optional-XGBoost-EB4C2B">
  
  <!-- Purpose / Features -->
  <img alt="Task" src="https://img.shields.io/badge/Task-Cluster%20interpretability-312E81">
  <img alt="Contrastive analysis" src="https://img.shields.io/badge/Analysis-Contrastive%20importance%20(A%20vs%20B)-F59E0B">
  <img alt="Distributions" src="https://img.shields.io/badge/Plots-Histograms%20%7C%20Stacked%20bars-4B5563">
  <img alt="Narratives" src="https://img.shields.io/badge/Output-Cluster%20narratives%20%26%20summaries-EC4899">
  <img alt="Exports" src="https://img.shields.io/badge/Exports-CSV%20summary%20%7C%20PNG%20SHAP%20plots-6B7280">
  
  <!-- Meta -->
  <img alt="License" src="https://img.shields.io/badge/License-MIT-000000">
  <img alt="Status" src="https://img.shields.io/badge/Status-Alpha%20library-brightgreen">
</p>


<p align="center">
  <!-- Replace the src URL below with your actual ClusterLens logo asset -->
  <img src="https://github.com/user-attachments/assets/31214062-e972-43e1-a6a6-4116821002cb"  
       alt="ClusterLens logo" width="320" height="320" />
</p>

Please note that the project is currently in its beta phase. If you encounter any issues or have suggestions, we encourage you to share them. We are committed to addressing feedback promptly. Your contributions and ideas are greatly appreciated!.

# **ClusterLens**

**ClusterLens** is an interpretability engine for **clustered / segmented data**.

You already have clusters - customer segments, user personas, product tiers, risk bands.  
ClusterLens answers the harder questions:

- What actually *drives* each cluster?
- How is Cluster 1 different from Cluster 3 in a statistically meaningful way?
- Which features make Cluster A "high value" or "high risk" compared to others?
- How can I turn a big table into cluster narratives that non-ML stakeholders can read?

ClusterLens sits on top of **any clustering method** (k-means, GMM, HDBSCAN, rule-based labels, etc.).  
All it requires is a `DataFrame` with a column that holds the cluster labels.


## Key ideas

ClusterLens wraps a **train-once, reuse-everywhere** pipeline:

1. One **shared train/test split**, stratified by cluster.  
   - Done inside `ClusterAnalyzer.fit(...)`.  
   - You pass a `DataFrame` with a cluster column; ClusterLens optionally takes a **stratified sample** (by cluster), then runs a **single `train_test_split`** using `cluster_col` as the stratification key.  
   - The same split is reused across all one-vs-rest models, so every downstream metric (SHAP, classification stats, contrastive importance) is computed on a **consistent hold-out**.

2. A **one-vs-rest classifier per cluster** (RandomForest by default, optional LightGBM / XGBoost).  
   - Also part of `fit(...)` via the internal `_make_model(...)` factory.  
   - For each unique cluster value, ClusterLens:
     - Builds a binary target: *this cluster* vs *all other rows*.  
     - Trains a classifier with **built-in class-imbalance handling**:
       - `"rf"` → `RandomForestClassifier(class_weight="balanced", n_jobs=-1)`  
       - `"lgbm"` → `LGBMClassifier` with `class_weight="balanced"` (if LightGBM is installed)  
       - `"xgb"` → `XGBClassifier` with **auto `scale_pos_weight`** based on positive/negative counts.  
   - All models share the same encoded feature matrix and split (`X_train`, `X_test`, `y_train`, `y_test`), so comparisons across clusters are meaningful.

3. **SHAP values** computed on a held-out evaluation set for each cluster.  
   - Still inside `fit(...)`, after each model is trained:
     - ClusterLens builds a SHAP explainer (`shap.Explainer` with a fallback to `shap.TreeExplainer`).  
     - It evaluates SHAP on **held-out data** (`X_test`) and optionally subsamples to `eval_max_n` rows per cluster for speed.  
     - All SHAP arrays and evaluation matrices are cached in `shap_cache[cluster_id]`, so later calls don't recompute SHAP:
       - `plot_cluster_shap(...)`
       - `get_top_shap_features(...)`
       - `contrastive_importance(...)`  

4. A set of utilities that reuse this shared state to give you interpretations and exports:

   - Global & per-cluster **classification metrics** - `get_cluster_classification_stats()`  
     - Treats each cluster model as a binary classifier on the shared test set (`y_test`).  
     - Returns per-cluster `Accuracy`, `Precision`, `Recall`, `F1`, `ROC_AUC`, and confusion-matrix counts (`TN`, `FP`, `FN`, `TP`) so you can see which clusters are easy vs hard to separate.

   - Per-cluster **feature rankings** - `get_top_shap_features(...)`, `plot_cluster_shap(...)`  
     - Aggregates SHAP values back to **original features** (numeric + categorical) via `_aggregate_importance(...)`.  
     - `get_top_shap_features(...)` returns a tidy DataFrame of `Cluster`, `Feature`, and `Abs_SHAP` for programmatic use.  
     - `plot_cluster_shap(...)` turns the same rankings into bar charts with **human-readable labels** (e.g., `feat: 3.1 (+0.7 vs global)` for numeric or `feat=cat (65%, global 20%)` for categorical).

   - **Contrastive importance** between two clusters - `contrastive_importance(...)`  
     - For any pair `(cluster_a, cluster_b)`, computes a **blend** of:
       - Normalized SHAP importance for A and B, and  
       - Effect-size statistics on raw data: standardized median gaps (in IQR units), `Cohen’s d` for numeric, and lift + `Cramér’s V` for categorical.  
     - You choose the mode:
       - `"shap"` (SHAP only),
       - `"effect"` (statistical contrasts only),
       - `"hybrid"` (weighted combination, default).  
     - Returns a ranked table of features that most distinguish the two clusters.

   - **Distribution plots** across clusters - `compare_feature_across_clusters(...)`  
     - For a **numeric feature**: overlays per-cluster histograms of raw counts, with optional automatic `log1p` on skewed distributions (`auto_log_skew`).  
     - For a **categorical feature**: draws stacked bar charts of counts by cluster.  
     - If you don’t pass a feature, it builds a **faceted grid** of histograms for all numeric features, which is ideal for quick EDA.

   - Markdown-ready **cluster narratives** - `generate_cluster_narratives(...)`  
     - Uses numeric and categorical effect sizes plus nearest-cluster comparisons to build **text blocks** like:  
       - "Cluster 0 — N=120 (24%): high on `spend` (+1.8 IQR), often `segment=A` (65% vs 20% global), versus Cluster 1 it has higher `tenure` (+1.1 IQR)."  
     - You can choose:
       - `output="markdown"` – cluster sections ready to paste into notebooks or reports, or  
       - `output="dict"` – structured Python dicts for custom rendering.

   - A **cluster summary table** and export helpers:
     - `get_cluster_summary(...)` builds a one-row-per-cluster DataFrame with:
       - `N`, `Pct` (cluster size and share of the dataset),  
       - Per-feature descriptive summaries (value + delta vs global),  
       - `Contrast_Numeric` / `Contrast_Categorical` columns summarizing key differences vs the nearest cluster.  
     - `export_summary(path)` writes that summary to CSV in one line.  
     - `save_shap_figs(folder_path)` exports the SHAP bar plots generated by `plot_cluster_shap(...)` to PNGs (one file per cluster) for slide decks or documentation.

It is built to be:

- **Model-agnostic on the clustering side**: ClusterLens never clusters; it interprets the labels you already have.
- **Numerically honest**: Combines SHAP with effect sizes (`Cohen's d`, standardized median gaps, Cramér’s V, lifts).
- **Report-friendly**: Outputs narratives and tables you can drop directly into notebooks, dashboards, or slide decks.

---

## Installation

Once published to PyPI:

```bash
pip install clusterlens

