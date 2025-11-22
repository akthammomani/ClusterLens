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

Please note that this open-source library is currently in its beta phase. If you encounter any issues or have suggestions, we encourage you to share them. We are committed to addressing feedback promptly. Your contributions and ideas are greatly appreciated!.

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
2. A **one-vs-rest classifier per cluster** (RandomForest by default, optional LightGBM / XGBoost).  
3. **SHAP values** computed on a held-out evaluation set for each cluster.  
4. A set of utilities that reuse this shared state to give you interpretations and exports:

   - Global & per-cluster **classification metrics** - `get_cluster_classification_stats()`  
   - Per-cluster **feature rankings** - `get_top_shap_features(...)`, `plot_cluster_shap(...)`  
   - **Contrastive importance** between two clusters - `contrastive_importance(...)`  
   - **Distribution plots** across clusters - `compare_feature_across_clusters(...)`  
   - Markdown-ready **cluster narratives** - `generate_cluster_narratives(...)`  
   - A **cluster summary table** and export helpers:

It is built to be:

- **Model-agnostic on the clustering side**: ClusterLens never clusters; it interprets the labels you already have.
- **Numerically honest**: Combines SHAP with effect sizes (`Cohen's d`, standardized median gaps, Cramér’s V, lifts).
- **Report-friendly**: Outputs narratives and tables you can drop directly into notebooks, dashboards, or slide decks.

---

## Installation

Once published to PyPI:

```bash
pip install clusterlens

