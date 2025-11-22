#===========================================================================================================================================================================#
#                                                                         ClusterLens - Core Module                                                                         #
#===========================================================================================================================================================================#
# Author      : Aktham Momani                                                                                                                                               #
# Created     : 2025-11-22                                                                                                                                                  #
# Version     : V1.0.0                                                                                                                                                      #
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Purpose     : Exposes ClusterAnalyzer: a train-once, reuse-everywhere interpretability engine for clustered / segmented data. Given a DataFrame with a cluster label, it  #
#               trains one-vs-rest classifiers, computes SHAP, contrastive statistics, andproduces cluster narratives and summaries.                                        #
#                                                                                                                                                                           #
# Change Log  :                                                                                                                                                             #
#  - V1.0.0 (2025-11-22): Initial release.                                                                                                                                  #
#  - V1.0.1 (planned)   : Minor bug fixes, UX improvements, polish the UI, performance, and stability.                                                                      #
#===========================================================================================================================================================================#

from __future__ import annotations

import builtins
max = builtins.max
min = builtins.min
abs = builtins.abs
sum = builtins.sum
round = builtins.round
pow = builtins.pow
try:
    del col  
except Exception:
    pass

import os
import math
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import mannwhitneyu, chi2_contingency, gaussian_kde
import shap

# compact encoders
try:
    from category_encoders import LeaveOneOutEncoder, CatBoostEncoder
except Exception:
    LeaveOneOutEncoder = CatBoostEncoder = None

# OPTIONAL: gradient-boosted models (these are optional extras)
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# ---------------------------- Utilities ---------------------------- 

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _std_delta(cluster_median: float, global_median: float, global_iqr: float) -> float:
    if global_iqr is None or not np.isfinite(global_iqr) or global_iqr == 0:
        return np.nan
    return (cluster_median - global_median) / float(global_iqr)

def _cohens_d(a, b):
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2: return np.nan
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    n1, n2 = len(a), len(b)
    denom = (n1 + n2 - 2)
    if denom <= 0: return np.nan
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / denom)
    return (m1 - m2) / sp if (sp and np.isfinite(sp)) else np.nan

def _cramers_v(tbl: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(tbl, correction=False)
    n = tbl.values.sum()
    if n == 0: return np.nan
    r, k = tbl.shape
    denom = min(k - 1, r - 1)
    return np.sqrt((chi2 / n) / denom) if denom > 0 else np.nan

def _nearest_cluster_centroid(df: pd.DataFrame, cluster_col: str, num_features: List[str]) -> Dict:
    if not num_features:
        return {cl: None for cl in df[cluster_col].unique()}
    centroids = (
        df.groupby(cluster_col)[num_features]
          .median(numeric_only=True)
          .astype(float)
    )
    ids = list(centroids.index)
    nearest = {}
    for cl in ids:
        v = centroids.loc[cl].values
        best, best_id = np.inf, None
        for cl2 in ids:
            if cl2 == cl: continue
            d = np.linalg.norm(v - centroids.loc[cl2].values)
            if d < best:
                best, best_id = d, cl2
        nearest[cl] = best_id
    return nearest

def _stratified_sample(
    df: pd.DataFrame,
    cluster_col: str,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: int = 1981
) -> pd.DataFrame:
    if n is not None:
        counts = df[cluster_col].value_counts()
        props = counts / counts.sum()
        alloc = (props * n).round().astype(int)
        if n >= len(alloc): alloc = alloc.clip(lower=1)
        diff = n - int(alloc.sum())
        if diff != 0:
            rema = (props * n - (props * n).round()).abs().sort_values(ascending=False)
            for cl in rema.index[:abs(diff)]:
                if diff > 0:
                    alloc.loc[cl] += 1
                elif diff < 0 and alloc.loc[cl] > 1:
                    alloc.loc[cl] -= 1
        parts = []
        for cl, take in alloc.items():
            g = df[df[cluster_col] == cl]
            take = min(int(take), len(g))
            parts.append(g.sample(n=take, random_state=random_state, replace=False))
        out = pd.concat(parts, axis=0)
        return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if frac is not None:
        return (df.groupby(cluster_col, group_keys=False)
                  .apply(lambda g: g.sample(frac=min(1.0, frac), random_state=random_state))
                  .reset_index(drop=True))
    return df.reset_index(drop=True)


# ---------------------------- Main class ---------------------------- 
class ClusterAnalyzer:
    """
    Train ONCE, reuse everywhere (OVR classifier + SHAP cache).

    - Auto-detects numeric/categorical features if not provided.
    - Cat path optional (We can have zero categorical features).
    - Train on full df (train/split 80/20 split) or a stratified sample.
    - model_type: "rf" (default), with optional "lgbm", "xgb".
      • RF & LGBM use class_weight="balanced" by default.
      • XGB auto-sets scale_pos_weight per cluster (neg/pos) 
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        cluster_col: str = "Cluster",
        random_state: int = 1981,
        encoder: str = "onehot",             # "onehot" | "loo" | "catboost"
        model_type: str = "rf",              # "rf" | "lgbm" | "xgb"
        model_params: Optional[dict] = None, # extra/override per-model params
        eval_max_n: Optional[int] = None     # cap rows for SHAP eval per cluster
    ):
        self.df = df.copy()
        self.cluster_col = cluster_col
        self.random_state = random_state
        self.encoder = encoder
        self.model_type = model_type.lower().strip()
        self.user_model_params = model_params or {}
        self.eval_max_n = eval_max_n

        # Auto-detect features if not provided:
        if num_features is None and cat_features is None:
            num_features = [c for c in self.df.columns if c != cluster_col and _is_numeric(self.df[c])]
            cat_features = [c for c in self.df.columns if c != cluster_col and not _is_numeric(self.df[c])]
        else:
            if num_features is None:
                num_features = [c for c in self.df.columns if c != cluster_col and _is_numeric(self.df[c])]
                if cat_features:
                    num_features = [c for c in num_features if c not in cat_features]
            if cat_features is None:
                cat_features = [c for c in self.df.columns if c != cluster_col and not _is_numeric(self.df[c])]
                if num_features:
                    cat_features = [c for c in cat_features if c not in num_features]

        self.num_features = num_features or []
        self.cat_features = cat_features or []

        # numeric reference stats:
        if self.num_features:
            self.global_means = self.df[self.num_features].mean(numeric_only=True)
            self.global_medians = self.df[self.num_features].median(numeric_only=True)
        else:
            self.global_means = pd.Series(dtype=float)
            self.global_medians = pd.Series(dtype=float)

        # caches populated by fit()
        self.encoder_obj = None
        self._encoded_mapping: Dict[str, Tuple[str, Optional[str]]] = {}
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.models: Dict[object, object] = {}
        self.shap_cache: Dict[object, dict] = {}
        self._nearest = {}
        self.shap_figs: List[plt.Figure] = []
        self._shap_cluster_ids: List = []

    # ---------------------------- model factory (per-cluster aware for imbalance) ----------------------------
    def _make_model(self, y_tr_bin: pd.Series):
        mt = self.model_type
        params = dict(self.user_model_params)  # user overrides

        if mt == "rf":
            # ensure balanced class_weight unless user explicitly set it:
            params.setdefault("n_estimators", 100)
            params.setdefault("class_weight", "balanced")
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", -1)
            return RandomForestClassifier(**params)

        elif mt == "lgbm":
            if LGBMClassifier is None:
                raise ImportError("LightGBM not installed. `pip install lightgbm` to use model_type='lgbm'.")
            # LGBM can use class_weight='balanced' (sklearn API) OR scale_pos_weight:
            params.setdefault("n_estimators", 100)
            params.setdefault("class_weight", "balanced")
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", -1)
            return LGBMClassifier(**params)

        elif mt == "xgb":
            if XGBClassifier is None:
                raise ImportError("XGBoost not installed. `pip install xgboost` to use model_type='xgb'.")
            # Auto compute scale_pos_weight if not provided by user:
            pos = int(y_tr_bin.sum())
            neg = int(len(y_tr_bin) - pos)
            spw = (neg / max(pos, 1)) if pos > 0 else 1.0
            params.setdefault("n_estimators", 100)
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", -1)
            params.setdefault("eval_metric", "logloss") # logloss for classification
            params.setdefault("tree_method", "auto")  # fast default
            params.setdefault("scale_pos_weight", spw)
            return XGBClassifier(**params)

        else:
            raise ValueError("model_type must be one of: 'rf', 'lgbm', 'xgb'.")

    # ---------------------------- Encoding ----------------------------
    # Default OneHotEncoder, optional "LeaveOneOutEncoder" or "CatBoostEncoder"
    def _fit_encoder(self, X_cat: pd.DataFrame, y: pd.Series):
        if not self.cat_features:
            self.encoder_obj = None
            self._encoded_mapping = {}
            return

        if self.encoder == "onehot":
            try:
                oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                oh = OneHotEncoder(handle_unknown="ignore", sparse=False)  # scikit-learn <1.2
            oh.fit(X_cat)
            self.encoder_obj = oh
            cat_cols = oh.get_feature_names_out(self.cat_features)
            categories = oh.categories_
            mapping = {}
            for feat, cats in zip(self.cat_features, categories):
                for lvl in cats:
                    lvl_str = str(lvl)
                    mapping[f"{feat}_{lvl_str}"] = (feat, lvl_str)
            self._encoded_mapping = mapping

        elif self.encoder == "loo":
            if LeaveOneOutEncoder is None:
                raise ImportError("Install category-encoders for LeaveOneOutEncoder.")
            enc = LeaveOneOutEncoder(cols=self.cat_features, random_state=self.random_state)
            enc.fit(X_cat, y)
            self.encoder_obj = enc
            self._encoded_mapping = {c: (c, None) for c in self.cat_features}

        elif self.encoder == "catboost":
            if CatBoostEncoder is None:
                raise ImportError("Install category-encoders for CatBoostEncoder.")
            enc = CatBoostEncoder(cols=self.cat_features, random_state=self.random_state)
            enc.fit(X_cat, y)
            self.encoder_obj = enc
            self._encoded_mapping = {c: (c, None) for c in self.cat_features}
        else:
            raise ValueError("encoder must be one of: 'onehot', 'loo', 'catboost'.")

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X_num = df[self.num_features].copy() if self.num_features else pd.DataFrame(index=df.index)
        if not self.cat_features:
            return X_num
        if self.encoder == "onehot":
            X_cat = pd.DataFrame(
                self.encoder_obj.transform(df[self.cat_features]),
                columns=self.encoder_obj.get_feature_names_out(self.cat_features),
                index=df.index
            )
            return pd.concat([X_num, X_cat], axis=1)
        else:  # LOO / CatBoost encoders
            X_cat = self.encoder_obj.transform(df[self.cat_features])
            return pd.concat([X_num, X_cat], axis=1)

    # ---------------------------- shap helper ----------------------------
    def _shap_values_2d(self, model, X_bg: pd.DataFrame, X_eval: pd.DataFrame) -> np.ndarray:
        try:
            explainer = shap.Explainer(model, X_bg)
            out = explainer(X_eval)
            vals = getattr(out, "values", out)
        except Exception:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X_eval)

        if isinstance(vals, list):  # classic tree binary: [neg, pos]
            vals = vals[1]
        vals = np.asarray(vals)

        if vals.ndim == 3:  # (n,f,c) or (c,n,f)
            n, f = X_eval.shape
            if vals.shape[0] == n and vals.shape[1] == f:
                cls_idx = 1 if vals.shape[2] > 1 else -1
                vals = vals[:, :, cls_idx]
            elif vals.shape[2] == f:
                cls_idx = 1 if vals.shape[0] > 1 else -1
                vals = vals[cls_idx, :, :]
            else:
                vals = np.take(vals, indices=-1, axis=-1)

        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        return vals

    def _select_scope(self, cache: dict, scope: str):
        X_eval = cache["X_eval"]
        y_eval = cache["y_eval_bin"]
        if scope == "positive":
            pos_mask = (y_eval == 1).values
            return cache["shap_all"][pos_mask], X_eval.iloc[pos_mask]
        elif scope == "negative":
            neg_mask = (y_eval == 0).values
            return cache["shap_all"][neg_mask], X_eval.iloc[neg_mask]
        elif scope == "all":
            return cache["shap_all"], X_eval
        else:
            raise ValueError('importance_scope must be "positive", "negative", or "all".')

    # ---------------------------- train ONCE ----------------------------
    def fit(
        self,
        test_size: float = 0.2,
        sample_n: Optional[int] = None,
        sample_frac: Optional[float] = None,
        stratify_sample: bool = True
    ):
        """
        Fit encoder, one stratified split, and a one-vs-rest classifier per cluster. Cache SHAP.

        - Use full df (default), or pass `sample_n` / `sample_frac`.
        - If `stratify_sample=True`, sampling preserves cluster proportions.
        """
        # Stratified sampling BEFORE split/encoding:
        if (sample_n is not None) or (sample_frac is not None):
            if stratify_sample:
                df = _stratified_sample(self.df, self.cluster_col, n=sample_n, frac=sample_frac, random_state=self.random_state)
            else:
                df = self.df.sample(
                    n=sample_n if sample_n is not None else None,
                    frac=sample_frac if sample_frac is not None else None,
                    random_state=self.random_state
                ).reset_index(drop=True)
        else:
            df = self.df.reset_index(drop=True)

        y = df[self.cluster_col]
        X_cat = df[self.cat_features] if self.cat_features else pd.DataFrame(index=df.index)
        self._fit_encoder(X_cat, y)

        X_all = self._transform(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        clusters = sorted(y.unique())
        self._nearest = _nearest_cluster_centroid(df, self.cluster_col, self.num_features)

        self.models.clear()
        self.shap_cache.clear()

        for cl in clusters:
            y_tr_bin = (y_train == cl).astype(int)
            y_te_bin = (y_test == cl).astype(int)

            model = self._make_model(y_tr_bin).fit(X_train, y_tr_bin)
            self.models[cl] = model

            # SHAP on a (capped or not) eval subset for speed:
            X_eval = X_test
            y_eval = y_te_bin
            if self.eval_max_n is not None and len(X_eval) > self.eval_max_n:
                X_eval = X_eval.sample(self.eval_max_n, random_state=self.random_state)
                y_eval = y_eval.loc[X_eval.index]

            shap_all = self._shap_values_2d(model, X_train, X_eval)
            self.shap_cache[cl] = dict(
                X_eval=X_eval, y_eval_bin=y_eval,
                shap_all=shap_all
            )

        return self

    # aggregate one-hot back to feature:
    def _aggregate_importance(self, vector: np.ndarray, columns: List[str]) -> pd.Series:
        s = pd.Series(vector, index=columns)
        buckets = {}
        for col, val in s.items():
            orig, _ = self._encoded_mapping.get(col, (col, None))
            buckets[orig] = buckets.get(orig, 0.0) + float(val)
        return pd.Series(buckets).sort_values(ascending=False)

    # user-friendly labels:
    def _cat_mode_label(self, df_cluster: pd.DataFrame, feat: str) -> str:
        vc_c = df_cluster[feat].value_counts(normalize=True)
        if vc_c.empty: return f"{feat} (None)"
        cat1, p_c = vc_c.index[0], vc_c.iloc[0]
        p_g = self.df[feat].value_counts(normalize=True).get(cat1, 0.0)
        return f"{feat}={cat1} ({p_c:.0%}, global {p_g:.0%})"

    def _num_label(self, df_cluster: pd.DataFrame, feat: str) -> str:
        series = df_cluster[feat]
        mean_val = series.mean(); median_val = series.median()
        skew_val = series.skew()
        use_median = abs(skew_val) > 1
        global_stat = self.global_medians.get(feat, np.nan) if use_median else self.global_means.get(feat, np.nan)
        chosen = median_val if use_median else mean_val
        delta = chosen - global_stat
        return f"{feat}: {chosen:.2f} ({delta:+.2f})"

    # ----------------------------  Shap plots ----------------------------
    def plot_cluster_shap(self, top_n: Optional[int] = None, importance_scope: str = "positive", show: bool = True):
        if not self.models:
            raise RuntimeError("Call fit() first.")

        self.shap_figs = []
        self._shap_cluster_ids = []

        df = self.df
        # palette size adapts to actual number shown
        for cl in sorted(self.models.keys()):
            cache = self.shap_cache[cl]
            shap_mat, X_eval = self._select_scope(cache, importance_scope)
            mean_abs = np.abs(shap_mat).mean(0)
            agg_all = self._aggregate_importance(mean_abs, list(X_eval.columns))
            agg = agg_all if (top_n is None) else agg_all.head(top_n)

            base_palette = sns.color_palette("viridis", max(int(len(agg)), 10))

            mask = (df[self.cluster_col] == cl)
            labels = []
            for feat in agg.index:
                if feat in self.num_features:
                    labels.append(self._num_label(df.loc[mask], feat))
                elif feat in self.cat_features:
                    labels.append(self._cat_mode_label(df.loc[mask], feat))
                else:
                    labels.append(feat)

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            sns.barplot(x=agg.values, y=labels, palette=base_palette[:len(agg)], ax=ax)
            ttl_suffix = f"Top {len(agg)}" if top_n is not None else "All features"
            ax.set(title=f"{ttl_suffix} SHAP - Cluster {cl}",
                   xlabel="Mean |SHAP|", ylabel="")
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()

            if show:
                plt.show()

            self.shap_figs.append(fig)
            self._shap_cluster_ids.append(cl)

    # ---------------------------- classification_stats per cluster ----------------------------
    def get_cluster_classification_stats(self) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Call fit() first.")
        rows = []
        for cl, model in self.models.items():
            y_te_bin = (self.y_test == cl).astype(int)
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            acc = accuracy_score(y_te_bin, y_pred)
            prec = precision_score(y_te_bin, y_pred, zero_division=0)
            rec = recall_score(y_te_bin, y_pred)
            f1 = f1_score(y_te_bin, y_pred)
            auc = roc_auc_score(y_te_bin, y_proba)
            tn, fp, fn, tp = confusion_matrix(y_te_bin, y_pred, labels=[0,1]).ravel()
            rows.append(dict(Cluster=cl, Accuracy=acc, Precision=prec, Recall=rec,
                             F1=f1, ROC_AUC=auc, TN=tn, FP=fp, FN=fn, TP=tp))
        return pd.DataFrame(rows)

    # ---------------------------- top features ----------------------------
    def get_top_shap_features(self, top_n: Optional[int] = None, importance_scope: str = "positive") -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Call fit() first.")
        rows = []
        for cl in sorted(self.models.keys()):
            cache = self.shap_cache[cl]
            shap_mat, X_eval = self._select_scope(cache, importance_scope)
            mean_abs = np.abs(shap_mat).mean(0)
            agg_all = self._aggregate_importance(mean_abs, list(X_eval.columns))
            agg = agg_all if (top_n is None) else agg_all.head(top_n)
            for feat, val in agg.items():
                rows.append(dict(Cluster=cl, Feature=feat, Abs_SHAP=float(val)))
        return pd.DataFrame(rows).sort_values(["Cluster", "Abs_SHAP"], ascending=[True, False])

    # ---------------------------- contrastive importance ----------------------------
    def contrastive_importance(
        self,
        cluster_a,
        cluster_b,
        top_n: Optional[int] = None,
        importance_scope: str = "positive",
        mode: str = "hybrid",                 # "shap" | "effect" | "hybrid"
        weight_shap: float = 1.0,
        weight_effect: float = 1.0,
        min_support: float = 0.0              # for categorical lift
    ) -> pd.DataFrame:
        """
        Compare features that separate cluster_a vs cluster_b.

        mode="shap": previous behavior (SHAP-only).
        mode="effect": statistical contrasts only (numeric: |Δ_median| in IQR units + |d|;
                       categorical: max lift + Cramér's V).
        mode="hybrid": normalized blend of SHAP and effect metrics (default).
        """
        if cluster_a not in self.models or cluster_b not in self.models:
            raise ValueError("Unknown cluster id(s). Call fit() first.")

        # --- SHAP parts (already aggregated back to original features) ---
        cacheA = self.shap_cache[cluster_a]
        SA, XA = self._select_scope(cacheA, importance_scope)
        meanA = self._aggregate_importance(np.abs(SA).mean(0), list(XA.columns))

        cacheB = self.shap_cache[cluster_b]
        SB, XB = self._select_scope(cacheB, importance_scope)
        meanB = self._aggregate_importance(np.abs(SB).mean(0), list(XB.columns))

        shap_sum = pd.concat([meanA.rename("A"), meanB.rename("B")], axis=1).fillna(0.0).sum(axis=1)
        # ensure we have full feature list (nums + cats)
        all_feats = list(dict.fromkeys(list(self.num_features) + list(self.cat_features)))
        shap_sum = shap_sum.reindex(all_feats).fillna(0.0)

        # --- Effect-size parts from raw data ---
        df = self.df
        mask_a = (df[self.cluster_col] == cluster_a)
        mask_b = (df[self.cluster_col] == cluster_b)

        # Numeric: standardized median gap (pooled IQR) + Cohen's d
        sd_vals, d_vals = {}, {}
        for feat in self.num_features:
            a = pd.to_numeric(df.loc[mask_a, feat], errors="coerce").dropna()
            b = pd.to_numeric(df.loc[mask_b, feat], errors="coerce").dropna()
            if len(a) < 2 or len(b) < 2:
                sd_vals[feat] = np.nan; d_vals[feat] = np.nan; continue
            med_a, med_b = a.median(), b.median()
            iqr_a = a.quantile(0.75) - a.quantile(0.25)
            iqr_b = b.quantile(0.75) - b.quantile(0.25)
            denom = np.nanmean([iqr_a, iqr_b])
            sd_vals[feat] = (med_a - med_b) / denom if denom and np.isfinite(denom) and denom != 0 else np.nan
            d_vals[feat] = _cohens_d(a, b)

        sd_series = pd.Series(sd_vals, dtype="float64")
        d_series  = pd.Series(d_vals, dtype="float64")

        # Categorical: max lift for any category + Cramér's V (2xK, A vs B only)
        lift_vals, V_vals = {}, {}
        sub = df.loc[mask_a | mask_b]
        for feat in self.cat_features:
            vc_a = sub.loc[mask_a, feat].value_counts(normalize=True, dropna=True)
            vc_b = sub.loc[mask_b, feat].value_counts(normalize=True, dropna=True)
            best_lift = []
            for cat, p_a in vc_a.items():
                if p_a < min_support: continue
                p_b = vc_b.get(cat, 0.0)
                lift = np.inf if p_b == 0 else (p_a / p_b)
                best_lift.append(lift)
            lift_vals[feat] = (max(best_lift) if best_lift else np.nan)

            tbl = pd.crosstab(sub[self.cluster_col].isin([cluster_a, cluster_b]), sub[feat])
            V_vals[feat] = _cramers_v(tbl) if tbl.values.sum() else np.nan

        lift_series = pd.Series(lift_vals, dtype="float64")
        V_series    = pd.Series(V_vals, dtype="float64")

        # Reindex to all features and take magnitudes where relevant
        sd_abs   = sd_series.abs().reindex(all_feats).fillna(0.0)
        d_abs    = d_series.abs().reindex(all_feats).fillna(0.0)
        lift_abs = pd.Series(lift_series, dtype="float64").reindex(all_feats).fillna(0.0)
        V_abs    = pd.Series(V_series, dtype="float64").reindex(all_feats).fillna(0.0)
        shap_abs = shap_sum.reindex(all_feats).fillna(0.0)

        def _norm01(s: pd.Series) -> pd.Series:
            s = s.astype(float)
            smax, smin = s.max(), s.min()
            if not np.isfinite(smax) or not np.isfinite(smin) or smax == smin:
                return pd.Series(0.0, index=s.index)
            return (s - smin) / (smax - smin)

        shap_n = _norm01(shap_abs)
        # numeric-only effects
        sd_n   = _norm01(sd_abs)
        d_n    = _norm01(d_abs)
        # cat-only effects
        lift_n = _norm01(lift_abs.replace(np.inf, np.nan).fillna(lift_abs[~np.isinf(lift_abs)].max() if np.isinf(lift_abs).any() else 0.0))
        V_n    = _norm01(V_abs)

        if mode not in {"shap", "effect", "hybrid"}:
            raise ValueError("mode must be one of {'shap','effect','hybrid'}")

        if mode == "shap":
            score = shap_n
        elif mode == "effect":
            score = sd_n + d_n + lift_n + V_n
        else:  # hybrid
            score = weight_shap * shap_n + weight_effect * (sd_n + d_n + lift_n + V_n)

        out = pd.DataFrame({
            "Feature": all_feats,
            "Score": score.reindex(all_feats).values,
            "SHAP_norm": shap_n.reindex(all_feats).values,
            "StdMedianGap_norm": sd_n.reindex(all_feats).values,
            "CohensD_norm": d_n.reindex(all_feats).values,
            "Lift_norm": lift_n.reindex(all_feats).values,
            "CramersV_norm": V_n.reindex(all_feats).values
        }).sort_values("Score", ascending=False)

        if top_n is not None:
            out = out.head(int(top_n))
        return out.reset_index(drop=True)

    # ---------------------------- distribution Plots ----------------------------
    def compare_feature_across_clusters(
        self,
        feature: Optional[str] = None,
        bins: int = 30,
        auto_log_skew: Optional[float] = None,   # DEFAULT: no log1p; pass a threshold (e.g., 1.5) to enable
        linewidth: float = 1.5,
        alpha: float = 0.9
    ):
        """
        Compare a feature's distribution across clusters using HISTOGRAMS with raw counts ONLY.
        - Numeric: overlaid step histograms (density=False).
        - Categorical: stacked bar chart of counts.
        """
        df = self.df
        clusters = sorted(df[self.cluster_col].unique())
        palette = sns.color_palette("tab10", n_colors=len(clusters))
        y_label = "Count"

        def _finite_numeric(s):
            s = pd.to_numeric(s, errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            return s

        def _maybe_log1p(s):
            if auto_log_skew is None:  # default: keep raw scale
                return s, False
            if len(s) == 0: return s, False
            if s.min() >= 0 and abs(pd.Series(s).skew()) > float(auto_log_skew):
                return np.log1p(s), True
            return s, False

        if feature:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not found.")
            if feature in self.num_features:
                fig, ax = plt.subplots(figsize=(10, 6))
                used_log = False
                for i, cl in enumerate(clusters):
                    s = _finite_numeric(df.loc[df[self.cluster_col] == cl, feature])
                    if len(s) < 2:
                        continue
                    s, did_log = _maybe_log1p(s); used_log |= did_log
                    ax.hist(
                        s, bins=bins, density=False, histtype="step",
                        label=f"C{cl}", linewidth=linewidth, alpha=alpha, color=palette[i]
                    )
                ax.set_title(f"Distribution of '{feature}' by Cluster" + (" (log1p)" if used_log else ""))
                ax.set_xlabel(feature); ax.set_ylabel(y_label)
                ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout(); plt.show()
            elif feature in self.cat_features:
                counts = (df.groupby(self.cluster_col)[feature]
                        .value_counts(normalize=False).rename("count").reset_index())
                pivot = counts.pivot(index=self.cluster_col, columns=feature, values="count").fillna(0.0)
                pivot = pivot.loc[clusters]
                ax = pivot.plot(kind="bar", stacked=True, figsize=(10,6), width=0.85, edgecolor="black")
                ax.set_title(f"Stacked distribution of '{feature}' by Cluster")
                ax.set_xlabel("Cluster"); ax.set_ylabel(y_label)
                ax.legend(title=feature, bbox_to_anchor=(1.02,1), loc="upper left")
                plt.tight_layout(); plt.show()
            else:
                raise ValueError(f"Feature '{feature}' must be numeric or categorical.")
            return

        # Faceted plots for ALL numeric features
        feats = self.num_features
        if not feats:
            raise ValueError("No numeric features to plot.")
        cols = 3; rows = math.ceil(len(feats)/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
        axes = np.array(axes).reshape(-1)
        for i, feat in enumerate(feats):
            ax = axes[i]
            used_log = False
            for j, cl in enumerate(clusters):
                s = _finite_numeric(df.loc[df[self.cluster_col] == cl, feat])
                if len(s) < 2:
                    continue
                s, did_log = _maybe_log1p(s); used_log |= did_log
                ax.hist(
                    s, bins=bins, density=False, histtype="step",
                    label=f"C{cl}" if i==0 else None,
                    linewidth=1.2, alpha=0.9, color=palette[j]
                )
            ax.set_title(f"{feat}" + (" (log1p)" if used_log else ""))
            ax.set_xlabel(feat); ax.set_ylabel(y_label)
        for j in range(len(feats), len(axes)): fig.delaxes(axes[j])
        handles, labels = axes[0].get_legend_handles_labels()
        if handles: fig.legend(handles, labels, title="Cluster", loc="upper right")
        plt.tight_layout(); plt.show()

    # ---------------------------- narratives ----------------------------
    def generate_cluster_narratives(self, top_n: Optional[int] = None, min_support: float = 0.05, output: str = "markdown"):
        if self.df is None:
            raise ValueError("DataFrame is missing.")
        df = self.df
        clusters = sorted(df[self.cluster_col].unique())
        n_total = len(df)

        g_med = df[self.num_features].median() if self.num_features else pd.Series(dtype=float)
        g_q1  = df[self.num_features].quantile(0.25) if self.num_features else pd.Series(dtype=float)
        g_q3  = df[self.num_features].quantile(0.75) if self.num_features else pd.Series(dtype=float)
        g_iqr = (g_q3 - g_q1).replace(0, np.nan) if self.num_features else pd.Series(dtype=float)

        out = []
        masks = {cl: (df[self.cluster_col] == cl) for cl in clusters}
        nearest_map = _nearest_cluster_centroid(df, self.cluster_col, self.num_features)

        for cl in clusters:
            mask = masks[cl]
            grp = df.loc[mask]
            n = len(grp)
            pct = n / n_total if n_total else 0
            nearest = nearest_map.get(cl)

            numeric_rank = []
            for feat in self.num_features:
                c_med = grp[feat].median()
                sd = _std_delta(c_med, g_med.get(feat, np.nan), g_iqr.get(feat, np.nan))
                others = df.loc[~mask, feat]
                try:
                    _, p = mannwhitneyu(grp[feat].dropna(), others.dropna(), alternative="two-sided")
                except ValueError:
                    p = np.nan
                d = _cohens_d(grp[feat], others)
                numeric_rank.append((feat, abs(sd if np.isfinite(sd) else 0), sd, d, p))
            numeric_rank.sort(key=lambda x: x[1], reverse=True)
            cut_num = len(numeric_rank) if top_n is None else int(top_n)
            lines_num = []
            for feat, _, sd, d, p in numeric_rank[:cut_num]:
                direction = "high" if (sd or 0) > 0 else "low"
                p_txt = "<0.001" if (p is not None and np.isfinite(p) and p < 0.001) else f"{p:.3f}" if p is not None and np.isfinite(p) else "n/a"
                lines_num.append(f"{feat}: {direction} ({sd:.2f} IQR units), d={d:.2f}, p={p_txt}")

            lines_cat = []
            if self.cat_features:
                cat_rank = []
                for feat in self.cat_features:
                    vc_c = grp[feat].value_counts(normalize=True, dropna=True)
                    if vc_c.empty: continue
                    vc_g = df[feat].value_counts(normalize=True, dropna=True)
                    best = []
                    for cat, p_c in vc_c.items():
                        if p_c < min_support: continue
                        p_g = vc_g.get(cat, 0)
                        lift = (p_c / p_g) if p_g > 0 else np.inf
                        best.append((feat, cat, p_c, p_g, lift))
                    if not best: continue
                    best.sort(key=lambda t: t[4], reverse=True)
                    tbl = pd.crosstab(df[self.cluster_col] == cl, df[feat])
                    V = _cramers_v(tbl)
                    cat_rank.append((feat, best[0], V))
                cat_rank.sort(key=lambda r: r[1][4], reverse=True)
                cut_cat = len(cat_rank) if top_n is None else int(top_n)
                for feat, (_, cat, p_c, p_g, lift), V in cat_rank[:cut_cat]:
                    lines_cat.append(f"{feat}={cat} ({p_c:.0%}, global {p_g:.0%}), lift={lift:.2f}, V={V:.2f}")

            lines_contrast = []
            if nearest is not None and self.num_features:
                grp_b = df.loc[masks[nearest]]
                deltas = []
                for feat in self.num_features:
                    d_med = grp[feat].median() - grp_b[feat].median()
                    iqr_a = grp[feat].quantile(0.75) - grp[feat].quantile(0.25)
                    iqr_b = grp_b[feat].quantile(0.75) - grp_b[feat].quantile(0.25)
                    denom = np.nanmean([iqr_a, iqr_b])
                    sd = d_med / denom if denom and np.isfinite(denom) and denom != 0 else np.nan
                    deltas.append((feat, abs(sd if np.isfinite(sd) else 0), sd))
                deltas.sort(key=lambda x: x[1], reverse=True)
                cut_con = len(deltas) if top_n is None else int(top_n)
                for feat, _, sd in deltas[:cut_con]:
                    direction = "higher" if (sd or 0) > 0 else "lower"
                    lines_contrast.append(f"vs C{nearest}: {feat} {direction} ({sd:.2f} IQR units)")

            header = f"### Cluster {cl} — N={n} ({pct:.0%})"
            bullets = [
                "High/low numeric drivers: " + (", ".join(lines_num) if lines_num else "none"),
                "Dominant categories: " + (", ".join(lines_cat) if lines_cat else "none")
            ]
            if lines_contrast:
                bullets.append("Key differences vs nearest: " + ", ".join(lines_contrast))

            if output == "dict":
                out.append({
                    "cluster": cl, "n": n, "pct": pct,
                    "numeric": lines_num, "categorical": lines_cat,
                    "contrast_nearest": lines_contrast
                })
            else:
                out.append("\n\n".join([header] + [f"- {b}" for b in bullets]))

        return out

    # ---------------------------- per-cluster table ----------------------------
    def get_cluster_summary(self, sample_size: Optional[int] = None, top_n_contrast: Optional[int] = None, min_support: float = 0.05):
        df_s = self.df.sample(sample_size, random_state=self.random_state).reset_index(drop=True) \
               if sample_size and sample_size < len(self.df) else self.df.copy().reset_index(drop=True)

        summaries = []
        nearest_map = _nearest_cluster_centroid(df_s, self.cluster_col, self.num_features)

        for cl in sorted(df_s[self.cluster_col].unique()):
            mask = df_s[self.cluster_col] == cl
            grp = df_s.loc[mask]
            summary = {self.cluster_col: cl, "N": len(grp), "Pct": f"{len(grp)/len(df_s):.0%}"}

            for feat in self.num_features:
                series = grp[feat]
                mean_val = series.mean()
                median_val = series.median()
                skew_val = series.skew()
                use_median = abs(skew_val) > 1
                global_stat = self.global_medians.get(feat, np.nan) if use_median else self.global_means.get(feat, np.nan)
                chosen = median_val if use_median else mean_val
                delta = chosen - global_stat
                summary[feat] = f"{chosen:.2f} ({delta:+.2f})"

            for feat in self.cat_features:
                vc_c = grp[feat].value_counts(normalize=True)
                if vc_c.empty:
                    summary[feat] = "None"; continue
                cat1 = vc_c.index[0]; p_c = vc_c.iloc[0]
                vc_g = df_s[feat].value_counts(normalize=True)
                p_g = vc_g.get(cat1, 0)
                summary[feat] = f"{cat1} ({p_c:.0%}, global: {p_g:.0%})"

            nearest = nearest_map.get(cl)
            contrast_num, contrast_cat = [], []
            if nearest is not None and self.num_features:
                grp_b = df_s.loc[df_s[self.cluster_col] == nearest]
                cand = []
                for feat in self.num_features:
                    d_med = grp[feat].median() - grp_b[feat].median()
                    iqr_a = grp[feat].quantile(0.75) - grp[feat].quantile(0.25)
                    iqr_b = grp_b[feat].quantile(0.75) - grp_b[feat].quantile(0.25)
                    denom = np.nanmean([iqr_a, iqr_b])
                    sd = d_med / denom if denom and np.isfinite(denom) and denom != 0 else np.nan
                    cand.append((feat, abs(sd if np.isfinite(sd) else 0), sd))
                cand.sort(key=lambda x: x[1], reverse=True)
                cut = len(cand) if top_n_contrast is None else int(top_n_contrast)
                for feat, _, sd in cand[:cut]:
                    direction = "higher" if (sd or 0) > 0 else "lower"
                    contrast_num.append(f"{feat} {direction} ({sd:.2f} IQR units)")

                for feat in self.cat_features:
                    vc_a = grp[feat].value_counts(normalize=True)
                    vc_b = grp_b[feat].value_counts(normalize=True)
                    best = []
                    for cat, p_a in vc_a.items():
                        if p_a < min_support: continue
                        p_b = vc_b.get(cat, 0)
                        lift = np.inf if p_b == 0 else (p_a / p_b)
                        best.append((cat, p_a, p_b, lift))
                    if not best: continue
                    best.sort(key=lambda t: t[3], reverse=True)
                    top = best[0]
                    contrast_cat.append(f"{feat}={top[0]} (C{cl} {top[1]:.0%} vs C{nearest} {top[2]:.0%}, lift {top[3]:.2f})")

            summary["Contrast_Numeric"] = ", ".join(contrast_num[:cut]) if contrast_num else ""
            summary["Contrast_Categorical"] = ", ".join(contrast_cat[:cut]) if contrast_cat else ""

            summaries.append(summary)

        return pd.DataFrame(summaries)
    # ---------------------------- split checks ----------------------------
    def get_split_table(self) -> pd.DataFrame:
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise RuntimeError("Call fit() first.")

        n_train = len(self.X_train)
        n_test  = len(self.X_test)
        total_rows = n_train + n_test

        clusters = pd.Index(self.y_train.unique()).union(self.y_test.unique())
        try:
            clusters = sorted(clusters)
        except Exception:
            pass

        rows = []
        for cl in clusters:
            tr_one = int((self.y_train == cl).sum())
            te_one = int((self.y_test  == cl).sum())

            tr_rest = n_train - tr_one
            te_rest = n_test  - te_one

            cluster_size = tr_one + te_one
            train_share = (tr_one / cluster_size) if cluster_size else float("nan")

            rows.append({
                "Cluster": cl,
                "train one": tr_one,
                "train rest": tr_rest,
                "test one": te_one,
                "test rest": te_rest,
                "cluster size": cluster_size,
                "dataset rows": total_rows,
                "%": train_share,
            })

        return pd.DataFrame(rows)

    # ---------------------------- export helpers ----------------------------
    def export_summary(self, path: str = "./cluster_summary.csv"):
        df = self.get_cluster_summary()
        df.to_csv(path, index=False)
        return path

    def save_shap_figs(self, folder_path: str = "./shap_figs"):
        os.makedirs(folder_path, exist_ok=True)
        for fig, cl in zip(self.shap_figs, self._shap_cluster_ids):
            out = os.path.join(folder_path, f"shap_cluster_{cl}.png")
            fig.savefig(out, bbox_inches="tight", dpi=150)
        return folder_path
