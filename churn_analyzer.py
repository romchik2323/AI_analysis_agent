

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ChurnReport:
    
    dataset_name: str
    total_customers: int
    churned_customers: int
    churn_rate: float
    model_accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    top_factors: list[dict]          
    high_risk_count: int
    high_risk_churn_rate: float
    segment_analysis: dict          
    recommendations: list[str]


class UniversalChurnAnalyzer:
    

    CHURN_COLUMN_ALIASES = [
        "churn", "churned", "is_churn", "target", "label",
        "attrition", "cancelled", "canceled", "left", "exited"
    ]

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model: Optional[RandomForestClassifier] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_columns: list[str] = []
        self.target_column: str = ""
        self.df: Optional[pd.DataFrame] = None

    
    def load_data(self, path: str, target_column: str = "auto") -> pd.DataFrame:
        
        self.df = pd.read_csv(path)
        print(f"✅ Загружено {len(self.df):,} строк, {len(self.df.columns)} колонок")

        
        if target_column == "auto":
            found = None
            for col in self.df.columns:
                if col.lower() in self.CHURN_COLUMN_ALIASES:
                    found = col
                    break
            if not found:
                raise ValueError(
                    f"Не найдена целевая колонка. Укажите явно. "
                    f"Доступные колонки: {list(self.df.columns)}"
                )
            self.target_column = found
        else:
            self.target_column = target_column

        print(f"🎯 Целевая колонка: {self.target_column}")
        return self.df

    def _preprocess(self) -> tuple[pd.DataFrame, pd.Series]:
        
        df = self.df.copy()

        
        target = df[self.target_column]
        
        is_string_type = (
            target.dtype == object
            or hasattr(target.dtype, 'name') and 'string' in str(target.dtype).lower()
            or pd.api.types.is_string_dtype(target)
        )
        if is_string_type:
            positive_values = {"yes", "true", "1", "churned", "left", "exited"}
            y = target.astype(str).str.lower().isin(positive_values).astype(int)
        else:
            y = target.astype(int)

        
        drop_cols = [self.target_column]
        for col in df.columns:
            if "id" in col.lower() and df[col].nunique() > len(df) * 0.9:
                drop_cols.append(col)

        X = df.drop(columns=drop_cols, errors="ignore")

        
        num_cols = X.select_dtypes(include=[np.number]).columns
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

        
        cat_cols = X.select_dtypes(include=[object]).columns
        for col in cat_cols:
            X[col] = X[col].fillna("Unknown")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        self.feature_columns = list(X.columns)
        return X, y

    
    def train(self, model_name: str = "random_forest") -> ChurnReport:
        
        X, y = self._preprocess()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if model_name == "gradient_boosting":
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1,
                class_weight="balanced"
            )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        
        acc = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        
        importances = self.model.feature_importances_
        top_factors = sorted(
            [{"feature": f, "importance": round(float(i), 4)}
             for f, i in zip(self.feature_columns, importances)],
            key=lambda x: x["importance"], reverse=True
        )[:10]

        
        cat_cols = list(self.label_encoders.keys())
        segment_analysis = {}
        df_with_target = self.df.copy()
        _target_col = df_with_target[self.target_column]
        if pd.api.types.is_string_dtype(_target_col):
            positive_values = {"yes", "true", "1", "churned", "left", "exited"}
            df_with_target["_churn_num"] = (
                _target_col.astype(str).str.lower()
                .isin(positive_values).astype(int)
            )
        else:
            df_with_target["_churn_num"] = _target_col.astype(int)

        for col in cat_cols[:5]:
            rates = (
                df_with_target.groupby(col)["_churn_num"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "churn_rate", "count": "total"})
            )
            rates["churn_rate"] = (rates["churn_rate"] * 100).round(1)
            segment_analysis[col] = rates.to_dict(orient="index")

        
        all_proba = self.model.predict_proba(X)[:, 1]
        high_risk_mask = all_proba > 0.6
        hr_count = high_risk_mask.sum()
        hr_churn_rate = y[high_risk_mask].mean() if hr_count > 0 else 0

        
        recommendations = self._generate_rule_based_recommendations(
            top_factors, segment_analysis
        )

        report = ChurnReport(
            dataset_name=Path(self.df.columns[0]).stem if hasattr(self, "_path") else "dataset",
            total_customers=len(self.df),
            churned_customers=int(y.sum()),
            churn_rate=round(float(y.mean()) * 100, 2),
            model_accuracy=round(acc * 100, 2),
            roc_auc=round(auc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            top_factors=top_factors,
            high_risk_count=int(hr_count),
            high_risk_churn_rate=round(float(hr_churn_rate) * 100, 2),
            segment_analysis=segment_analysis,
            recommendations=recommendations
        )

        
        self._save_model()
        print(f"\n✅ Модель обучена: Accuracy={acc:.1%}, ROC-AUC={auc:.4f}")
        return report

    def _generate_rule_based_recommendations(
        self, top_factors: list, segment_analysis: dict
    ) -> list[str]:
        
        recs = []
        top_feature = top_factors[0]["feature"] if top_factors else ""
        recs.append(
            f"Ключевой фактор риска: '{top_feature}'. "
            "Сфокусируйте удержание на клиентах с высоким значением этого признака."
        )
        for col, data in segment_analysis.items():
            high_churn_seg = max(data.items(), key=lambda x: x[1]["churn_rate"])
            recs.append(
                f"Сегмент '{col}={high_churn_seg[0]}' имеет наибольший отток "
                f"({high_churn_seg[1]['churn_rate']}%). Требует приоритетного внимания."
            )
        recs.append(
            "Внедрите систему скоринга риска: регулярно пересчитывайте вероятности "
            "оттока и автоматически триггерируйте retention-кампании."
        )
        return recs[:5]

    
    def score_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if self.model is None:
            raise RuntimeError("Модель не обучена. Вызовите train() сначала.")

        X = df.copy()
        drop_cols = []
        for col in X.columns:
            if "id" in col.lower() and X[col].nunique() > len(X) * 0.9:
                drop_cols.append(col)
        if self.target_column in X.columns:
            drop_cols.append(self.target_column)
        X = X.drop(columns=drop_cols, errors="ignore")

        num_cols = X.select_dtypes(include=[np.number]).columns
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna("Unknown").astype(str)
                known = set(le.classes_)
                X[col] = X[col].apply(lambda v: v if v in known else le.classes_[0])
                X[col] = le.transform(X[col])

        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        X = X[self.feature_columns]
        proba = self.model.predict_proba(X)[:, 1]

        result = df.copy()
        result["churn_probability"] = (proba * 100).round(1)
        result["risk_level"] = pd.cut(
            proba,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["🟢 LOW", "🟡 MEDIUM", "🔴 HIGH"],
            include_lowest=True
        )
        return result.sort_values("churn_probability", ascending=False)

    def score_single(self, customer: dict) -> dict:
        
        df = pd.DataFrame([customer])
        result = self.score_customers(df)
        row = result.iloc[0]
        return {
            "churn_probability": row["churn_probability"],
            "risk_level": str(row["risk_level"]),
        }

   
    def _save_model(self):
        data = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
        }
        with open(self.model_dir / "churn_model.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Модель сохранена в {self.model_dir}/churn_model.pkl")

    def load_model(self):
        path = self.model_dir / "churn_model.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_encoders = data["label_encoders"]
        self.feature_columns = data["feature_columns"]
        self.target_column = data["target_column"]
        print("✅ Модель загружена")