from __future__ import annotations

import json
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Matplotlib / Seaborn settings
sns.set(context="talk", style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------
# Data loading
# -----------------------------
class DataLoader:
    """
    Loads the grasp dataset from CSV and returns features and labels.

    This class centralizes:
    - Input validation
    - Column selection
    - Type casting
    """

    REQUIRED_COLS: List[str] = [
        "rel_x",
        "rel_y",
        "rel_z",
        "rel_roll",
        "rel_pitch",
        "rel_yaw",
        "success",
    ]

    FEATURE_COLS: List[str] = [
        "rel_x",
        "rel_y",
        "rel_z",
        "rel_roll",
        "rel_pitch",
        "rel_yaw",
    ]
    TARGET_COL: str = "success"

    def __init__(self, csv_path: str) -> None:
        """
        Parameters
        ----------
        csv_path : str
            Path to the grasp dataset CSV generated from simulation.
        """
        self.csv_path = csv_path

    def load(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load and validate CSV, return (df, X, y).

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray, np.ndarray]
            DataFrame df, features X, labels y.
        """
        df = pd.read_csv(self.csv_path)
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        X = df[self.FEATURE_COLS].values.astype(np.float32)
        y = df[self.TARGET_COL].astype(int).values
        return df, X, y


# -----------------------------
# Model specifications (OOP)
# -----------------------------
class BaseModelSpec(ABC):
    """
    Abstract base class representing a model "specification":
    - A named estimator pipeline (can include preprocessors like scalers)
    - A parameter grid for hyperparameter search

    Subclasses must implement:
    - build_pipeline()
    - param_grid()
    """

    name: str

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """Create and return the sklearn Pipeline for this model family."""
        raise NotImplementedError

    @abstractmethod
    def param_grid(self) -> Dict[str, List]:
        """Return the parameter grid for GridSearchCV."""
        raise NotImplementedError


class LogisticRegressionSpec(BaseModelSpec):
    """Logistic Regression with StandardScaler."""

    name = "LogisticRegression"

    def build_pipeline(self) -> Pipeline:
        # Standardize features before a linear model
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        )

    def param_grid(self) -> Dict[str, List]:
        # Typical L2 regularization sweep
        return {
            "clf__penalty": ["l2"],
            "clf__C": [0.1, 1, 10, 100],
            "clf__solver": ["lbfgs"],
        }


class RandomForestSpec(BaseModelSpec):
    """Random Forest (tree-based, no scaling required)."""

    name = "RandomForest"

    def build_pipeline(self) -> Pipeline:
        # Tree-based models do not need scaling; keep pipeline for API parity.
        return Pipeline(
            steps=[
                ("clf", RandomForestClassifier(random_state=42)),
            ]
        )

    def param_grid(self) -> Dict[str, List]:
        # A compact yet effective grid for small-to-medium datasets
        return {
            "clf__n_estimators": [100, 300, 600],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        }


class KNNSpec(BaseModelSpec):
    """K-Nearest Neighbors with StandardScaler."""

    name = "KNN"

    def build_pipeline(self) -> Pipeline:
        # Scale features to avoid dominant dimensions
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]
        )

    def param_grid(self) -> Dict[str, List]:
        # Classic KNN hyperparameters
        return {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],  # Manhattan (1) or Euclidean (2)
        }


class SVMRBFSpec(BaseModelSpec):
    """SVM with RBF kernel and StandardScaler."""

    name = "SVM_RBF"

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True)),
            ]
        )

    def param_grid(self) -> Dict[str, List]:
        # RBF hyperparameters (C, gamma)
        return {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", 0.01, 0.1, 1],
        }


class GradientBoostingSpec(BaseModelSpec):
    """Gradient Boosting Classifier (tree-based)."""

    name = "GradientBoosting"

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]
        )

    def param_grid(self) -> Dict[str, List]:
        # Common hyperparameters: number of estimators, learning rate, depth
        return {
            "clf__n_estimators": [100, 200, 400],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [2, 3, 4],
        }


# -----------------------------
# Training components
# -----------------------------
@dataclass
class TrainingResult:
    """
    Holds the result of training a single model spec.

    Attributes
    ----------
    model_name : str
        Name of the model family.
    best_estimator : BaseEstimator
        Best estimator found by GridSearchCV.
    best_params : Dict
        Best hyperparameters.
    cv_f1 : float
        Best cross-validated F1 score.
    search_obj : GridSearchCV
        The search object (for access to cv results if needed).
    """
    model_name: str
    best_estimator: BaseEstimator
    best_params: Dict
    cv_f1: float
    search_obj: GridSearchCV


class ModelTrainer:
    """
    Trains one model spec with GridSearchCV using F1 scoring.

    This class demonstrates "apply function" pattern via delegation:
    - Accepts a BaseModelSpec to build pipeline & grid
    - Applies a standard training procedure across different specs
    """

    def __init__(self, cv_splits: int = 5, scoring: str = "f1") -> None:
        """
        Parameters
        ----------
        cv_splits : int
            Number of StratifiedKFold splits.
        scoring : str
            Scoring metric for model selection (e.g., 'f1').
        """
        self.cv_splits = cv_splits
        self.scoring = scoring

    def fit(self, spec: BaseModelSpec, X_train: np.ndarray, y_train: np.ndarray) -> TrainingResult:
        """
        Fit GridSearchCV for a given spec and return the best model and metadata.

        Parameters
        ----------
        spec : BaseModelSpec
            A concrete model specification.
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.

        Returns
        -------
        TrainingResult
            Best model, parameters, CV F1, and the search object.
        """
        pipeline = spec.build_pipeline()
        param_grid = spec.param_grid()
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=42)

        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        return TrainingResult(
            model_name=spec.name,
            best_estimator=gs.best_estimator_,
            best_params=gs.best_params_,
            cv_f1=float(gs.best_score_),
            search_obj=gs,
        )


class Evaluator:
    """
    Generates evaluation metrics and plots for a trained model.

    Responsibilities:
    - Predict on held-out test set
    - Compute metrics (accuracy, f1, precision, recall)
    - Confusion matrix
    - ROC curve / PR curve (if probabilities available)
    - Learning curve (bias/variance diagnostic)
    """

    def __init__(self, out_dir: str) -> None:
        """
        Parameters
        ----------
        out_dir : str
            Output directory for metrics and plots.
        """
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def evaluate_and_plot(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Evaluate model on the hold-out test set and save plots and metrics.

        Returns
        -------
        Dict
            Dictionary with evaluation metrics and a full classification report.
        """
        metrics = {}

        # Predictions (hard labels)
        y_pred = model.predict(X_test)

        # Core metrics
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1"] = float(f1_score(y_test, y_pred))
        metrics["precision"] = float(precision_score(y_test, y_pred))
        metrics["recall"] = float(recall_score(y_test, y_pred))
        metrics["report"] = classification_report(y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm)

        # ROC / PR curves when probabilities are available
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
            except Exception:
                metrics["roc_auc"] = None
            try:
                metrics["avg_precision"] = float(average_precision_score(y_test, y_score))
            except Exception:
                metrics["avg_precision"] = None

            self._plot_roc(y_test, y_score)
            self._plot_pr(y_test, y_score)

        # Learning curve (diagnostic for bias/variance)
        self._plot_learning_curve(model, X_train, X_test, y_train, y_test)

        # Persist metrics
        self._save_json("metrics.json", metrics)

        return metrics

    # ---- Plot helpers (apply functions / composition) ----
    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "confusion_matrix.png"))
        plt.close()

    def _plot_roc(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        plt.figure(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_true, y_score)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "roc_curve.png"))
        plt.close()

    def _plot_pr(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        plt.figure(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_true, y_score)
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "pr_curve.png"))
        plt.close()

    def _plot_learning_curve(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_all = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_all,
            y_all,
            cv=cv,
            scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 5),
            shuffle=True,
            n_jobs=-1,
        )
        plt.figure(figsize=(6, 4))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="Training")
        plt.plot(train_sizes, np.mean(val_scores, axis=1), "o-", label="Cross-val")
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "learning_curve.png"))
        plt.close()

    def _save_json(self, filename: str, obj: Dict) -> None:
        with open(os.path.join(self.out_dir, filename), "w") as f:
            json.dump(obj, f, indent=2)


class TrainingOrchestrator:
    """
    Coordinates end-to-end training across multiple model specifications:
    - Splits data (stratified)
    - Trains each spec with ModelTrainer
    - Selects the best by CV F1
    - Evaluates on the test set
    - Saves model and artifacts

    Demonstrates OOP composition:
    - Orchestrator composes DataLoader, ModelTrainer, Evaluator, and Specs.
    """

    def __init__(
        self,
        specs: List[BaseModelSpec],
        cv_splits: int = 5,
        scoring: str = "f1",
        out_dir: str = "models",
        test_size: float = 0.25,
    ) -> None:
        self.specs = specs
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.out_dir = out_dir
        self.test_size = test_size

        os.makedirs(self.out_dir, exist_ok=True)

    def run(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Run the full orchestration and return test metrics for the best model.

        Steps:
        - Train/test split
        - For each model spec: training via GridSearchCV
        - Select best by CV F1
        - Evaluate best on test set
        - Persist artifacts (model, best params)
        """
        # Stratified train/test split to preserve class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        # Train per spec
        trainer = ModelTrainer(cv_splits=self.cv_splits, scoring=self.scoring)
        results: List[TrainingResult] = []
        print("Starting model selection via GridSearchCV (LR, RF, KNN, SVM-RBF, GB) ...")
        for spec in self.specs:
            res = trainer.fit(spec, X_train, y_train)
            print(f"[{res.model_name}] best CV F1={res.cv_f1:.3f} params={res.best_params}")
            results.append(res)

        # Select the best by CV F1
        best = max(results, key=lambda r: r.cv_f1)
        print(f"\nBest model: {best.model_name} with CV F1={best.cv_f1:.3f}")

        # Evaluate and plot using the best model
        evaluator = Evaluator(out_dir=self.out_dir)
        metrics = evaluator.evaluate_and_plot(best.best_estimator, X_train, X_test, y_train, y_test)

        # Persist the best model and params
        model_path = os.path.join(self.out_dir, "best_model.pkl")
        dump(best.best_estimator, model_path)
        print(f"Saved best model to: {model_path}")

        params_path = os.path.join(self.out_dir, "best_params.json")
        with open(params_path, "w") as f:
            json.dump(
                {
                    "model_name": best.model_name,
                    "best_params": best.best_params,
                    "cv_f1": float(best.cv_f1),
                },
                f,
                indent=2,
            )
        print(f"Saved best params to: {params_path}")

        return metrics

def main(data="three_finger_cylinder.csv", out="models", test_size=0.25, cv=5) -> Dict:

    # Load data (composition: Orchestrator uses DataLoader)
    loader = DataLoader(data)
    df, X, y = loader.load()

    # Define the model specs to consider (inheritance: each is a BaseModelSpec)
    specs: List[BaseModelSpec] = [
        LogisticRegressionSpec(),
        RandomForestSpec(),
        KNNSpec(),
        SVMRBFSpec(),
        GradientBoostingSpec(),
    ]

    # Run training orchestration
    orchestrator = TrainingOrchestrator(
        specs=specs,
        cv_splits=cv,
        scoring="f1",
        out_dir=out,
        test_size=test_size,
    )
    metrics = orchestrator.run(X, y)
    print(
        f"Test metrics: accuracy={metrics['accuracy']:.3f}, "
        f"f1={metrics['f1']:.3f}, precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}"
    )


if __name__ == "__main__":
    main()