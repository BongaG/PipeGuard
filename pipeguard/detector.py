import os
import json
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from .simulator import EVENT_CLASSES
from .config import Config


class StaticThresholdDetector:
    def __init__(self, nominal, drop_kpa=15.0):
        self.nominal = np.asarray(nominal)
        self.drop_kpa = drop_kpa

    def predict_alarm(self, pressure_now):
        return bool(((self.nominal - pressure_now) > self.drop_kpa).any())


class LeakDetector:
    def __init__(self, pipes):
        self.pipes = list(pipes)
        self.event_xgb = None
        self.event_mlp = None
        self.loc_xgb = None
        self.loc_mlp = None

    def fit(self, X, y_event, y_pipe, X_loc=None, seed=42):
        X_loc = X if X_loc is None else X_loc
        self.event_xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, random_state=seed, n_jobs=4, eval_metric="mlogloss")
        self.event_xgb.fit(X, y_event)
        self.event_mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True, random_state=seed))
        self.event_mlp.fit(X, y_event)
        mask = np.isin(y_event, [EVENT_CLASSES.index("leak"), EVENT_CLASSES.index("burst")])
        yp = np.array([self.pipes.index(p) for p in y_pipe[mask]])
        self.loc_xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, random_state=seed, n_jobs=4, eval_metric="mlogloss")
        self.loc_xgb.fit(X_loc[mask], yp)
        self.loc_mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=seed))
        self.loc_mlp.fit(X_loc[mask], yp)
        return self

    def predict_event(self, X, model="xgb"):
        m = self.event_xgb if model == "xgb" else self.event_mlp
        return m.predict(np.atleast_2d(X))

    def predict_event_proba(self, X, model="xgb"):
        m = self.event_xgb if model == "xgb" else self.event_mlp
        return m.predict_proba(np.atleast_2d(X))

    def predict_pipe(self, X, model="mlp"):
        m = self.loc_mlp if model == "mlp" else self.loc_xgb
        idx = m.predict(np.atleast_2d(X))
        return [self.pipes[i] for i in idx]

    def predict_pipe_proba(self, X, model="mlp"):
        m = self.loc_mlp if model == "mlp" else self.loc_xgb
        return m.predict_proba(np.atleast_2d(X))

    def save(self, directory=Config.MODEL_DIR):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self, os.path.join(directory, "leak_detector.joblib"))

    @staticmethod
    def load(directory=Config.MODEL_DIR):
        path = os.path.join(directory, "leak_detector.joblib")
        if not os.path.exists(path):
            return None
        return joblib.load(path)
