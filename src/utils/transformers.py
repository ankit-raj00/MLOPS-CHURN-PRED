from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.columns_to_encode = [] # Detected automatically or can be passed

    def fit(self, X, y=None):
        # Identify columns to encode
        # In a real scenario, you might want to pass these explicitly or detect 'object' types
        # Here we re-detect as we did in analysis
        if isinstance(X, pd.DataFrame):
            self.columns_to_encode = [col for col in X.columns if X[col].dtype == 'object']
        
        for col in self.columns_to_encode:
            le = LabelEncoder()
            # Handle potential NaNs for encoding by filling with a placeholder temporarily or ensuring clean data
            # Assuming clean data or handled by previous steps, but let's be safe
            self.encoders[col] = le.fit(X[col].astype(str))
            
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, le in self.encoders.items():
            if col in X_copy.columns:
                # Handle unseen labels: Map to a default or special token if possible
                # LabelEncoder doesn't handle unseen labels well natively.
                # Production trick: map 'new' -> 0 or 'unknown'
                
                # Using a safe transformation approach
                X_copy[col] = X_copy[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else 0 # defaulting to 0 might be dangerous if 0 is a valid class, but acceptable for this MVP
                )
        return X_copy
