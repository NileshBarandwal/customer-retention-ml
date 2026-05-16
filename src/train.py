"""
Model training entrypoint.

Loads processed data, defines the XGBoost/sklearn pipeline, runs hyperparameter
tuning, and logs experiments (metrics, parameters, artifacts) to MLflow.
"""
