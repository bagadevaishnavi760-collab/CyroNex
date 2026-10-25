import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model("superconductor_model.json")

print("Model loaded successfully!")
print("Expected number of features:", model.n_features_in_)

print("Feature names:", model.feature_names_in_
      if model.feature_names_in_ is not None
      else "Feature names not stored in model")