Focus Bear retrained Android ONNX assets

Source: FocusBear_Final_Iterations/06_Disagreement_Manual_Review/ensemble_disagreement_retrained.pkl
Feature width: 5545
Ensemble weights used by Android: LR=1, XGB=2, XGB tuned=2

Files:
- logistic_regression.onnx
- xgboost.onnx
- xgboost_tuned.onnx

These files expose a float tensor input named `input` with shape [N,5545] and a float tensor output named `probabilities` with shape [N,3], matching the Android RelevanceOnnxModel integration.

Important: the model package was pickled with a newer scikit-learn version than the export environment, so the resulting files should be validated in Android against the source pickle before replacing production assets.
