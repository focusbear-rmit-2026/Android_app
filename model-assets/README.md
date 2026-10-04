# Android relevance model assets

Required runtime files:

```text
logistic_regression.onnx
xgboost.onnx
xgboost_tuned.onnx
tfidf_goal.json
tfidf_context.json
minilm/model.onnx
minilm/vocab.txt
```

The TF-IDF JSON files are included here. Copy the ONNX/MiniLM binary assets from the model repository by running the root script:

```powershell
.\scripts\sync-android-relevance-assets.ps1 -ModelRepo "C:\path\to\FocusBear-Distraction-Detection-Model"
```

Check the asset set with:

```powershell
.\scripts\check-android-relevance-assets.ps1
```

Expected approximate binary sizes from the validated export are:

```text
logistic_regression.onnx   ~84 KB
xgboost.onnx               ~713 KB
xgboost_tuned.onnx         ~2.0 MB
minilm/model.onnx          ~90 MB
```

If the model repository does not already contain `deployment/android_assets`, export directly with:

```powershell
.\scripts\export-and-sync-android-relevance-assets.ps1 -ModelRepo "C:\path\to\FocusBear-Distraction-Detection-Model"
```
