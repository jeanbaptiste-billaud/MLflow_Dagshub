# BUSI classifier – BentoML service

Serves the best PyTorch model (ResNet-18, pt2 format) trained in the MLflow
notebook. The Bento packages the model, the preprocessing and the API code.

## Files

| File | Role |
|---|---|
| `bentoml/preprocess.py` | Inference transforms (resize, tensor, ImageNet normalization). Single source of truth, must match the notebook `eval_tf`. |
| `bentoml/import_model.py` | Downloads the model from the MLflow server into the local BentoML store. |
| `bentoml/service.py` | BentoML service: `/predict` (image upload) and `/model_info`. |
| `bentoml/bentofile.yaml` | Build configuration (included files, Python dependencies). |
| `bentoml/requirements.txt` | Runtime dependencies of the Bento. |
| `bentoml/test_client.py` | Calls a running service with an image. |

## 1. Import the model from MLflow

Requires `bentoml`, `mlflow`, `torch` and `torchvision` in the build environment.

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<dagshub user>
export MLFLOW_TRACKING_PASSWORD=<dagshub token>

cd bentoml
python import_model.py                       # best "torch*" run by test_f1_macro
# or: python import_model.py --model-uri runs:/<run_id>/model
bentoml models list
```

The credentials are only needed here: the model is copied into the Bento,
the service never talks to MLflow.

## 2. Build

```bash
bentoml build            # from this folder
# or, from the repo root: bentoml build bentoml
bentoml list
```

## 3. Serve and test

```bash
bentoml serve busi_classifier:latest --port 3000
python test_client.py path/to/image.png
curl -X POST http://localhost:3000/predict -F "image=@path/to/image.png"
```

Swagger UI: http://localhost:3000

## 4. Ship

```bash
bentoml export busi_classifier:latest busi_classifier.bento   # portable archive (model included)
bentoml containerize busi_classifier:latest                   # requires Docker
```
