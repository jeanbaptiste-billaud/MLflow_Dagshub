"""BentoML service exposing the BUSI breast ultrasound classifier."""

import bentoml
import torch
from PIL.Image import Image

from preprocess import CLASSES, preprocess

MODEL_TAG = "busi_resnet18:latest"


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class BusiClassifier:
    # Declaring the model as a class attribute packages it into the Bento.
    bento_model = bentoml.models.BentoModel(MODEL_TAG)

    def __init__(self) -> None:
        # The model was logged by MLflow as a pt2 archive (torch.export): it is
        # loaded with torch only, no pickle and no MLflow needed at runtime.
        # Eval mode is baked into the exported graph (`.eval()` is not supported).
        pt2_path = self.bento_model.path_of("mlflow_model/data/model.pt2")
        self.model = torch.export.load(pt2_path).module()

    @bentoml.api
    def predict(self, image: Image) -> dict:
        """Classify one ultrasound image (PNG/JPEG upload)."""
        x = preprocess(image)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        best = int(probs.argmax())
        return {
            "label": CLASSES[best],
            "confidence": round(float(probs[best]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        }

    @bentoml.api
    def model_info(self) -> dict:
        """Return the BentoML model tag and the MLflow run it comes from."""
        return {"bento_model": str(self.bento_model.tag),
                **{k: str(v) for k, v in self.bento_model.info.metadata.items()}}
