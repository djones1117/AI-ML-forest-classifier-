from pathlib import Path
from fastai.vision.all import load_learner, PILImage

# Path to the exported model (.pkl), relative to this file
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "forest_classifier.pkl"

_learner = None  # cached global


def get_learner():
    """
    Lazily load and cache the fastai learner.
    """
    global _learner
    if _learner is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _learner = load_learner(MODEL_PATH)
    return _learner


def predict_image(image_path: str):
    """
    Run prediction on a single image path.

    Returns a dict with:
      - label: predicted class label (str)
      - probs: dict of class -> probability
    """
    learner = get_learner()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # fastai can take the path directly, but we load via PILImage for clarity
    img = PILImage.create(img_path)

    pred, pred_idx, probs = learner.predict(img)
    vocab = learner.dls.vocab

    return {
        "label": str(pred),
        "probs": {str(vocab[i]): float(probs[i]) for i in range(len(vocab))},
    }
