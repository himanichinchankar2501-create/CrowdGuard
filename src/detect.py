from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best (1).pt"


def load_model():

    print("🔍 Loading CrowdGuard model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CrowdGuard model not found at:\n{MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    print("✅ CrowdGuard model loaded!")
    print(f"📍 {MODEL_PATH}")
    print(f"🧠 Model classes: {model.names}")

    return model


def detect_people(model, image):

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    people_count = 0

    for result in results:

        if result.boxes is None:
            continue

        for cls in result.boxes.cls:

            class_id = int(cls)

            if class_id == 0:
                people_count += 1

    annotated_image = results[0].plot()

    return people_count, annotated_image