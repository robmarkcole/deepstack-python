"""
Deepstack core.
"""
import requests
from PIL import Image
from typing import Union, List, Set, Dict

## Const
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
TIMEOUT = 20  # seconds


def format_confidence(confidence: Union[str, float]) -> float:
    """Takes a confidence from the API like 
       0.55623 and returne 55.6 (%).
    """
    return round(float(confidence) * 100, 1)


def get_confidences_above_threshold(
    confidences: List[float], confidence_threshold: float
) -> List[float]:
    """Takes a list of confidences and returns those above a confidence_threshold."""
    return [val for val in confidences if val >= confidence_threshold]


def get_object_labels(predictions: List[Dict]) -> Set[str]:
    """
    Get a list of the unique object labels predicted.
    """
    labels = [pred["label"] for pred in predictions]
    return set(labels)


def get_label_confidences(predictions: List[Dict], target_label: str):
    """
    Return the list of confidences of instances of target label.
    """
    confidences = [
        pred["confidence"] for pred in predictions if pred["label"] == target_label
    ]
    return confidences


def get_objects_summary(predictions: List[Dict]):
    """
    Get a summary of the objects detected.
    """
    labels = get_object_labels(predictions)
    return {
        label: len(get_label_confidences(predictions, target_label=label))
        for label in labels
    }


def post_image(url: str, image: bytes):
    """Post an image to Deepstack."""
    response = requests.post(url, files={"image": image}, timeout=TIMEOUT)
    return response


class DeepstackObject:
    """The object detection API locates and classifies 80 
    different kinds of objects in a single image.."""

    def __init__(self, ip_address: str, port: str):

        self._url_object_detection = "http://{}:{}/v1/vision/detection".format(
            ip_address, port
        )
        self._predictions = []

    def process_file(self, file_path: str):
        """Process an image file."""
        with open(file_path, "rb") as image_bytes:
            self.process_image_bytes(image_bytes)

    def process_image_bytes(self, image_bytes: bytes):
        """Process an image."""
        self._predictions = []

        response = post_image(self._url_object_detection, image_bytes)
        if response:
            if response.status_code == HTTP_OK:
                self._predictions = response.json()["predictions"]

    @property
    def predictions(self):
        """Return the classifier attributes."""
        return self._predictions
