"""
Deepstack core.
"""
import requests
from PIL import Image
from typing import Union, List, Set, Dict

## Const
HTTP_OK = 200
DEFAULT_TIMEOUT = 10  # seconds

## API urls
URL_OBJECT_DETECTION = "http://{}:{}/v1/vision/detection"
URL_FACE_DETECTION = "http://{}:{}/v1/vision/face"


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


def get_object_labels(predictions: List[Dict]) -> List[str]:
    """
    Get a list of the unique object labels predicted.
    """
    labels = [pred["label"] for pred in predictions]
    return list(set(labels))


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


def post_image(url: str, image: bytes, api_key: str, timeout: int):
    """Post an image to Deepstack."""
    try:
        response = requests.post(
            url, files={"image": image}, data={"api_key": api_key}, timeout=timeout
        )
        return response
    except requests.exceptions.Timeout:
        raise DeepstackException(
            f"Timeout connecting to Deepstack, current timeout is {timeout} seconds"
        )


class DeepstackException(Exception):
    pass


class Deepstack(object):
    """Base class for deepstack."""

    def __init__(
        self,
        ip_address: str,
        port: str,
        api_key: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        url_detection: str = "",
    ):

        self._ip_address = ip_address
        self._port = port
        self._url_detection = url_detection
        self._api_key = api_key
        self._timeout = timeout
        self._predictions = []

    def process_file(self, file_path: str):
        """Process an image file."""
        with open(file_path, "rb") as image_bytes:
            self.process_image_bytes(image_bytes)

    def process_image_bytes(self, image_bytes: bytes):
        """Process an image."""
        self._predictions = []
        url = self._url_detection.format(self._ip_address, self._port)

        response = post_image(url, image_bytes, self._api_key, self._timeout)

        if response.status_code == HTTP_OK:
            if response.json()["success"]:
                self._predictions = response.json()["predictions"]
            else:
                error = response.json()["error"]
                raise DeepstackException(f"Error from Deepstack: {error}")

    @property
    def predictions(self):
        """Return the classifier attributes."""
        return self._predictions


class DeepstackObject(Deepstack):
    """Work with objects"""

    def __init__(
        self,
        ip_address: str,
        port: str,
        api_key: str = "",
        timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__(
            ip_address, port, api_key, timeout, url_detection=URL_OBJECT_DETECTION
        )

class DeepstackFace(Deepstack):
    """Work with objects"""

    def __init__(
        self,
        ip_address: str,
        port: str,
        api_key: str = "",
        timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__(
            ip_address, port, api_key, timeout, url_detection=URL_FACE_DETECTION
        )
