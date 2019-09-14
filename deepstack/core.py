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
URL_FACE_REGISTRATION = "http://{}:{}/v1/vision/face/register"
URL_FACE_RECOGNITION = "http://{}:{}/v1/vision/face/recognize"


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


def get_recognised_faces(predictions: List[Dict]) -> List[Dict]:
    """
    Get the recognised faces.
    """
    try:
        matched_faces = {
            face["userid"]: round(face["confidence"] * 100, 1)
            for face in predictions
            if not face["userid"] == "unknown"
        }
        return matched_faces
    except:
        return {}


def get_objects(predictions: List[Dict]) -> List[str]:
    """
    Get a list of the unique objects predicted.
    """
    labels = [pred["label"] for pred in predictions]
    return list(set(labels))


def get_object_confidences(predictions: List[Dict], target_object: str):
    """
    Return the list of confidences of instances of target label.
    """
    confidences = [
        pred["confidence"] for pred in predictions if pred["label"] == target_object
    ]
    return confidences


def get_objects_summary(predictions: List[Dict]):
    """
    Get a summary of the objects detected.
    """
    objects = get_objects(predictions)
    return {
        target_object: len(get_object_confidences(predictions, target_object))
        for target_object in objects
    }


def post_image(
    url: str, image_bytes: bytes, api_key: str, timeout: int, data: dict = {}
):
    """Post an image to Deepstack."""
    try:
        data["api_key"] = api_key
        response = requests.post(
            url, files={"image": image_bytes}, data=data, timeout=timeout
        )
        return response
    except requests.exceptions.Timeout:
        raise DeepstackException(
            f"Timeout connecting to Deepstack, current timeout is {timeout} seconds"
        )
    except requests.exceptions.ConnectionError as exc:
        raise DeepstackException(f"Connection error: {exc}")


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

    def detect(self, image_bytes: bytes):
        """Process image_bytes, performing detection."""
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

    def register_face(self, name: str, image_bytes: bytes):
        """
        Register a face name to a file.
        """

        response = post_image(
            url=URL_FACE_REGISTRATION.format(self._ip_address, self._port),
            image_bytes=image_bytes,
            api_key=self._api_key,
            timeout=self._timeout,
            data={"userid": name},
        )

        if response.status_code == 200 and response.json()["success"] == True:
            return
        elif response.status_code == 200 and response.json()["success"] == False:
            error = response.json()["error"]
            raise DeepstackException(f"Error from Deepstack: {error}")

    def recognise(self, image_bytes: bytes):
        """Process image_bytes, performing recognition."""
        self._predictions = []
        url = URL_FACE_RECOGNITION.format(self._ip_address, self._port)

        response = post_image(url, image_bytes, self._api_key, self._timeout)

        if response.status_code == HTTP_OK:
            if response.json()["success"]:
                self._predictions = response.json()["predictions"]
            else:
                error = response.json()["error"]
                raise DeepstackException(f"Error from Deepstack: {error}")
