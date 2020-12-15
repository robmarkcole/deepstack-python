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
URL_CUSTOM = "http://{ip}:{port}/v1/vision/custom/{custom_model}"
URL_OBJECT_DETECTION = "http://{ip}:{port}/v1/vision/detection"
URL_FACE_DETECTION = "http://{ip}:{port}/v1/vision/face"
URL_FACE_REGISTER = "http://{ip}:{port}/v1/vision/face/register"
URL_FACE_RECOGNIZE = "http://{ip}:{port}/v1/vision/face/recognize"
URL_SCENE_DETECTION = "http://{ip}:{port}/v1/vision/scene"


def format_confidence(confidence: Union[str, float]) -> float:
    """Takes a confidence from the API like
    0.55623 and returns 55.6 (%).
    """
    return round(float(confidence) * 100, 1)


def get_confidences_above_threshold(confidences: List[float], confidence_threshold: float) -> List[float]:
    """Takes a list of confidences and returns those above a confidence_threshold."""
    return [val for val in confidences if val >= confidence_threshold]


def get_recognized_faces(predictions: List[Dict]) -> List[Dict]:
    """
    Get the recognized faces.
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
    confidences = [pred["confidence"] for pred in predictions if pred["label"] == target_object]
    return confidences


def get_objects_summary(predictions: List[Dict]):
    """
    Get a summary of the objects detected.
    """
    objects = get_objects(predictions)
    return {target_object: len(get_object_confidences(predictions, target_object)) for target_object in objects}


def post_image(url: str, image_bytes: bytes, api_key: str, timeout: int, data: dict = {}):
    """Post an image to Deepstack."""
    try:
        data["api_key"] = api_key  # Insert the api_key
        response = requests.post(url, files={"image": image_bytes}, data=data, timeout=timeout)
        return response
    except requests.exceptions.Timeout:
        raise DeepstackException(f"Timeout connecting to Deepstack, current timeout is {timeout} seconds")
    except requests.exceptions.ConnectionError as exc:
        raise DeepstackException(f"Connection error: {exc}")


class DeepstackException(Exception):
    pass


class Deepstack(object):
    """Base class for deepstack."""

    def __init__(
        self,
        api_key: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        url_detect: str = None,
        url_recognize: str = None,
        url_register: str = None,
    ):

        self._url_detect = url_detect
        self._url_recognize = url_recognize
        self._url_register = url_register
        self._api_key = api_key
        self._timeout = timeout
        self._response = None

    def detect(self, image_bytes: bytes):
        """Process image_bytes and detect."""
        self._response = None
        response = post_image(self._url_detect, image_bytes, self._api_key, self._timeout)

        if not response.status_code == HTTP_OK:
            raise DeepstackException(f"Error from request, status code: {response.status_code}")
            return

        self._response = response.json()
        if not self._response["success"]:
            error = self._response["error"]
            raise DeepstackException(f"Error from Deepstack: {error}")

    def recognize(self):
        """Process image_bytes and recognize."""
        raise NotImplementedError

    def register(self):
        """Perform a registration."""
        raise NotImplementedError

    @property
    def predictions(self):
        """Return the predictions."""
        raise NotImplementedError


class DeepstackObject(Deepstack):
    """Work with objects"""

    def __init__(
        self, ip: str, port: str, api_key: str = "", timeout: int = DEFAULT_TIMEOUT, custom_model: str = None,
    ):
        if not custom_model:
            super().__init__(
                api_key, timeout, url_detect=URL_OBJECT_DETECTION.format(ip=ip, port=port),
            )
        else:
            super().__init__(
                api_key, timeout, url_detect=URL_CUSTOM.format(ip=ip, port=port, custom_model=custom_model),
            )

    @property
    def predictions(self):
        """Return the predictions."""
        return self._response["predictions"]


class DeepstackScene(Deepstack):
    """Work with scenes"""

    def __init__(
        self, ip: str, port: str, api_key: str = "", timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__(
            api_key, timeout, url_detect=URL_SCENE_DETECTION.format(ip=self._ip, port=self._port),
        )

    @property
    def predictions(self):
        """Return the predictions."""
        return self._response


class DeepstackFace(Deepstack):
    """Work with objects"""

    def __init__(
        self, ip: str, port: str, api_key: str = "", timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__(
            api_key,
            timeout,
            url_detect=URL_FACE_DETECTION.format(ip=self._ip, port=self._port),
            url_register=URL_FACE_REGISTER.format(ip=self._ip, port=self._port),
            url_recognize=URL_FACE_RECOGNIZE.format(ip=self._ip, port=self._port),
        )

    @property
    def predictions(self):
        """Return the classifier attributes."""
        return self._response["predictions"]

    def register(self, name: str, image_bytes: bytes):
        """
        Register a face name to a file.
        """

        response = post_image(
            url=self._url_register,
            image_bytes=image_bytes,
            api_key=self._api_key,
            timeout=self._timeout,
            data={"userid": name},
        )

        if response.status_code == HTTP_OK and response.json()["success"] == True:
            return

        elif response.status_code == HTTP_OK and response.json()["success"] == False:
            error = response.json()["error"]
            raise DeepstackException(f"Error from Deepstack: {error}")

    def recognize(self, image_bytes: bytes):
        """Process image_bytes, performing recognition."""

        response = post_image(self._url_recognize, image_bytes, self._api_key, self._timeout)

        if not response.status_code == HTTP_OK:
            raise DeepstackException(f"Error from request, status code: {response.status_code}")
            return

        self._response = response.json()
        if not self._response["success"]:
            error = self._response["error"]
            raise DeepstackException(f"Error from Deepstack: {error}")
