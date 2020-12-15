"""
Deepstack core.
"""
import requests
from PIL import Image
from typing import Union, List, Set, Dict

## Const
DEFAULT_API_KEY = ""
DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_IP = "localhost"
DEFAULT_PORT = 80

## HTTP codes
HTTP_OK = 200
BAD_URL = 404

## API urls
URL_BASE_VISION = "http://{ip}:{port}/v1/vision"
URL_CUSTOM = "/custom/{custom_model}"
URL_OBJECT_DETECTION = "/detection"
URL_FACE_DETECTION = "/face"
URL_FACE_REGISTER = "/face/register"
URL_FACE_RECOGNIZE = "/face/recognize"
URL_SCENE_DETECTION = "/scene"


class DeepstackException(Exception):
    pass


def format_confidence(confidence: Union[str, float]) -> float:
    """
    Takes a confidence from the API like
    0.55623 and returns 55.6 (%).
    """
    DECIMALS = 1
    return round(float(confidence) * 100, DECIMALS)


def get_confidences_above_threshold(
    confidences: List[float], confidence_threshold: float
) -> List[float]:
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
    return sorted(list(set(labels)))


def get_object_confidences(predictions: List[Dict], target_object: str) -> List[float]:
    """
    Return the list of confidences of instances of target label.
    """
    confidences = [
        float(pred["confidence"])
        for pred in predictions
        if pred["label"] == target_object
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
    """Post an image to Deepstack. Only handles excpetions."""
    try:
        data["api_key"] = api_key  # Insert the api_key
        return requests.post(
            url, files={"image": image_bytes}, data=data, timeout=timeout
        )
    except requests.exceptions.Timeout:
        raise DeepstackException(
            f"Timeout connecting to Deepstack, current timeout is {timeout} seconds"
        )
    except requests.exceptions.ConnectionError as exc:
        raise DeepstackException(f"Connection error: {exc}")


def process_image(url: str, image_bytes: bytes, api_key: str, timeout: int):
    """Process image_bytes and detect. Handles common status codes"""
    response = post_image(url, image_bytes, api_key, timeout)
    if response.status_code == HTTP_OK:
        return response.json()
    elif response.status_code == BAD_URL:
        raise DeepstackException(f"Bad url supplied, url {url} raised error {BAD_URL}")
    else:
        raise DeepstackException(
            f"Error from Deepstack request, status code: {response.status_code}"
        )


class DeepstackVision:
    """Base class for Deepstack vision."""

    def __init__(
        self,
        ip: str = DEFAULT_IP,
        port: int = DEFAULT_PORT,
        api_key: str = DEFAULT_API_KEY,
        timeout: int = DEFAULT_TIMEOUT,
        url_detect: str = "",
        url_recognize: str = "",
        url_register: str = "",
    ):

        self._url_base = URL_BASE_VISION.format(ip=ip, port=port)
        self._url_detect = self._url_base + url_detect
        self._url_recognize = self._url_base + url_recognize
        self._url_register = self._url_base + url_register
        self._api_key = api_key
        self._timeout = timeout

    def detect(self):
        """Process image_bytes and detect."""
        raise NotImplementedError

    def recognize(self):
        """Process image_bytes and recognize."""
        raise NotImplementedError

    def register(self):
        """Perform a registration."""
        raise NotImplementedError


class DeepstackObject(DeepstackVision):
    """Work with objects"""

    def __init__(
        self,
        ip: str = DEFAULT_IP,
        port: int = DEFAULT_PORT,
        api_key: str = DEFAULT_API_KEY,
        timeout: int = DEFAULT_TIMEOUT,
        custom_model: str = None,
    ):
        if not custom_model:
            super().__init__(
                ip, port, api_key, timeout, url_detect=URL_OBJECT_DETECTION,
            )
        else:
            super().__init__(
                ip,
                port,
                api_key,
                timeout,
                url_detect=URL_CUSTOM.format(custom_model=custom_model),
            )

    def detect(self, image_bytes: bytes):
        """Process image_bytes and detect."""
        response_json = process_image(
            self._url_detect, image_bytes, self._api_key, self._timeout
        )
        return response_json["predictions"]


class DeepstackScene(DeepstackVision):
    """Work with scenes"""

    def __init__(
        self,
        ip: str,
        port: str,
        api_key: str = DEFAULT_API_KEY,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__(
            api_key, timeout, url_detect=URL_SCENE_DETECTION,
        )

    def detect(self, image_bytes: bytes):
        """Process image_bytes and detect."""
        response_json = process_image(self, image_bytes, self._api_key, self._timeout)
        return response_json


class DeepstackFace(DeepstackVision):
    """Work with objects"""

    def __init__(
        self,
        ip: str,
        port: str,
        api_key: str = DEFAULT_API_KEY,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__(
            api_key,
            timeout,
            url_detect=URL_FACE_DETECTION.format(ip=self._ip, port=self._port),
            url_register=URL_FACE_REGISTER.format(ip=self._ip, port=self._port),
            url_recognize=URL_FACE_RECOGNIZE.format(ip=self._ip, port=self._port),
        )

    def detect(self, image_bytes: bytes):
        """Process image_bytes and detect."""
        response_json = process_image(self, image_bytes, self._api_key, self._timeout)
        return response_json

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

        response = post_image(
            self._url_recognize, image_bytes, self._api_key, self._timeout
        )

        self._response = response.json()
        if not self._response["success"]:
            error = self._response["error"]
            raise DeepstackException(f"Error from Deepstack: {error}")
