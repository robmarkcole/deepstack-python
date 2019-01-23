"""
Deepstack core.
"""
import imghdr
import requests
from PIL import Image

## Const
CLASSIFIER = "deepstack"
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
TIMEOUT = 30  # seconds


def get_matched_faces(predictions: dict):
    """
    Get the predicted faces and their confidence.
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


def is_valid_image(file_path: str):
    """
    Check file_path is valid image, using PIL then imghdr.
    """
    try:
        with Image.open(file_path):
            pass

        image_extension = imghdr.what(file_path)
        if image_extension in ["jpeg", ".jpg", ".png"]:
            return True
        return False
    except Exception as exc:
        print(exc)
        return False


def post_image(url: str, image: bytes):
    """Post an image to the classifier."""
    try:
        response = requests.post(url, files={"image": image}, timeout=TIMEOUT)
        return response
    except requests.exceptions.ConnectionError:
        print("ConnectionError: Is %s running?", CLASSIFIER)
        return None
    except requests.exceptions.Timeout:
        print("Timeout error from %s", CLASSIFIER)
        return None


class DeepstackFace:
    """Work with faces."""

    def __init__(self, ip_address: str, port: str):

        self._url_check = "http://{}:{}/v1/vision/face/recognize".format(
            ip_address, port
        )

        self._faces = None
        self._matched = {}

    def process_file(self, file_path: str):
        """Process an image file."""
        if is_valid_image(file_path):
            with open(file_path, "rb") as image_bytes:
                self.process_image_bytes(image_bytes)

    def process_image_bytes(self, image_bytes: bytes):
        """Process an image."""
        response = post_image(self._url_check, image_bytes)
        if response:
            if response.status_code == HTTP_OK:
                predictions_json = response.json()["predictions"]
                self._faces = len(predictions_json)
                self._matched = get_matched_faces(predictions_json)

        else:
            self._faces = None
            self._matched = {}

    @property
    def attributes(self):
        """Return the classifier attributes."""
        return {
            "faces": self._faces,
            "matched_faces": self._matched,
            "total_matched_faces": len(self._matched),
        }
