"""
Deepstack core.
"""
import requests
from PIL import Image

## Const
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
TIMEOUT = 30  # seconds


def format_confidence(confidence):
    """Takes a confidence from the API like 
       0.55623 and returne 55.6 (%).
    """
    return round(float(confidence)*100, 1)


def get_matched_faces(predictions: dict):
    """
    Get the predicted faces and their confidence.
    """
    matched_faces = {}
    matched_faces = {
        face["userid"]: format_confidence(face["confidence"])
        for face in predictions
        if not face["userid"] == "unknown"
    }
    return matched_faces


def post_image(url: str, image: bytes):
    """Post an image to the classifier."""
    response = requests.post(url, files={"image": image}, timeout=TIMEOUT)
    return response


class DeepstackFace:
    """Work with faces."""

    def __init__(self, ip_address: str, port: str):

        self._url_check = "http://{}:{}/v1/vision/face/recognize".format(
            ip_address, port
        )

        self._faces = None
        self._matched = {}

    def register_face(self, file_path: str, userid: str):
        """Register a face with Deepstack."""

    def process_file(self, file_path: str):
        """Process an image file."""
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
