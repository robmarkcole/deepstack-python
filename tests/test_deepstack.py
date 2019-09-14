import deepstack.core as ds
import requests
import requests_mock
import pytest

MOCK_IP_ADDRESS = "localhost"
MOCK_PORT = 5000
MOCK_URL = "http://{}:{}/v1/vision/detection".format(MOCK_IP_ADDRESS, MOCK_PORT)

MOCK_BYTES = b"Test"
MOCK_API_KEY = "mock_api_key"
MOCK_TIMEOUT = 8

MOCK_OBJECT_DETECTION_RESPONSE = {
    "success": True,
    "predictions": [
        {
            "confidence": 0.6998661,
            "label": "person",
            "y_min": 0,
            "x_min": 258,
            "y_max": 676,
            "x_max": 485,
        },
        {
            "confidence": 0.7996547,
            "label": "person",
            "y_min": 0,
            "x_min": 405,
            "y_max": 652,
            "x_max": 639,
        },
        {
            "confidence": 0.59745613,
            "label": "dog",
            "y_min": 311,
            "x_min": 624,
            "y_max": 591,
            "x_max": 825,
        },
    ],
}

MOCK_FACE_RECOGNITION_RESPONSE = {
    "success": True,
    "predictions": [
        {
            "confidence": 0.74999994,
            "userid": "Idris Elba",
            "y_min": 176,
            "x_min": 209,
            "y_max": 825,
            "x_max": 677,
        },
        {
            "confidence": 0,
            "userid": "unknown",
            "y_min": 230,
            "x_min": 867,
            "y_max": 729,
            "x_max": 1199,
        },
    ],
}

MOCK_OBJECT_PREDICTIONS = MOCK_OBJECT_DETECTION_RESPONSE["predictions"]
MOCK_OBJECT_CONFIDENCES = [0.6998661, 0.7996547]
CONFIDENCE_THRESHOLD = 0.7
MOCK_RECOGNISED_FACES = {"Idris Elba": 75.0}


def test_DeepstackObject_detect():
    """Test a good response from server."""
    with requests_mock.Mocker() as mock_req:
        mock_req.post(
            MOCK_URL, status_code=ds.HTTP_OK, json=MOCK_OBJECT_DETECTION_RESPONSE
        )

        dsobject = ds.DeepstackObject(MOCK_IP_ADDRESS, MOCK_PORT)
        dsobject.detect(MOCK_BYTES)
        assert dsobject.predictions == MOCK_OBJECT_PREDICTIONS


def test_DeepstackObject_detect_timeout():
    """Test a timeout. THIS SHOULD FAIL"""
    with pytest.raises(ds.DeepstackException) as excinfo:
        with requests_mock.Mocker() as mock_req:
            mock_req.post(MOCK_URL, exc=requests.exceptions.ConnectTimeout)
            dsobject = ds.DeepstackObject(MOCK_IP_ADDRESS, MOCK_PORT)
            dsobject.detect(MOCK_BYTES)
            assert False
            assert "SHOULD FAIL" in str(excinfo.value)


def test_get_objects():
    """Cant always be sure order of returned list items."""
    objects = ds.get_objects(MOCK_OBJECT_PREDICTIONS)
    assert type(objects) is list
    assert "dog" in objects
    assert "person" in objects
    assert len(objects) == 2


def test_get_objects_summary():
    objects_summary = ds.get_objects_summary(MOCK_OBJECT_PREDICTIONS)
    assert objects_summary == {"dog": 1, "person": 2}


def test_get_object_confidences():
    object_confidences = ds.get_object_confidences(MOCK_OBJECT_PREDICTIONS, "person")
    assert object_confidences == MOCK_OBJECT_CONFIDENCES


def test_get_confidences_above_threshold():
    assert (
        len(
            ds.get_confidences_above_threshold(
                MOCK_OBJECT_CONFIDENCES, CONFIDENCE_THRESHOLD
            )
        )
        == 1
    )


def test_get_recognised_faces():
    predictions = MOCK_FACE_RECOGNITION_RESPONSE["predictions"]
    assert ds.get_recognised_faces(predictions) == MOCK_RECOGNISED_FACES
