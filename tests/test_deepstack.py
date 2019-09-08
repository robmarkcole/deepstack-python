# Placeholder

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

MOCK_RESPONSE = {
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

MOCK_PREDICTIONS = MOCK_RESPONSE["predictions"]
MOCK_CONFIDENCES = [0.6998661, 0.7996547]
CONFIDENCE_THRESHOLD = 0.7


def test_DeepstackObject_process_image_bytes():
    """Test a good response from server."""
    with requests_mock.Mocker() as mock_req:
        mock_req.post(MOCK_URL, status_code=ds.HTTP_OK, json=MOCK_RESPONSE)

        dsobject = ds.DeepstackObject(MOCK_IP_ADDRESS, MOCK_PORT)
        dsobject.process_image_bytes(MOCK_BYTES)
        assert dsobject.predictions == MOCK_PREDICTIONS


def test_DeepstackObject_process_image_bytes_timeout():
    """Test a timeout. THIS SHOULD FAIL"""
    with pytest.raises(ds.DeepstackException) as excinfo:
        with requests_mock.Mocker() as mock_req:
            mock_req.post(MOCK_URL, exc=requests.exceptions.ConnectTimeout)
            dsobject = ds.DeepstackObject(MOCK_IP_ADDRESS, MOCK_PORT)
            dsobject.process_image_bytes(MOCK_BYTES)
            assert False
            assert "SHOULD FAIL" in str(excinfo.value)


def test_get_object_labels():
    """Cant always be sure order of returned list items."""
    object_labels = ds.get_object_labels(MOCK_PREDICTIONS)
    assert type(object_labels) is list
    assert "dog" in object_labels
    assert "person" in object_labels
    assert len(object_labels) == 2


def test_get_objects_summary():
    objects_summary = ds.get_objects_summary(MOCK_PREDICTIONS)
    assert objects_summary == {"dog": 1, "person": 2}


def test_get_label_confidences():
    label_confidences = ds.get_label_confidences(MOCK_PREDICTIONS, "person")
    assert label_confidences == MOCK_CONFIDENCES


def test_get_confidences_above_threshold():
    assert (
        len(ds.get_confidences_above_threshold(MOCK_CONFIDENCES, CONFIDENCE_THRESHOLD))
        == 1
    )
