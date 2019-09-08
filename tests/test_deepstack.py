# Placeholder

import deepstack.core as ds

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


def test_get_object_labels():
    """Cant always be sure order of returned list items."""
    object_labels = ds.get_object_labels(MOCK_PREDICTIONS)
    assert type(object_labels) is list
    assert "dog" in object_labels
    assert "person" in object_labels
    assert len(object_labels) == 2

def test_get_objects_summary():
    objects_summary = ds.get_objects_summary(MOCK_PREDICTIONS)
    assert objects_summary == {'dog': 1, 'person': 2}

def test_get_label_confidences():
    label_confidences = ds.get_label_confidences(MOCK_PREDICTIONS, 'person')
    assert label_confidences == MOCK_CONFIDENCES


def test_get_confidences_above_threshold():
    assert len(ds.get_confidences_above_threshold(MOCK_CONFIDENCES, CONFIDENCE_THRESHOLD)) == 1
