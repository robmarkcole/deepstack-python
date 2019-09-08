# Placeholder

import deepstack.core as ds

MOCK_RESPONSE ={'success': True,
    'predictions': [{'confidence': 0.6998661,
    'label': 'person',
    'y_min': 0,
    'x_min': 258,
    'y_max': 676,
    'x_max': 485},
    {'confidence': 0.7996547,
    'label': 'person',
    'y_min': 0,
    'x_min': 405,
    'y_max': 652,
    'x_max': 639},
    {'confidence': 0.59745613,
    'label': 'dog',
    'y_min': 311,
    'x_min': 624,
    'y_max': 591,
    'x_max': 825}]}

MOCK_PREDICTIONS = MOCK_RESPONSE['predictions']

def test_get_object_labels():
    object_labels = ds.get_object_labels(MOCK_PREDICTIONS)
    assert 'dog' in object_labels
    assert 'person' in object_labels
    assert len(object_labels) == 2