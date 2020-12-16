[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPI Version](https://img.shields.io/pypi/v/deepstack-python.svg)](https://pypi.org/project/deepstack-python/)

# deepstack-python
Unofficial python API for [DeepStack](https://python.deepstack.cc/). Provides classes for making requests to the object detection & face detection/recognition endpoints. Also includes some helper functions for processing the results. See the Jupyter notebooks for usage.

Run deepstack with all three endpoints active (CPU, noAVX mode):
```
docker run \
      -e VISION-SCENE=True \
      -e VISION-DETECTION=True \
      -e VISION-FACE=True \
      -v localstorage:/datastore \
      -p 5000:5000 \
      -e API-KEY="" \
      --name deepstack deepquestai/deepstack:noavx
```
Check deepstack is running using curl (from root of this repo):
```
curl -X POST -F image=@tests/images/test-image3.jpg 'http://localhost:5000/v1/vision/detection'
```
If all goes well you should see the following returned:
```
{"success":true,"predictions":[{"confidence":0.9998661,"label":"person","y_min":0,"x_min":258,"y_max":676,"x_max":485},{"confidence":0.9996547,"label":"person","y_min":0,"x_min":405,"y_max":652,"x_max":639},{"confidence":0.99745613,"label":"dog","y_min":311,"x_min":624,"y_max":591,"x_max":825}]}
```

## Development
* Create venv -> `python3.7 -m venv venv`
* Use venv -> `source venv/bin/activate`
* `pip3 install -r requirements.txt` and `pip3 install -r requirements-dev.txt`
* Run tests with `venv/bin/pytest tests/*`
* Black format with `venv/bin/black .`

## Jupyter
* Docs are created using Jupyter notebooks
* Install in venv with -> `pip3 install jupyterlab`
* Run -> `venv/bin/jupyter lab`
