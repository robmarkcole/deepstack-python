[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPI Version](https://img.shields.io/pypi/v/deepstack-python.svg)](https://pypi.org/project/deepstack-python/)
[![build status](http://img.shields.io/travis/robmarkcole/deepstack-python/master.svg?style=flat)](https://travis-ci.org/robmarkcole/deepstack-python)

# deepstack-python
Unofficial python API for [DeepStack](https://python.deepstack.cc/). Provides class for making requests to the object detection endpoint, and functions for processing the result. See the Jupyter notebooks for usage.
Run deepstack (CPU, noAVX mode):
```
docker run -e VISION-DETECTION=True -e VISION-FACE=True -e MODE=High -d \
      -v localstorage:/datastore -p 5000:5000 \
      -e API-KEY="Mysecretkey" \
       --name deepstack deepquestai/deepstack:noavx
```

## Development
* Use `venv` -> `source venv/bin/activate`
* `pip install -r requirements-dev.txt`
* Run tests with `venv/bin/pytest tests/*`
* Black format with `venv/bin/black deepstack/core.py` and `venv/bin/black tests/test_deepstack.py`
