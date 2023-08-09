IMAGE_NAME = ploting-floats-converter

build:
	docker build -t $(IMAGE_NAME) .

terminal:
	docker run -it --rm -v $(PWD)/src:/app/src -v $(PWD)/tests:/app/tests $(IMAGE_NAME) /bin/bash

run:
	docker run -it --rm $(IMAGE_NAME) python /app/src/ploting_floats.py

testing:
	docker run -it --rm -e PYTHONPATH=/app -v $(PWD)/src:/app/src -v $(PWD)/tests:/app/tests $(IMAGE_NAME) pytest -s /app/tests/src/test_ploting_floats.py
