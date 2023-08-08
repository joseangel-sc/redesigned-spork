IMAGE_NAME = ploting-floats-converter

build:
	docker build -t $(IMAGE_NAME) .

terminal:
	docker run -it --rm -v $(PWD)/src:/app/src -v $(PWD)/tests:/app/tests $(IMAGE_NAME) /bin/bash

run:
	docker run -it --rm $(IMAGE_NAME) python src/ploting_floats.py

tests:
	docker run -it --rm $(IMAGE_NAME) python tests/src/your_test_script.py
