FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . /app

CMD ["python", "src/ploting_floats.py"]
