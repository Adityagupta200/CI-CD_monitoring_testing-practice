ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && apt-get install -y python${PYTHON_VERSION} python3-pip

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python3", "main.py"]