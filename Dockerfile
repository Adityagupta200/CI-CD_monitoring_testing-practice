ARG BASE_IMAGE=ubuntu
ARG PYTHON_VERSION=3.10

FROM ${BASE_IMAGE}:latest

RUN apt-get update && apt-get install -y python${PYTHON_VERSION} python3-pip

WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]