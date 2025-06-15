ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "main.py"]