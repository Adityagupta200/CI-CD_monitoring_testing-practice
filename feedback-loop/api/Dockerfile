FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "app.py", "--host", "0.0.0.0", "--port", "8000"]