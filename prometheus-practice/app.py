from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Business endpoint
@app.get("/")
def root():
    return {"status": "ok"}

Instrumentator().instrument(app).expose(app, endpoint="/metrics")
