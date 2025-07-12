from fastapi import FastAPI
from pydantic import BaseModel

from backend.python.model_inference.model_inference_pipeline import ModelInference

app = FastAPI()
model_inference = ModelInference()

class RequestDAO(BaseModel):
    question: str
    input: object

@app.get("/health")
def get_health():
    return {"status": "ok"}

@app.post("/image_question")
async def image_question(request_dao : RequestDAO):
    model_inference.inference(request_dao=request_dao)
    return {"question": request_dao.question}
