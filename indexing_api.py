import faiss
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from utils.feature_indexing import get_arg
from utils.feature_indexing import IndexingRetrievalModel
from configs.config import init_config

app = FastAPI()
my_cfg = init_config()

class Request(BaseModel):
    query: str

@app.post("/retrieval")
def retrieval(req: Request):
    req = req.dict()
    query = req["query"]
    return {"result": retrieval_model.retrieval(query)}

if __name__ == '__main__':
    retrieval_model = IndexingRetrievalModel(get_arg(), my_cfg)
    uvicorn.run(app, host="0.0.0.0", port=5000)

