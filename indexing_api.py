from typing import Optional
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from utils.feature_indexing import get_arg
from utils.feature_indexing import IndexingRetrievalModel
from configs.config import init_config
from pathlib import Path
from PIL import Image

app = FastAPI()
my_cfg = init_config()


class Request(BaseModel):
    query: str


@app.post("/retrieval")
async def retrieval(req: Request):
    req = req.dict()
    query = req["query"]
    return {"result": retrieval_model.retrieval(query)}


@app.get("/CLIP/search")
def text_query(q: str):
    return retrieval_model.retrieval(text_query=q)


@app.get("/CLIP/visual_similar")
def find_visual_similar(
    dataset: str,
    shot_id: str,
    frame_id: str,
):
    keyframe_dir = Path(
        "/mlcv/Databases/VBS/Processed_Data/Thumbnail/TransNetV2_200x113")
    keyframe_path = keyframe_dir / dataset / \
        shot_id / f"{shot_id}_{frame_id}.png"
    if not keyframe_path.exists():
        return {}
    img = Image.open(str(keyframe_path))
    return retrieval_model.retrieval(image_query=img)


if __name__ == '__main__':
    retrieval_model = IndexingRetrievalModel(get_arg(), my_cfg)
    uvicorn.run(app, host="0.0.0.0", port=5050)
