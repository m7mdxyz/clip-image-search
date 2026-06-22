import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from embed import embed_image, embed_text
from vector_store import search

app = FastAPI()
app.mount("/data", StaticFiles(directory="data"), name="data")


class TextQuery(BaseModel):
    query: str


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/search/text")
def search_text(body: TextQuery):
    vector = embed_text(body.query)
    return {"results": search(vector)}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    vector = embed_image(image)
    return {"results": search(vector)}
