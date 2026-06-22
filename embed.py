from PIL import Image
from sentence_transformers import SentenceTransformer

# Two text/image towers trained to land in the same 512-dim CLIP vector space.
image_model = SentenceTransformer("clip-ViT-B-32")
text_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1")


def embed_image(image: Image.Image) -> list[float]:
    return image_model.encode(image).tolist()


def embed_text(text: str) -> list[float]:
    return text_model.encode(text).tolist()
