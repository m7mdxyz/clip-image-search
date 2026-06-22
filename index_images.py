from pathlib import Path

from PIL import Image

from embed import embed_image
from vector_store import upsert_image

IMAGES_DIR = Path("data/images")

if __name__ == "__main__":
    paths = sorted(IMAGES_DIR.glob("*.jpg"))
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        vector = embed_image(image)
        upsert_image(image_id=path.stem, vector=vector, path=path.as_posix())
        if i % 100 == 0:
            print(f"{i}/{len(paths)} indexed")

    print(f"Done. Indexed {len(paths)} images.")
