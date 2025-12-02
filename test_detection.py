import random
import os
from pathlib import Path
from PIL import Image, ImageDraw

from face_detection import face_detector

from argparse import Namespace

face_detection_args = Namespace(
	mtcnn=True,
	haar=False,
	retina=False,
	debug=False,
)

def get_active_detector(args):
    if getattr(args, 'mtcnn', False):
        return 'mtcnn'
    if getattr(args, 'haar', False):
        return 'haar'
    if getattr(args, 'retina', False):
        return 'retina'
    return 'unknown'

def run_and_save(dataset_name: str, img_paths: list):
    save_dir = Path("vis") / get_active_detector(face_detection_args) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        filename = img_path.name

        detections = face_detector(face_detection_args, img)

        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)

        for i, (face_tensor, box) in enumerate(detections):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            face_np = (
                face_tensor
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .clamp(0, 255)
                .byte()
                .numpy()
            )

            face_pil = Image.fromarray(face_np, mode="RGB")
            try:
                face_pil.save(save_dir / f"crop_{i}_{filename}")
            except:
                print('-' * 20, i)
                pass

        vis_img.save(save_dir / f"vis_{filename}")
        print(f"Processed: {img_path}")

def get_random_images(folder: Path, n: int = 5):
    all_imgs = list(folder.rglob("*.jpg"))
    return random.sample(all_imgs, n)

if __name__ == "__main__":
    random.seed(42)
    datasets = ['lfw-deepfunneled', 'img_align_celeba']
    for dataset in datasets:
        folder = Path(f"data/{dataset}")
        random_imgs = get_random_images(folder, 5)
        run_and_save(dataset, random_imgs)