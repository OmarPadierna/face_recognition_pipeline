"""
Face Detection Module
Provides face detection functionality with multiple algorithm flavors.
"""

from typing import List
import torch
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np

def face_detector(face_detection_args, image) -> List[tuple]:
    """
    Selects and runs the appropriate face detection algorithm.

    Args:
        face_detection_args: Namespace with face detection flags (mtcnn, haar)
        image: Input image

    Returns:
        List of tuples (face_tensor, bounding_box) from the selected detection algorithm
    """
    if face_detection_args.mtcnn:
        is_debug_enabled = getattr(face_detection_args, 'debug', False)
        return face_detection_mtcnn(image, is_debug_enabled)
    elif face_detection_args.haar:
        return face_detection_haar(image)
    elif face_detection_args.retina:
        return face_detection_retina(image)
    return []

haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def face_detection_haar(image: Image.Image) -> List[tuple]:
    """
    Haar Cascade face detection.

    Args:
        image: PIL Image object

    Returns:
        List of tuples (face_tensor, bounding_box)
        where bounding_box = [x1, y1, x2, y2]
    """

    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    results = []
    for (x, y, w, h) in faces:
        face_crop = img_np[y:y+h, x:x+w]
        face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float()
        box = [float(x), float(y), float(x + w), float(y + h)]
        results.append((face_tensor, box))

    return results


# Modify here to include bounding boxes. Output should be a list of tuples. Tensor + bounding box. This way they can be used for labeling.
def face_detection_mtcnn(image: Image.Image, is_debug_enabled: bool) -> List[tuple]:
    """MTCNN face detection implementation.

    Args:
        image: PIL Image object
        is_debug_enabled: A flag that enables/disabled visualization and logs for debugging purposes
    Returns:
        List of tuples (face_tensor, bounding_box) where bounding_box is [x1, y1, x2, y2]
    """
    mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cpu')

    # Display original image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')

    # Detect faces and get bounding boxes
    # First call to get boxes and probs
    boxes, probs = mtcnn.detect(image)

    # Second call to get aligned face tensors
    faces = mtcnn(image)

    if is_debug_enabled:
        # Visualize detected faces
        if faces is not None and len(faces) > 0:
            num_faces = len(faces)
            fig, axes = plt.subplots(1, num_faces, figsize=(4 * num_faces, 4))

            # Handle single face case (axes won't be an array)
            if num_faces == 1:
                axes = [axes]

            for i, face in enumerate(faces):
                axes[i].imshow(face.permute(1, 2, 0).int().numpy())
                axes[i].axis('off')
                axes[i].set_title(f'Face {i + 1}')

            fig.show()
    # Note: MTCNN does face alignment automatically.
    if faces is not None and boxes is not None:
        # Return list of tuples: (face_tensor, bounding_box)
        return [(faces[i], boxes[i].tolist()) for i in range(len(faces))]
    else:
        return []



app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1) # -1 for CPU

def face_detection_retina(image: Image.Image):
    img_np = np.array(image.convert("RGB"))
    faces = app.get(img_np)

    results = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int).tolist()
        face_crop = img_np[y1:y2, x1:x2]
        face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float()
        box = [float(x1), float(y1), float(x2), float(y2)]
        results.append((face_tensor, box))
    return results