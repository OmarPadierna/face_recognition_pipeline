"""
Face Detection Module
Provides face detection functionality with multiple algorithm flavors.
"""

from typing import List
import torch
# from facenet_pytorch import MTCNN
# from insightface.app import FaceAnalysis
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


from typing import List, Tuple
from PIL import Image
import numpy as np
import torch
from mtcnn import MTCNN
from matplotlib import pyplot as plt


def face_detection_mtcnn(image: Image.Image, is_debug_enabled: bool) -> List[Tuple[torch.Tensor, list]]:
    """MTCNN face detection implementation (using `mtcnn` package).

    Args:
        image: PIL Image object
        is_debug_enabled: Enables/disables visualization and logs for debugging purposes

    Returns:
        List of tuples (face_tensor, bounding_box)
        where:
          - face_tensor is a CHW PyTorch tensor
          - bounding_box is [x1, y1, x2, y2]
    """
    # Create detector (CPU in this example; use "GPU:0" if you have TF+GPU set up)
    detector = MTCNN(device="CPU:0")

    # Convert PIL image to numpy array (H, W, C)
    image_np = np.array(image)

    # Display original image (as before)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.axis('off')

    # Detect faces: use xyxy to match your [x1, y1, x2, y2] expectation
    detections = detector.detect_faces(image_np, box_format="xyxy")

    face_tensors: List[torch.Tensor] = []
    bboxes: List[list] = []

    for det in detections:
        # det["box"] is [x1, y1, x2, y2] thanks to box_format="xyxy"
        x1, y1, x2, y2 = det["box"]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Safety clamp just in case
        h, w = image_np.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue  # skip degenerate boxes

        # Crop the face from the numpy image (HWC)
        face_crop = image_np[y1:y2, x1:x2, :]

        # Convert to CHW tensor (float)
        face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float()

        face_tensors.append(face_tensor)
        bboxes.append([float(x1), float(y1), float(x2), float(y2)])

    if is_debug_enabled and len(face_tensors) > 0:
        num_faces = len(face_tensors)
        fig, axes = plt.subplots(1, num_faces, figsize=(4 * num_faces, 4))

        # Handle single-face case
        if num_faces == 1:
            axes = [axes]

        for i, face in enumerate(face_tensors):
            # Convert back to HWC uint8 for plotting
            axes[i].imshow(face.permute(1, 2, 0).byte().numpy())
            axes[i].axis('off')
            axes[i].set_title(f'Face {i + 1}')

        fig.show()

    # Return list of (face_tensor, bounding_box)
    return list(zip(face_tensors, bboxes))




# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=-1) # -1 for CPU

# def face_detection_retina(image: Image.Image):
#     img_np = np.array(image.convert("RGB"))
#     faces = app.get(img_np)

#     results = []
#     for f in faces:
#         x1, y1, x2, y2 = f.bbox.astype(int).tolist()
#         face_crop = img_np[y1:y2, x1:x2]
#         face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float()
#         box = [float(x1), float(y1), float(x2), float(y2)]
#         results.append((face_tensor, box))
#     return results