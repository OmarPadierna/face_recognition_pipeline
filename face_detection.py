"""
Face Detection Module
Provides face detection functionality with multiple algorithm flavors.
"""

from typing import List
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from matplotlib import pyplot as plt

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


def face_detection_haar(image: Image.Image) -> List[tuple]:
    """Haar Cascade face detection stub.

    Args:
        image: PIL Image object

    Returns:
        List of tuples (face_tensor, bounding_box)
    """
    print("Running the haar flavor")
    return []


# Modify here to include bounding boxes. Output should be a list of tuples. Tensor + bounding box. This way they can be used for labeling.
def face_detection_mtcnn(image: Image.Image, is_debug_enabled: bool) -> List[tuple]:
    """MTCNN face detection implementation.

    Args:
        image: PIL Image object
        is_debug_enabled: A flag that enables/disabled visualization and logs for debugging purposes
    Returns:
        List of tuples (face_tensor, bounding_box) where bounding_box is [x1, y1, x2, y2]
    """
    mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cuda')

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