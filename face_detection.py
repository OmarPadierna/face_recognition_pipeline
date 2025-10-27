"""
Face Detection Module
Provides face detection functionality with multiple algorithm flavors.
"""

from typing import List
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from matplotlib import pyplot as plt

def face_detector(face_detection_args, image) -> List[torch.Tensor]:
    """
    Selects and runs the appropriate face detection algorithm.

    Args:
        face_detection_args: Namespace with face detection flags (mtcnn, haar)
        image: Input image

    Returns:
        Aligned faces from the selected detection algorithm
    """
    if face_detection_args.mtcnn:
        is_debug_enabled = getattr(face_detection_args, 'debug', False)
        return face_detection_mtcnn(image, is_debug_enabled)
    elif face_detection_args.haar:
        return face_detection_haar(image)

def face_detection_haar(image: Image.Image) -> List[torch.Tensor]:
    """Haar Cascade face detection stub.

    Args:
        image: PIL Image object

    Returns:
        List of detected face tensors
    """
    print("Running the haar flavor")
    return []

def face_detection_mtcnn(image: Image.Image, is_debug_enabled:bool) -> List[torch.Tensor]:
    """MTCNN face detection implementation.

    Args:
        image: PIL Image object
        is_debug_enabled: A flag that enables/disabled visualization and logs for debugging purposes
    Returns:
        List of detected face tensors
    """
    mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cuda')

    # Display original image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')

    # Detect faces
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
                axes[i].set_title(f'Face {i+1}')

            fig.show()
    #Note: MTCNN does face alignment automatically.
    return faces if faces is not None else []