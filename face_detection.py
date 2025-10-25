"""
Face Detection Module
Provides face detection functionality with multiple algorithm flavors.
"""

def face_detector(face_detection_args, image):
    """
    Selects and runs the appropriate face detection algorithm.

    Args:
        face_detection_args: Namespace with face detection flags (mtcnn, haar, scrfd)
        image: Input image

    Returns:
        Aligned faces from the selected detection algorithm
    """
    if face_detection_args.mtcnn:
        return face_detection_mtcnn(image)
    elif face_detection_args.haar:
        return face_detection_haar(image)
    elif face_detection_args.scrfd:
        return face_detection_scrfd(image)

def face_detection_haar(image):
    """Haar Cascade face detection stub."""
    print("Running the haar flavor")
    return image


def face_detection_mtcnn(image):
    """MTCNN face detection stub."""
    print("Running the mtcnn flavor")
    return image


def face_detection_scrfd(image):
    """SCRFD face detection stub."""
    print("Running the scrfd flavor")
    return image