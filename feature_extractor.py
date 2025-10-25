"""
Feature Extraction Module
Provides feature extraction functionality with multiple algorithm flavors.
"""

def feature_extractor(feature_extraction_args, aligned_faces):
    """
    Selects and runs the appropriate feature extraction algorithm.

    Args:
        feature_extraction_args: Namespace with feature extraction flags (vggface, facenet, deepface)
        aligned_faces: Aligned faces from face detection

    Returns:
        Feature vectors from the selected extraction algorithm
    """
    if feature_extraction_args.vggface:
        return feature_extraction_vggface(aligned_faces)
    elif feature_extraction_args.facenet:
        return feature_extraction_facenet(aligned_faces)
    elif feature_extraction_args.deepface:
        return feature_extraction_deepface(aligned_faces)


def feature_extraction_vggface(aligned_faces):
    """VGGFace feature extraction stub."""
    print("Running the vggface flavor")
    return aligned_faces


def feature_extraction_facenet(aligned_faces):
    """FaceNet feature extraction stub."""
    print("Running the facenet flavor")
    return aligned_faces


def feature_extraction_deepface(aligned_faces):
    """DeepFace feature extraction stub."""
    print("Running the deepface flavor")
    return aligned_faces
