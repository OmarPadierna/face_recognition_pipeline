#!/usr/bin/env python3
"""
Face Detection Pipeline
Orchestrates face detection, feature extraction, and feature matching
with multiple algorithm flavors.
"""

from argparse import ArgumentParser
from PIL import Image
from face_detection import face_detector
from feature_extractor import feature_extractor
from feature_matcher import feature_matcher


def run_pipeline(args, image):
    """
    Orchestrates the face detection pipeline.

    Args:
        args: CLI arguments including all pipeline flags
        image: Input image

    Returns:
        Labeled image with identified faces
    """
    aligned_faces = face_detector(args, image)
    feature_vector = feature_extractor(args, aligned_faces)
    labeled_image = feature_matcher(args, feature_vector)

    return labeled_image


def main():
    parser = ArgumentParser(
        description='Face Detection Pipeline with multiple algorithm flavors'
    )

    # Input image argument
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image'
    )

    # Debug flag
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with visualization'
    )

    # Face Detection arguments
    face_detection_group = parser.add_mutually_exclusive_group(required=True)
    face_detection_group.add_argument(
        '--mtcnn',
        action='store_true',
        help='Use MTCNN for face detection'
    )
    face_detection_group.add_argument(
        '--haar',
        action='store_true',
        help='Use Haar Cascade for face detection'
    )
    face_detection_group.add_argument(
        '--scrfd',
        action='store_true',
        help='Use SCRFD for face detection'
    )

    # Feature Extraction arguments
    feature_extraction_group = parser.add_mutually_exclusive_group(required=True)
    feature_extraction_group.add_argument(
        '--vggface',
        action='store_true',
        help='Use VGGFace for feature extraction'
    )
    feature_extraction_group.add_argument(
        '--facenet',
        action='store_true',
        help='Use FaceNet for feature extraction'
    )
    feature_extraction_group.add_argument(
        '--deepface',
        action='store_true',
        help='Use DeepFace for feature extraction'
    )

    # Feature Matching arguments
    feature_matching_group = parser.add_mutually_exclusive_group(required=True)
    feature_matching_group.add_argument(
        '--euclidean',
        action='store_true',
        help='Use Euclidean distance for feature matching'
    )
    feature_matching_group.add_argument(
        '--cosine',
        action='store_true',
        help='Use Cosine similarity for feature matching'
    )
    feature_matching_group.add_argument(
        '--svm',
        action='store_true',
        help='Use SVM for feature matching'
    )
    feature_matching_group.add_argument(
        '--knn',
        action='store_true',
        help='Use KNN for feature matching'
    )

    args = parser.parse_args()

    # Load the input image
    image = Image.open(args.input)
    run_pipeline(args, image)

if __name__ == '__main__':
    main()