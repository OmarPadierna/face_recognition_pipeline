#!/usr/bin/env python3
"""
Face Detection Pipeline
Orchestrates face detection, feature extraction, and feature matching
with multiple algorithm flavors.
"""

from argparse import ArgumentParser
from argparse import Namespace
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


class FaceRecognitionPipeline:
    """
    Face recognition pipeline for generating descriptors and detecting/labeling faces.
    """

    def __init__(self, args):
        """
        Initialize the pipeline with configuration arguments.

        Args:
            args: Namespace with pipeline configuration flags
        """
        self.args = args

    def generate_descriptors(self, image):
        """
        Generate face descriptors from an image.

        Args:
            image: Input image (PIL Image)

        Returns:
            List of tuples containing (descriptor, label) for each face
        """
        pass

    def detect_faces(self, image, descriptors):
        """
        Detect faces in an image and label them using the provided descriptors.

        Args:
            image: Input image (PIL Image)
            descriptors: List of tuples containing (descriptor, label) from generate_descriptors

        Returns:
            None (prints/displays the labeled image)
        """
        pass

def main():
    parser = ArgumentParser(
        description='Face Detection Pipeline with multiple algorithm flavors'
    )

    # Input image argument
    parser.add_argument(
        '--input_train',
        type=str,
        required=True,
        help='Path to input image'
    )

    parser.add_argument(
        '--input_test',
        type=str,
        required=True,
        help='Path to input image'
    )

    pipeline_args_mtcnn_lbp_chi = Namespace(
        mtcnn=True,  # Face detection method
        haar=False,

        lbp=True,  # Feature extraction method
        facenet=False,

        euclidean=False,  # Feature matching method
        cosine=False,
        chi=True,

        debug=True  # Optional: enable debug visualization
    )

    pipeline_args_mtcnn_lbp_cosine = Namespace(
        mtcnn=True,  # Face detection method
        haar=False,

        lbp=True,  # Feature extraction method
        facenet=False,

        euclidean=False,  # Feature matching method
        cosine=True,
        chi=False,

        debug=True  # Optional: enable debug visualization
    )

    pipeline_args_mtcnn_lbp_euclidean = Namespace(
        mtcnn=True,  # Face detection method
        haar=False,

        lbp=True,  # Feature extraction method
        facenet=False,

        euclidean=True,  # Feature matching method
        cosine=False,
        chi=False,

        debug=True  # Optional: enable debug visualization
    )

    args = parser.parse_args()

    image_train = Image.open(args.input_train)
    image_test = Image.open(args.input_test)


    mtcnn_lbp_cosine = FaceRecognitionPipeline(pipeline_args_mtcnn_lbp_cosine)
    mtcnn_lbp_chi = FaceRecognitionPipeline(pipeline_args_mtcnn_lbp_chi)
    mtcnn_lbp_euclidean = FaceRecognitionPipeline(pipeline_args_mtcnn_lbp_euclidean)

    #Database generation only uses mtcnn->lbp (no feature_matching) so its the same for all these three pipelines
    database = mtcnn_lbp_cosine.generate_descriptors(image_train)

    mtcnn_lbp_chi.detect_faces(image_test, database)
    mtcnn_lbp_euclidean.detect_faces(image_test, database)
    mtcnn_lbp_cosine.detect_faces(image_test, database)

if __name__ == '__main__':
    main()