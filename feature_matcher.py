"""
Feature Matching Module
Provides feature matching functionality with multiple algorithm flavors.
"""
import numpy as np


# Modify this to expect the args, the database (i.e vector of tuples (descriptor and string)), the calculated feature_vector (output from feature_extraction.py)
def feature_matcher(feature_matching_args, database, test_feature_vectors):
    """
    Selects and runs the appropriate feature matching algorithm.

    Args:
        feature_matching_args: Namespace with feature matching flags (euclidean, cosine, chi) and threshold
        database: List of tuples (descriptor, label) from generate_descriptors
        test_feature_vectors: List of tuples (descriptor, bbox) from feature extraction on test image

    Returns:
        List of tuples (descriptor, bbox, label) for matched faces only
    """
    threshold = getattr(feature_matching_args, 'threshold', None)

    if feature_matching_args.euclidean:
        return feature_matching_euclidean(database, test_feature_vectors, threshold)
    elif feature_matching_args.cosine:
        return feature_matching_cosine(database, test_feature_vectors, threshold)
    elif feature_matching_args.chi:
        return feature_matching_chi_square(database, test_feature_vectors, threshold)


# Modify functions below to implement 1-to-n matching logic with database.
# In the matching logic you should compare the database against the descriptors that match then take the bounding boxes from the matched descriptors (and the labels from the database) and apply them to the input image.
def feature_matching_chi_square(database, test_feature_vectors, threshold):
    """
    Chi-square distance feature matching (hand-implemented).

    Args:
        database: List of tuples (descriptor, label) from training
        test_feature_vectors: List of tuples (descriptor, bbox) from test image
        threshold: Maximum chi-square distance for a match (lower is better)

    Returns:
        List of tuples (descriptor, bbox, label) for matched faces only
    """
    print("Running the chi_square flavor (hand-implemented)")

    matched_faces = []

    for test_descriptor, test_bbox in test_feature_vectors:
        best_match_distance = float('inf')
        best_match_label = None
        best_match_descriptor = None

        # Find best match in database
        for db_descriptor, db_label in database:
            distance = chi_square_distance(test_descriptor, db_descriptor)

            if distance < best_match_distance:
                best_match_distance = distance
                best_match_label = db_label
                best_match_descriptor = db_descriptor

        # Only add if below threshold
        if threshold is None or best_match_distance < threshold:
            print(f"  Match found: {best_match_label} with distance {best_match_distance:.4f}")
            matched_faces.append((best_match_descriptor, test_bbox, best_match_label))
        else:
            print(f"  No match found (best distance {best_match_distance:.4f} > threshold {threshold})")

    return matched_faces


def chi_square_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Computes chi-square distance between two histograms.
    Implemented by hand for learning purposes.

    Args:
        hist1: First histogram (normalized)
        hist2: Second histogram (normalized)

    Returns:
        Chi-square distance value
    """
    distance = 0.0

    for i in range(len(hist1)):
        numerator = (hist1[i] - hist2[i]) ** 2
        denominator = hist1[i] + hist2[i]

        # Avoid division by zero
        if denominator > 0:
            distance += numerator / denominator

    return distance * 0.5  # Multiply by 0.5 for standard chi-square formula


def feature_matching_euclidean(database, test_feature_vectors, threshold):
    """
    Euclidean distance feature matching using numpy.

    Args:
        database: List of tuples (descriptor, label) from training
        test_feature_vectors: List of tuples (descriptor, bbox) from test image
        threshold: Maximum Euclidean distance for a match (lower is better)

    Returns:
        List of tuples (descriptor, bbox, label) for matched faces only
    """
    print("Running the euclidean flavor (numpy implementation)")

    matched_faces = []

    for test_descriptor, test_bbox in test_feature_vectors:
        best_match_distance = float('inf')
        best_match_label = None
        best_match_descriptor = None

        # Find best match in database
        for db_descriptor, db_label in database:
            distance = np.linalg.norm(test_descriptor - db_descriptor)

            if distance < best_match_distance:
                best_match_distance = distance
                best_match_label = db_label
                best_match_descriptor = db_descriptor

        # Only add if below threshold
        if threshold is None or best_match_distance < threshold:
            print(f"  Match found: {best_match_label} with distance {best_match_distance:.4f}")
            matched_faces.append((best_match_descriptor, test_bbox, best_match_label))
        else:
            print(f"  No match found (best distance {best_match_distance:.4f} > threshold {threshold})")

    return matched_faces


def feature_matching_cosine(database, test_feature_vectors, threshold):
    """
    Cosine similarity feature matching using numpy.

    Args:
        database: List of tuples (descriptor, label) from training
        test_feature_vectors: List of tuples (descriptor, bbox) from test image
        threshold: Maximum cosine distance for a match (lower is better, distance = 1 - similarity)

    Returns:
        List of tuples (descriptor, bbox, label) for matched faces only
    """
    print("Running the cosine flavor (numpy implementation)")

    matched_faces = []

    for test_descriptor, test_bbox in test_feature_vectors:
        best_match_distance = float('inf')
        best_match_label = None
        best_match_descriptor = None

        # Find best match in database
        for db_descriptor, db_label in database:
            # Cosine similarity = dot product / (norm1 * norm2)
            dot_product = np.dot(test_descriptor, db_descriptor)
            norm1 = np.linalg.norm(test_descriptor)
            norm2 = np.linalg.norm(db_descriptor)
            similarity = dot_product / (norm1 * norm2)
            # Convert to distance: distance = 1 - similarity
            distance = 1 - similarity

            if distance < best_match_distance:
                best_match_distance = distance
                best_match_label = db_label
                best_match_descriptor = db_descriptor

        # Only add if below threshold
        if threshold is None or best_match_distance < threshold:
            print(f"  Match found: {best_match_label} with distance {best_match_distance:.4f}")
            matched_faces.append((best_match_descriptor, test_bbox, best_match_label))
        else:
            print(f"  No match found (best distance {best_match_distance:.4f} > threshold {threshold})")

    return matched_faces