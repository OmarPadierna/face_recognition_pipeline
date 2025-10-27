"""
Feature Matching Module
Provides feature matching functionality with multiple algorithm flavors.
"""
import numpy as np

def feature_matcher(feature_matching_args, feature_vector):
    """
    Selects and runs the appropriate feature matching algorithm.

    Args:
        feature_matching_args: Namespace with feature matching flags (Euclidean, cosine, svm, knn)
        feature_vector: Feature vectors from feature extraction

    Returns:
        Labeled image with identified faces
    """
    if feature_matching_args.euclidean:
        return feature_matching_euclidean(feature_vector)
    elif feature_matching_args.cosine:
        return feature_matching_cosine(feature_vector)
    elif feature_matching_args.chi:
        return feature_matching_chi_square(feature_vector)



def feature_matching_chi_square(feature_vectors):
    """
    Chi-square distance feature matching (hand-implemented).

    Args:
        feature_vectors: List of feature descriptors from extraction

    Returns:
        Match results
    """
    print("Running the chi_square flavor (hand-implemented)")

    # TODO: Implement 1-to-N matching logic with database
    # For now, compute pairwise distances between all detected faces
    if len(feature_vectors) > 1:
        print(f"Computing chi-square distances between {len(feature_vectors)} faces")
        for i in range(len(feature_vectors)):
            for j in range(i + 1, len(feature_vectors)):
                dist = chi_square_distance(feature_vectors[i], feature_vectors[j])
                print(f"  Distance between face {i} and face {j}: {dist:.4f}")

    return feature_vectors

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

def feature_matching_euclidean(feature_vectors):
    """
    Euclidean distance feature matching using numpy.

    Args:
        feature_vectors: List of feature descriptors from extraction

    Returns:
        Match results
    """
    print("Running the euclidean flavor (numpy implementation)")

    # TODO: Implement 1-to-N matching logic with database
    # For now, compute pairwise distances between all detected faces
    if len(feature_vectors) > 1:
        print(f"Computing Euclidean distances between {len(feature_vectors)} faces")
        for i in range(len(feature_vectors)):
            for j in range(i + 1, len(feature_vectors)):
                dist = np.linalg.norm(feature_vectors[i] - feature_vectors[j])
                print(f"  Distance between face {i} and face {j}: {dist:.4f}")

    return feature_vectors


def feature_matching_cosine(feature_vectors):
    """
    Cosine similarity feature matching using numpy.

    Args:
        feature_vectors: List of feature descriptors from extraction

    Returns:
        Match results
    """
    print("Running the cosine flavor (numpy implementation)")

    # TODO: Implement 1-to-N matching logic with database
    # For now, compute pairwise similarities between all detected faces
    if len(feature_vectors) > 1:
        print(f"Computing cosine similarities between {len(feature_vectors)} faces")
        for i in range(len(feature_vectors)):
            for j in range(i + 1, len(feature_vectors)):
                # Cosine similarity = dot product / (norm1 * norm2)
                dot_product = np.dot(feature_vectors[i], feature_vectors[j])
                norm1 = np.linalg.norm(feature_vectors[i])
                norm2 = np.linalg.norm(feature_vectors[j])
                similarity = dot_product / (norm1 * norm2)
                # Convert to distance: distance = 1 - similarity
                distance = 1 - similarity
                print(f"  Distance between face {i} and face {j}: {distance:.4f}")

    return feature_vectors