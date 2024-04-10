"""utils.features Module"""

import numpy as np

def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """Prepare data for training.

    :param data: input data.
    :param polynomial_degree: degree of additional polynomial features.
    :param sinusoid_degree: degree of sinusoid features.
    :param normalize_data: flag indicating whether to normalize data.
    :return: processed data, features mean, features deviation.
    """
    processed_data = data.copy()

    # Generate polynomial features.
    polynomial_features = generate_polynomial_features(data, polynomial_degree)
    processed_data = np.hstack((processed_data, polynomial_features))

    # Generate sinusoid features.
    sinusoid_features = generate_sinusoid_features(data, sinusoid_degree)
    processed_data = np.hstack((processed_data, sinusoid_features))

    # Normalize data if required.
    if normalize_data:
        processed_data, features_mean, features_deviation = normalize_features(processed_data)
        return processed_data, features_mean, features_deviation

    return processed_data


def generate_polynomial_features(data, degree):
    """Generate polynomial features.

    :param data: input data.
    :param degree: degree of polynomial features.
    :return: polynomial features.
    """
    polynomial_features = []
    for d in range(1, degree + 1):
        polynomial_features.append(data ** d)
    return np.hstack(polynomial_features)


def generate_sinusoid_features(data, degree):
    """Generate sinusoid features.

    :param data: input data.
    :param degree: degree of sinusoid features.
    :return: sinusoid features.
    """
    sinusoid_features = []
    for d in range(1, degree + 1):
        sinusoid_features.append(np.sin(d * data))
    return np.hstack(sinusoid_features)


def normalize_features(data):
    """Normalize features.

    :param data: input data.
    :return: normalized data, features mean, features deviation.
    """
    features_mean = np.mean(data, axis=0)
    features_deviation = np.std(data, axis=0, ddof=1)
    normalized_data = (data - features_mean) / features_deviation
    return normalized_data, features_mean, features_deviation
