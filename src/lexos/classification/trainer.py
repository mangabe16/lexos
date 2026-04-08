"""trainer.py.

Last Updated: June 30, 2025
Last Tested: TBD.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer



# Function to train and evaluate a classifier
def train_classifier(
    feature_matrix, target_labels, model: str = "svc", test_size: float = 0.4, random_state=None, normalize=None, bootstrap = False
):
    """Train the classifier.

    Split the input document-term matrix (feature_matrix) and labels (target_labels) into training and testing sets, train a classifier, and return the trained model and a performance report.

    Args:
        feature_matrix: document-term matrix (input features)
        target_labels: list of labels (target values)
        model: classifier to train. Supported: 'svc', 'logistic', 'decision_tree',
            'random_forest', 'knn', 'naive_bayes', 'mlp'.
        test_size: fraction of data to reserve for testing
        random_state: seed for reproducibility
        normalize: choice to normalize features
        bootstrap: whether to apply bootstrapping (resampling with replacement) to the training data

    Returns:
        clf: trained classifier model
        report: classification report as a string
    """
    # Split features and labels into training and test sets
    features_train, features_test, labels_train, labels_test = train_test_split(
        feature_matrix,
        target_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=target_labels if len(set(target_labels)) > 1 else None,
    )

    if bootstrap:
        features_train, labels_train = resample( # resample the training data with replacement
            features_train, labels_train, random_state=random_state
        )

    if normalize: # if user requests normalization
        scaler = normalize_features(normalize)
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

    # Initialize the chosen classifier
    if model == "svc":
        clf = SVC(kernel="linear", random_state=random_state)
    elif model == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model == "decision_tree":
        clf = DecisionTreeClassifier(random_state=random_state)
    elif model == "random_forest":
        clf = RandomForestClassifier(random_state=random_state)
    elif model == "knn":
        clf = KNeighborsClassifier()
    elif model == "naive_bayes":
        clf = MultinomialNB()
    elif model == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=600,
            early_stopping=False,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Fit the classifier to the training data
    clf.fit(features_train, labels_train)

    # Predict labels on the test set
    predicted_labels = clf.predict(features_test)

    # Generate a classification report
    report = classification_report(labels_test, predicted_labels)
    return clf, report


# Function to predict labels on new data
def predict_labels(clf, new_feature_matrix):
    """Use a trained classifier to predict labels for new/unseen data.

    Args:
        clf: trained classifier model
        new_feature_matrix: new document-term matrix (input features)

    Returns:
        predicted_labels: list of predicted labels
    """
    return clf.predict(new_feature_matrix)


# Function to fit a classifier on a pre-split feature matrix
def fit_classifier(feature_matrix, target_labels, model: str = "svc", normalize=None, class_weight=None, **kwargs):
    """Fit a classifier on a pre-split feature matrix (no internal train/test split).

    Returns the fitted sklearn estimator.
    """
    if normalize: # if user requests normalization
        scaler = normalize_features(normalize)
        feature_matrix = scaler.fit_transform(feature_matrix)

    sample_weight = kwargs.pop("sample_weight", None)


    registry = {
        "svc": SVC,
        "decision_tree": DecisionTreeClassifier,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "naive_bayes": MultinomialNB,
        "mlp": MLPClassifier,
    }
    key = model.lower()
    if key not in registry:
        raise ValueError(f"Unknown model '{model}'. Choose from {sorted(registry.keys())}.")
    Estimator = registry[key]

    if key == "mlp":
        effective_class_weight = kwargs.pop("class_weight", class_weight)
        if effective_class_weight is not None:
            class_sample_weight = compute_sample_weight(
                class_weight=effective_class_weight,
                y=target_labels,
            )
            if sample_weight is None:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.asarray(sample_weight) * class_sample_weight

    clf = Estimator(**kwargs)
    fit_kwargs = {}
    if key == "mlp" and sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    clf.fit(feature_matrix, target_labels, **fit_kwargs)
    return clf

def normalize_features(normalize):
    """Return the appropriate scaler based on the normalization method.

    Args:
        normalize: The normalization method ('standard', 'minmax', 'robust', 'l2').

    Returns:
        scaler: An instance of the selected scaler.
    """
    if normalize == "standard":
        scaler = StandardScaler()
    elif normalize == "minmax":
        scaler = MinMaxScaler()
    elif normalize == "robust":
        scaler = RobustScaler()
    elif normalize == "l2":
        scaler = Normalizer(norm="l2")
    else:
        raise ValueError(f"Unsupported normalization method: {normalize}")
    return scaler
