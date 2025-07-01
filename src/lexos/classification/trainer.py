# trainer.py


from sklearn.model_selection import train_test_split # type:ignore
from sklearn.svm import SVC # type:ignore
from sklearn.linear_model import LogisticRegression # type:ignore
from sklearn.metrics import classification_report # type:ignore
from sklearn.tree import DecisionTreeClassifier  # type:ignore
from sklearn.ensemble import RandomForestClassifier  # type:ignore
from sklearn.naive_bayes import MultinomialNB  # type:ignore
from sklearn.neighbors import KNeighborsClassifier  # type:ignore



# Function to train and evaluate a classifier
def train_classifier(feature_matrix, target_labels, model='svc', test_size=0.4, random_state=None):
    """
    Split the input document-term matrix (feature_matrix) and labels (target_labels) into 
    training and testing sets, train a classifier, and return the trained model and a performance report.
    
    Parameters:
        feature_matrix : document-term matrix (input features)
        target_labels : list of labels (target values)
        model : classifier to train. Supported: 'svc', 'logistic', 'decision_tree',
                'random_forest', 'knn'.
        test_size : fraction of data to reserve for testing
        random_state : seed for reproducibility
        
    Returns:
        clf : trained classifier model
        report : classification report as a string
    """

    # Split features and labels into training and test sets
    features_train, features_test, labels_train, labels_test = train_test_split(
        feature_matrix, target_labels, test_size=test_size, random_state=random_state
    )

    # Initialize the chosen classifier
    if model == 'svc':
        clf = SVC(kernel="linear")
    elif model == 'logistic':
        clf = LogisticRegression(max_iter=1000)
    elif model == 'decision_tree':
        clf = DecisionTreeClassifier()
    elif model == 'random_forest':
        clf = RandomForestClassifier()
    elif model == 'knn':
        clf = KNeighborsClassifier()
    else:
        raise ValueError(f"Unknown model: {model}")

    # Fit the classifier to the training data
    clf.fit(features_train, labels_train)

    # Predict labels on the test set
    predicted_labels = clf.predict(features_test)

    # Generate a classification report
    report = classification_report(labels_test, predicted_labels)
    return clf, report

# Function to predict labels on new data
def predict_labels(clf, new_feature_matrix):
    """
    Use a trained classifier to predict labels for new/unseen data.

    Parameters:
        clf : trained classifier model
        new_feature_matrix : new document-term matrix (input features)
    
    Returns:
        predicted_labels : list of predicted labels
    """
    return clf.predict(new_feature_matrix)
