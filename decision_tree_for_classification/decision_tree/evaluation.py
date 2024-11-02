import numpy as np
import decision_tree.decision_tree
from decision_tree.decision_tree import Decision_Tree

# Load the datasets
clean_data = np.loadtxt("wifi_db/clean_dataset.txt")
noisy_data = np.loadtxt("wifi_db/noisy_dataset.txt")


# Separate features and labels 
X_clean = clean_data[:, :-1]  
y_clean = clean_data[:, -1]  

X_noisy = noisy_data[:, :-1]
y_noisy = noisy_data[:, -1]


# Cross-validation, Confusion matrix, and Metrics
def cross_validation(X, y, decision_tree, k=10):
    fold_size = len(y) // k
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    
    class_labels = np.unique(y)
    all_confusion_matrices = np.zeros((len(class_labels), len(class_labels)))  # set confusion matrix size
    
    accuracies = []
    precision_scores_per_class = []
    recall_scores_per_class = []
    f1_scores_per_class = []
    
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Train the decision tree on the training set
        decision_tree.root, _ = decision_tree.decision_tree_learning(X_train, y_train, depth=0)
        
        # Evaluate the decision tree (get accuracy)
        accuracy = evaluate(X_test, y_test, decision_tree)
        accuracies.append(accuracy)
        
        # Generate predictions for confusion matrix and other metrics
        predictions = decision_tree.predict(X_test)
        
        # Compute the confusion matrix
        cm = confusion_matrix(y_test, predictions, class_labels)
        all_confusion_matrices += cm
        
        # Compute precision, recall, and F1-score for each class
        per_class_precision, _ = precision_score(cm)
        per_class_recall, _ = recall_score(cm)
        per_class_f1, _ = f1_score_from_confusion(cm)

        precision_scores_per_class.append(per_class_precision)
        recall_scores_per_class.append(per_class_recall)
        f1_scores_per_class.append(per_class_f1)
    
    # Average the metrics over the k folds for each class
    avg_accuracy = np.round(np.mean(accuracies),decimals=3)
    avg_confusion_matrix = np.round(all_confusion_matrices/k,decimals=3)
    avg_precision_per_class = np.round(np.mean(precision_scores_per_class, axis=0),decimals=3)
    avg_recall_per_class = np.round(np.mean(recall_scores_per_class, axis=0),decimals=3)
    avg_f1_per_class = np.round(np.mean(f1_scores_per_class, axis=0),decimals=3)
    
    return {
        'accuracy': avg_accuracy,
        'confusion_matrix': avg_confusion_matrix,
        'precision_per_class': avg_precision_per_class,
        'recall_per_class': avg_recall_per_class,
        'f1_per_class': avg_f1_per_class
    }


def evaluate(X_test, y_test, tree):

    # Get predictions using the tree's predict function
    predictions = tree.predict(X_test)
    
    # Calculate the accuracy: proportion of correct predictions
    accuracy = np.mean(predictions == y_test)
    
    return accuracy


def confusion_matrix(y_gold, y_prediction, class_labels):
    """ Compute the confusion matrix. """
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    for (i, label) in enumerate(class_labels):
        indices = (y_gold == label)
        predictions = y_prediction[indices]

        unique_labels, counts = np.unique(predictions, return_counts=True)
        frequency_dict = dict(zip(unique_labels, counts))

        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


def precision_score(confusion):
    """ Compute precision score and macro precision """
    p = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])

    macro_p = np.mean(p)
    return p, macro_p


def recall_score(confusion):
    """ Compute recall score and macro recall """
    r = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])

    macro_r = np.mean(r)
    return r, macro_r


def f1_score_from_confusion(confusion):
    """ Compute F1 score and macro F1 """
    precisions, _ = precision_score(confusion)
    recalls, _ = recall_score(confusion)

    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    macro_f = np.mean(f)
    return f, macro_f


if __name__ == "__main__":
    decision_tree = Decision_Tree()
    
    # Running cross-validation on the clean dataset
    results_clean = cross_validation(X_clean, y_clean, decision_tree, k=10)
    print("Clean Dataset Results:")
    print("Accuracy:", results_clean['accuracy'])
    print("Confusion Matrix:\n", results_clean['confusion_matrix'])
    print("Precision per class:", results_clean['precision_per_class'])
    print("Recall per class:", results_clean['recall_per_class'])
    print("F1-measure per class:", results_clean['f1_per_class'])

    decision_tree = Decision_Tree()
    
    # Running cross-validation on the noisy dataset
    results_noisy = cross_validation(X_noisy, y_noisy, decision_tree, k=10)
    print("\nNoisy Dataset Results:")
    print("Accuracy:", results_noisy['accuracy'])
    print("Confusion Matrix:\n", results_noisy['confusion_matrix'])
    print("Precision per class:", results_noisy['precision_per_class'])
    print("Recall per class:", results_noisy['recall_per_class'])
    print("F1-measure per class:", results_noisy['f1_per_class'])
