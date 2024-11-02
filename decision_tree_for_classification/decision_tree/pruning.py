import numpy as np
from decision_tree.decision_tree import Decision_Tree
from decision_tree.evaluation import confusion_matrix, precision_score, recall_score, f1_score_from_confusion, evaluate

# Load the datasets
clean_data = np.loadtxt("wifi_db/clean_dataset.txt")
noisy_data = np.loadtxt("wifi_db/noisy_dataset.txt")


# Separate features and labels 
X_clean = clean_data[:, :-1]  
y_clean = clean_data[:, -1]  

X_noisy = noisy_data[:, :-1]
y_noisy = noisy_data[:, -1]


# Cross-validation, Confusion matrix, and Metrics with pruning
def cross_validation_with_internal_cv(X, y, decision_tree, k=10, internal_k=9):
    # Shuffle the indices for cross-validation
    fold_size = len(y) // k
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    
    # Initiate evaluation metrics for internal cross-validation 
    class_labels = np.unique(y)
    all_confusion_matrices = np.zeros((len(class_labels), len(class_labels)))
    all_confusion_matrices_before_prune = np.zeros((len(class_labels), len(class_labels)))
    accuracies_before_prune = []
    precision_scores_per_class_before_prune = []
    recall_scores_per_class_before_prune = []
    f1_scores_per_class_before_prune = []
    accuracies = []
    precision_scores_per_class = []
    recall_scores_per_class = []
    f1_scores_per_class = []
    tree_depths_before_prune = []
    tree_nodes_nums_before_prune = []
    tree_depths_after_prune = []
    tree_nodes_nums_after_prune = []

    # Start cross-validation process
    for i in range(k):
        # Outer loop: test fold
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        remaining_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Inner loop: internal cross-validation on k-1 folds
        internal_fold_size = len(remaining_indices) // internal_k
        internal_indices = np.arange(len(remaining_indices))
        np.random.shuffle(internal_indices)
        
        # Start internal cross-validation process for each outer fold
        for j in range(internal_k):
            # Internal cross-validation fold
            validation_indices = internal_indices[j * internal_fold_size: (j + 1) * internal_fold_size]
            training_indices = np.concatenate([internal_indices[:j * internal_fold_size], internal_indices[(j + 1) * internal_fold_size:]])
            
            X_train_internal, X_val_internal = X[remaining_indices[training_indices]], X[remaining_indices[validation_indices]]
            y_train_internal, y_val_internal = y[remaining_indices[training_indices]], y[remaining_indices[validation_indices]]
            
            # Train the model on the training set
            decision_tree.root, _ = decision_tree.decision_tree_learning(X_train_internal, y_train_internal, depth=0)


            # Get confusion matrix and related information of the tree before pruning
            node_num, depth = decision_tree.get_tree_info()
            tree_depths_before_prune.append(depth)
            tree_nodes_nums_before_prune.append(node_num)
            accuracy_before_prune = evaluate(X_test, y_test, decision_tree)
            accuracies_before_prune.append(accuracy_before_prune)
            predictions_before_prune = decision_tree.predict(X_test)
            cm_before = confusion_matrix(y_test, predictions_before_prune, class_labels)
            all_confusion_matrices_before_prune += cm_before
            per_class_precision_before, _ = precision_score(cm_before)
            per_class_recall_before, _ = recall_score(cm_before)
            per_class_f1_before, _ = f1_score_from_confusion(cm_before)
            precision_scores_per_class_before_prune.append(per_class_precision_before)
            recall_scores_per_class_before_prune.append(per_class_recall_before)
            f1_scores_per_class_before_prune.append(per_class_f1_before)


            # Prune the decision tree using the validation set
            decision_tree.prune(X_val_internal, y_val_internal)

            # Get the number of nodes and depth of the tree after pruning
            node_num, depth = decision_tree.get_tree_info()
            tree_nodes_nums_after_prune.append(node_num)
            tree_depths_after_prune.append(depth)

            # Test the pruned tree on the test set
            accuracy = evaluate(X_test, y_test, decision_tree)
            accuracies.append(accuracy)

            # Generate predictions for confusion matrix and other metrics
            predictions = decision_tree.predict(X_test)

            # Compute the confusion matrix
            cm = confusion_matrix(y_test, predictions, class_labels)
            all_confusion_matrices += cm

            # Compute precision, recall, and F1-score for the test set
            per_class_precision, _ = precision_score(cm)
            per_class_recall, _ = recall_score(cm)
            per_class_f1, _ = f1_score_from_confusion(cm)

            precision_scores_per_class.append(per_class_precision)
            recall_scores_per_class.append(per_class_recall)
            f1_scores_per_class.append(per_class_f1)
    
    # Average the metrics over the 9*10 results for each class
    # Calculate average of accuracy, confusion matrix, precision, recall, f1 before prune
    avg_accuracies_before_prune = np.round(np.mean(accuracies_before_prune),decimals=3)
    avg_confusion_matrix_before_prune = np.round(all_confusion_matrices_before_prune/(k*internal_k),decimals=3)
    avg_precision_per_class_before_prune = np.round(np.mean(precision_scores_per_class_before_prune, axis=0),decimals=3)
    avg_recall_per_class_before_prune = np.round(np.mean(recall_scores_per_class_before_prune, axis=0),decimals=3)
    avg_f1_per_class_before_prune = np.round(np.mean(f1_scores_per_class_before_prune, axis=0),decimals=3)
    
    # Calculate average of accuracy, confusion matrix, precision, recall, f1 after prune
    avg_accuracy = np.round(np.mean(accuracies),decimals=3)
    avg_confusion_matrix = np.round(all_confusion_matrices/(k*internal_k),decimals=3)
    avg_precision_per_class = np.round(np.mean(precision_scores_per_class, axis=0),decimals=3)
    avg_recall_per_class = np.round(np.mean(recall_scores_per_class, axis=0),decimals=3)
    avg_f1_per_class = np.round(np.mean(f1_scores_per_class, axis=0),decimals=3)
    
    # Calculate averge number of node and average tree depth before and after prune
    avg_node_num_before = np.round(np.mean(tree_nodes_nums_before_prune),decimals=3)
    avg_tree_depth_before = np.round(np.mean(tree_depths_before_prune),decimals=3)
    avg_node_num_after = np.round(np.mean(tree_nodes_nums_after_prune),decimals=3)
    avg_tree_depth_after = np.round(np.mean(tree_depths_after_prune),decimals=3)
    

    
    return {
        'accuracy_before_prune': avg_accuracies_before_prune,
        'confusion_matrix_before_prune': avg_confusion_matrix_before_prune,
        'precision_per_class_before_prune': avg_precision_per_class_before_prune,
        'recall_per_class_before_prune': avg_recall_per_class_before_prune,
        'f1_per_class_before_prune': avg_f1_per_class_before_prune,
        'accuracy': avg_accuracy,
        'confusion_matrix': avg_confusion_matrix,
        'precision_per_class': avg_precision_per_class,
        'recall_per_class': avg_recall_per_class,
        'f1_per_class': avg_f1_per_class,
        'avg_node_num_before':  avg_node_num_before,
        'avg_tree_depth_before': avg_tree_depth_before,
        'avg_node_num_after': avg_node_num_after,
        'avg_tree_depth_after': avg_tree_depth_after
    }


if __name__ == "__main__":
    decision_tree = Decision_Tree()
    
    # Running 10-fold cross-validation with pruning on the clean dataset
    results_clean = cross_validation_with_internal_cv(X_clean, y_clean, decision_tree, k=10)
    print("Clean Dataset Results:")
    print("--------------before pruning-----------------")
    print("Accuracy:", results_clean['accuracy_before_prune'])
    print("Confusion Matrix:\n", results_clean['confusion_matrix_before_prune'])
    print("Precision per class:", results_clean['precision_per_class_before_prune'])
    print("Recall per class:", results_clean['recall_per_class_before_prune'])
    print("F1-measure per class:", results_clean['f1_per_class_before_prune'])

    print("--------------after pruning-----------------")
    print("Accuracy:", results_clean['accuracy'])
    print("Confusion Matrix:\n", results_clean['confusion_matrix'])
    print("Precision per class:", results_clean['precision_per_class'])
    print("Recall per class:", results_clean['recall_per_class'])
    print("F1-measure per class:", results_clean['f1_per_class'])
    print("Average number of nodes before pruning:", results_clean['avg_node_num_before'])
    print("Average tree depth before pruning:", results_clean['avg_tree_depth_before'])
    print("Average number of nodes after pruning:", results_clean['avg_node_num_after'])
    print("Average tree depth after pruning:", results_clean['avg_tree_depth_after'])
    decision_tree = Decision_Tree()
    
    # Running 10-fold cross-validation with pruning on the noisy dataset
    results_noisy = cross_validation_with_internal_cv(X_noisy, y_noisy, decision_tree, k=10)
    print("\nNoisy Dataset Results:")
    print("--------------before pruning-----------------")
    print("Accuracy:", results_noisy['accuracy_before_prune'])
    print("Confusion Matrix:\n", results_noisy['confusion_matrix_before_prune'])
    print("Precision per class:", results_noisy['precision_per_class_before_prune'])
    print("Recall per class:", results_noisy['recall_per_class_before_prune'])
    print("F1-measure per class:", results_noisy['f1_per_class_before_prune'])

    print("--------------after pruning-----------------")
    print("Accuracy:", results_noisy['accuracy'])
    print("Confusion Matrix:\n", results_noisy['confusion_matrix'])
    print("Precision per class:", results_noisy['precision_per_class'])
    print("Recall per class:", results_noisy['recall_per_class'])
    print("F1-measure per class:", results_noisy['f1_per_class'])
    print("Average number of nodes before pruning:", results_noisy['avg_node_num_before'])
    print("Average tree depth before pruning:", results_noisy['avg_tree_depth_before'])
    print("Average number of nodes after pruning:", results_noisy['avg_node_num_after'])
    print("Average tree depth after pruning:", results_noisy['avg_tree_depth_after'])
