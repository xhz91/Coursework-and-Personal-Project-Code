import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

# Load the datasets
clean_data = np.loadtxt("wifi_db/clean_dataset.txt")
noisy_data = np.loadtxt("wifi_db/noisy_dataset.txt")


# Separate features and labels 
X_clean = clean_data[:, :-1]  
y_clean = clean_data[:, -1]  

X_noisy = noisy_data[:, :-1]
y_noisy = noisy_data[:, -1]


class tree_node():
    def __init__(self, attribute, data, data_num=0, is_leaf=False):
        self.attribute = attribute
        self.data = data
        self.left = None
        self.right = None
        self.is_leaf = is_leaf
        self.data_num = data_num


class Decision_Tree():
    def __init__(self):   
        self.depth = 0
        self.root = None
        self.node_cnt = 0
    

    def traverse(self, node, depth):
        if node.is_leaf:
            self.node_cnt += 1
            # print(node.data)
            if depth > self.depth:
                self.depth = depth
            return

        self.node_cnt += 1
        self.traverse(node.left, depth + 1)
        self.traverse(node.right, depth + 1)


    def get_tree_info(self):
        self.node_cnt = 0
        self.depth = 0
        self.traverse(self.root, 0)
        return self.node_cnt, self.depth


    def prune(self, x_val, y_val):
        # Prune the tree
        self._prune(self.root, x_val, y_val)


    def _prune(self, node, x_val, y_val):
        left_prune_success = True
        right_prune_success = True
        while (left_prune_success or right_prune_success):
            left_prune_success = False
            right_prune_success = False
            if (node.left.is_leaf and node.right.is_leaf):
                # Calculate the error without pruning
                predictions = self.predict(x_val)
                error_without_pruning = np.sum(predictions != y_val) / len(y_val)
                
                # keep the information of original nodes
                cur_attribute = node.attribute
                cur_data = node.data
                left_node = node.left
                right_node = node.right

                # Prune the node
                node.is_leaf = True
                # keep the majority data
                if (left_node.data_num > right_node.data_num):
                    node.data = left_node.data
                else:
                    node.data = right_node.data
                node.left = None
                node.right = None
                node.data_num = max(left_node.data_num, right_node.data_num)
                
                # Calculate the error with pruning
                predictions = self.predict(x_val)
                error_with_pruning = np.sum(predictions != y_val) / len(y_val)
                
                # Compare the errors
                if error_with_pruning <= error_without_pruning:
                    return True
                else:
                    # Revert the pruning
                    node.is_leaf = False
                    node.attribute = cur_attribute
                    node.data = cur_data
                    node.left = left_node
                    node.right = right_node
                    return False
                        
            elif not node.left.is_leaf and not node.right.is_leaf: 
                left_prune_success = self._prune(node.left, x_val, y_val)
                right_prune_success = self._prune(node.right, x_val, y_val)
            elif not node.right.is_leaf:
                right_prune_success = self._prune(node.right, x_val, y_val)
            elif not node.left.is_leaf:
                left_prune_success = self._prune(node.left, x_val, y_val)


    def predict(self, x_test):
        # predict the test data
        return np.array([self._predict_single(sample, self.root) for sample in x_test])


    def _predict_single(self, sample, node):
        if node.data is not None and node.is_leaf:
            return node.data
        if sample[node.attribute] <= node.data:  
            return self._predict_single(sample, node.left)
        else:
            return self._predict_single(sample, node.right)


    def decision_tree_learning(self, x_train, y_train, depth=0):
        if np.unique(y_train).shape[0] == 1:
            return tree_node(None, y_train[0], len(y_train), True), depth    
        
        best_attribute, best_threshold, best_gain = self.find_split(x_train, y_train)
        root = tree_node(best_attribute, best_threshold)
        l_branch, l_depth = self.decision_tree_learning(
            x_train[x_train[:, best_attribute] <= best_threshold], 
            y_train[x_train[:, best_attribute] <= best_threshold], 
            depth + 1
        )
        r_branch, r_depth = self.decision_tree_learning(
            x_train[x_train[:, best_attribute] > best_threshold], 
            y_train[x_train[:, best_attribute] > best_threshold], 
            depth + 1
        )
        root.left = l_branch
        root.right = r_branch
        self.root = root
        self.depth = max(l_depth, r_depth)
        return root, max(l_depth, r_depth)


    def find_split(self, x_train, y_train):
        # Total entropy
        total_entropy = self.entropy(y_train)
        best_attribute = 0
        best_threshold = 0
        best_gain = 0

        for i in range(x_train.shape[1]):
            # Get the unique values of the current attribute
            unique_values = np.unique(x_train[:, i])
            
            # Go through all the unique values
            for j in range(len(unique_values) - 1):
                threshold = (unique_values[j] + unique_values[j + 1]) / 2
                left_mask = x_train[:, i] <= threshold
                right_mask = x_train[:, i] > threshold

                left_y = y_train[left_mask]
                right_y = y_train[right_mask]

                # Calculate the entropy of the left and right sides
                left_entropy = self.entropy(left_y) if left_y.size > 0 else 0
                right_entropy = self.entropy(right_y) if right_y.size > 0 else 0

                # Calculate the information gain
                gain = total_entropy - (left_entropy * left_y.size / y_train.size + right_entropy * right_y.size / y_train.size)

                # Update the best gain
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_attribute = i

        return best_attribute, best_threshold, best_gain

    
    def entropy(self, y_train):
        # Count the number of each unique label
        unique, counts = np.unique(y_train, return_counts=True)
        label_count = dict(zip(unique, counts))
        
        # Calculate the probability of each label
        probabilities = np.array([label_count[label] / len(y_train) for label in unique])
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Avoiding log(0)
        return entropy
    

    def visualisation(self):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(20, 10))  # Wide figure size to accommodate horizontal spread
        
        # Define the plotting helper function
        def plot_node(node, x, y, dx, depth):
            if node is None:
                return
            
            # Define rectangle size
            box_width, box_height = 12, 1
            
            # Plot the node: differentiate between leaf and decision node
            if node.is_leaf:
                ax.add_patch(patches.Rectangle((x - box_width/2, y - box_height/2), box_width, box_height,
                                            edgecolor='blue', facecolor='lightgreen'))
                ax.text(x, y, f'leaf:{node.data:.3f}', ha='center', va='center', fontsize=10)
            else:
                ax.add_patch(patches.Rectangle((x - box_width/2, y - box_height/2), box_width, box_height,
                                            edgecolor='blue', facecolor='lightblue'))
                ax.text(x, y, f'[X{node.attribute} < {node.data:.1f}]', ha='center', va='center', fontsize=10)
            
            # If not a leaf, plot the children recursively
            if node.left:
                # Calculate position for left child
                new_x = x - dx
                new_y = y - 2
                ax.plot([x, new_x], [y - box_height / 2, new_y + box_height / 2], color='black', lw=2)
                plot_node(node.left, new_x, new_y, dx * 0.5, depth + 1) 
            
            if node.right:
                # Calculate position for right child
                new_x = x + dx
                new_y = y - 2
                ax.plot([x, new_x], [y - box_height / 2, new_y + box_height / 2], color='black', lw=2)
                plot_node(node.right, new_x, new_y, dx * 0.5, depth + 1) 

        # Start plotting from the root, centered at (0, 0) and with increased initial spacing dx
        initial_dx = 50  # Adjust horizontal spacing for better spread
        plot_node(self.root, 0, 0, initial_dx, 0)  # Use black for the root
        
        # Set axis limits and hide the axes
        ax.set_xlim(-100, 100)  # Adjust x-axis range for better horizontal spacing
        ax.set_ylim(-self.depth * 2 - 2, 2)  # Adjust y-axis range based on depth
        ax.axis('off')  # Hide axes for better visualization
        
        # Show the plot
        plt.show()


if __name__ == "__main__":
    decision_tree = Decision_Tree()
    decision_tree.root, _ = decision_tree.decision_tree_learning(X_clean, y_clean)
    
    decision_tree.visualisation()
