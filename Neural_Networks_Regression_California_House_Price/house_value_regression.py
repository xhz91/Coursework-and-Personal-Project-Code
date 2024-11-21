import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Regressor(nn.Module):

    def __init__(self, x, nb_epoch = 1000, learning_rate=0.001, mini_batch_size=500, dropout_rate=0.5):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        super(Regressor, self).__init__()

        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.dropout = nn.Dropout(dropout_rate)

        # Initialise the LabelBinarizer instance used for the mapping from categorical values to 1-hot vectors
        self.lb = LabelBinarizer()

        # Initialise the normalising constants
        self.x_max = {}
        self.x_min = {}
        self.y_max = None
        self.y_min = None
        
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1

        self.hidden_layer1 = nn.Linear(self.input_size, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.hidden_layer3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, self.output_size)
        
        self.to(device)


    def forward(self, x):
        hidden1 = torch.relu(self.hidden_layer1(x))
        hidden1 = self.dropout(hidden1)
        hidden2 = torch.relu(self.hidden_layer2(hidden1))
        hidden2 = self.dropout(hidden2)
        hidden3 = torch.relu(self.hidden_layer3(hidden2))
        hidden3 = self.dropout(hidden3)
        output = self.output_layer(hidden3)
        return output
    
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        
        # handle text data(one-hot encoding)
        if training:
            ocean_proximity_encoded = self.lb.fit_transform(x['ocean_proximity'])
        else:
            ocean_proximity_encoded = self.lb.transform(x['ocean_proximity'])

        ocean_proximity_df = pd.DataFrame(ocean_proximity_encoded, columns=self.lb.classes_)

        x = x.drop('ocean_proximity', axis=1).reset_index(drop=True)
        ocean_proximity_df = ocean_proximity_df.reset_index(drop=True)
        x = pd.concat([x, ocean_proximity_df], axis=1)

        # fill NaN values with mean
        x = x.fillna(x.mean())

        # Process X
        # Store the normalising constants for X from training data to apply to test data
        if training:
            for column in x.columns:
                self.x_max[column] = x[column].max()
                self.x_min[column] = x[column].min()
        
        # Normalize the data
        for column in x.columns:
            x[column] = self.apply(x[column], self.x_max[column], self.x_min[column])
        
        # Process y
        if y is not None:
            # fill NaN values with mean
            y = y.fillna(y.mean())

            if training: # and y is not None:
                # Store the normalising constants for X from training data to apply to test data
                self.y_max = y.max()
                self.y_min = y.min()
            y = self.apply(y, self.y_max, self.y_min)
        
        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32) if y is not None else None

        return x_tensor, y_tensor

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        # Ask PyTorch to store any computed gradients so that we can examine them
        X.requires_grad_(True)

        # Select loss function
        criterion = torch.nn.MSELoss().to(device)

        # Select optimiser
        optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Loop through all epoch
        for epoch in range(self.nb_epoch):
            
            for i in range(0, X.size(0), self.mini_batch_size):
                X_batch = X[i:i+self.mini_batch_size]
                Y_batch = Y[i:i+self.mini_batch_size]
                
                # Reset gradients
                optimiser.zero_grad()
                
                # Forward pass
                predictions =self.forward(X_batch)
                
                # Calculate loss
                loss = criterion(predictions, Y_batch)
                #epoch_loss += loss.item()

                # Backpropagation and calculate gradient
                loss.backward()

                # Update weight and bias
                optimiser.step()
                
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def apply(self, data, max, min):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        return (data - min) / (max - min)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def revert(self, data, max, min):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Check if data is a tensor, and if so, move it to CPU and convert to NumPy
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()  # Move to CPU and convert to numpy if it's a tensor

        for i in range(len(data)):
            data[i] = data[i] * (max - min) + min

        return data


    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Preprocess input data
        y_pred = self.forward(X).detach().cpu().numpy()     # Forward pass and detach
        y_pred = self.revert(y_pred, self.y_max, self.y_min)  # Revert normalization

        return y_pred  # Return predictions

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {dict} -- A dictionary with different performance indicators.

        """

        # Preprocess the data
        X, Y = self._preprocessor(x, y=y, training=False)
        Y = self.revert(Y.cpu().numpy(), self.y_max, self.y_min)  # Revert true values
        
        # Get predictions from the model
        y_pred = self.predict(x)

        # Calculate performance indicators
        mse = mean_squared_error(Y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y, y_pred)
        r2 = r2_score(Y, y_pred)

        # Return or print the performance indicators
        results = {
            "Mean Squared Error (MSE)": mse,
            "Root Mean Squared Error (RMSE)": rmse,
            "Mean Absolute Error (MAE)": mae,
            "R² Score": r2
        }
        
        return results

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def perform_hyperparameter_search(x_train, y_train, x_val, y_val): 
    import optuna
    
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        - x_train {pd.DataFrame} -- Input features for training.
        - y_train {pd.DataFrame} -- Target values for training.
        - x_val {pd.DataFrame} -- Input features for validation.
        - y_val {pd.DataFrame} -- Target features for validation.
        
    Returns:
        The best-performing hyperparameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    def objective(trial):
        # Hyperparameter ranges
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        nb_epoch = trial.suggest_int("nb_epoch", 200, 1000)
        mini_batch_size = trial.suggest_categorical("mini_batch_size", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

        # Initialize model
        model = Regressor(x_train, nb_epoch=nb_epoch, learning_rate=learning_rate,
                          mini_batch_size=mini_batch_size, dropout_rate=dropout_rate)
        
        # Train model
        model.fit(x_train, y_train) 
            
        # Evaluate on validation set
        val_error = model.score(x_val, y_val)["Root Mean Squared Error (RMSE)"]
        # print(f"Validation RMSE = {val_error:.4f}")

        # Return validation score
        return val_error
  
    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction="minimize") # Minimize validation error
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    return best_params


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    # Split into training, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Perform hyperparameter search
    best_params = perform_hyperparameter_search(x_train, y_train, x_val, y_val)
    print(f"\nBest hyperparameters: {best_params}\n")

    # Train the final model using the best hyperparameters
    final_model = Regressor(
        x_train, 
        nb_epoch=best_params["nb_epoch"], 
        learning_rate=best_params["learning_rate"], 
        mini_batch_size=best_params["mini_batch_size"], 
        dropout_rate=best_params["dropout_rate"]
    )
    final_model.fit(x_train, y_train)
    save_regressor(final_model)

    # Evaluate the model on the training set
    train_error = final_model.score(x_train, y_train)
    print(f"Training error: {train_error}")

    # Evaluate on validation set
    val_error = final_model.score(x_val, y_val)
    print(f"Validation error: {val_error}")

    # Evaluate on test set
    test_error = final_model.score(x_test, y_test)
    print(f"Test error: {test_error}")


if __name__ == "__main__":
    example_main()




