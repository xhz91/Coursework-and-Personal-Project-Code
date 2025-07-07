import pytest
import datetime
from unittest.mock import patch, MagicMock
import pandas as pd
from model import Model  # Replace 'your_module' with the actual module name


# Sample test data
test_data = pd.DataFrame({
    'sex': [1],
    'age': [65],
    'mean_creatinine': [1.2],
    'latest_creatinine': [1]
})

model_weights = "decision_tree_model.pkl"
def test_prediction(mocker):
    # Arrange
    predictor = Model(model_weights)

    # Act
    result = predictor.predict(test_data)
    print(result)

    assert result in [0, 1]
    


# Sample test data

mrn = '123765409'
  

def test_pager(mocker):
    # Arrange
    predictor = Model(model_weights)
    # Mock the model to return a prediction of 1 (AKI)
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]  # Simulate AKI prediction

    # Mock pickle.load to return the mock model
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("pickle.load", return_value=mock_model)

    # Mock requests.post to avoid making actual HTTP requests
    mock_post = mocker.patch("requests.post")

    # Act
    predictor.send_aki_alert(mrn)

    # Assert
    # Check if requests.post was called with the correct arguments
    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == "http://localhost:8441/page"
    # assert mock_post.call_args[1]["headers"] == {"Content-Type": "application/json"}
    # assert mock_post.call_args[1]["data"] == f"{test_data['mrn'].iloc[0]},{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    