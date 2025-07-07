import pickle
import requests
import datetime
import os


class Model:
    def __init__(self, weights_filepath):
        self.features = ['sex', 'age', 'mean_creatinine', 'latest_creatinine']
        self.target = 'aki'
        self.mllp_address = os.getenv("PAGER_ADDRESS", "localhost:8441").strip("/")
        if not self.mllp_address.startswith(("http://", "https://")):
            self.mllp_address = f"http://{self.mllp_address}"

        with open(weights_filepath, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, test_data):
        """Make predictions on new data and save the results"""
        return self.model.predict(test_data)

    def send_aki_alert(self, mrn, timestamp=None):
        # Set timestamp to current datetime if None is provided
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Prepare data in the desired format
        data = {"mrn": mrn, "date": timestamp}  # Use the format YYYYMMDDHHMMSS for timestamp

        # Headers to ensure correct content type
        headers = {"Content-Type": "text/plain"}

        # Sending POST request with formatted data
        try:
            response = requests.post(f"{self.mllp_address}/page", data=f"{data['mrn']},{data['date']}", headers=headers)


            # Check response status and print appropriate message
            if response.status_code == 200:
                print(f"AKI alert sent successfully for MRN {mrn} at {timestamp}")
            else:
                print(f"Failed to send AKI alert: {response.status_code} {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
