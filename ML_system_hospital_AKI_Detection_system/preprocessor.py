from database_manager import DatabaseManager
from datetime import datetime
import pandas as pd


class PreProcessor:
    """ Parse retrieved data from the database and convert to model inputs."""
    def __init__(self):
        self.features = ['sex', 'age', 'mean_creatinine', 'latest_creatinine']

    def _get_age_in_years(self, dob: str) -> int:
        """
        Calculate the age of a person given their date of birth in 'YYYY-MM-DD'
        format.

        Args:
            dob (str): Date of birth as a string in 'YYYY-MM-DD' format.
        Returns:
            int: The person's age.
        """
        birth_date = datetime.strptime(dob, "%Y-%m-%d").date()
        today = datetime.today().date()
        # Calculate the initial age difference in years
        age = today.year - birth_date.year
        # Check if the birthday has not yet occurred this year
        has_not_had_birthday = 1 if (today.month, today.day) < (birth_date.month, birth_date.day) else 0
        # Adjust age based on whether the birthday has passed or not
        age -= has_not_had_birthday

        return age

    def process_data(self, mrn, db_filepath):
        """ Retrieves complete patient information from database,
        transforming it into a pd.DataFrame required by model.

        Args:
            mrn (str):
            db_filepath (str):
        Returns:
            patient_df (pd.DataFrame)
        """
        db_manager = DatabaseManager(db_filepath)
        patient = db_manager.get_patient_info(mrn)
        try:
            patient_data_dict = {
                "sex": 0 if patient.sex == 'F' else 1,
                "age": self._get_age_in_years(patient.date_of_birth),
                "mean_creatinine": patient.mean_creatinine,
                "latest_creatinine": patient.latest_creatinine
            }
            patient_df = pd.DataFrame(patient_data_dict, index=[0])

            return patient_df

        except AttributeError as e:
            print(f"Error accessing patient attributes: {e}")
            raise ValueError("Missing expected patient attributes.") from e


if __name__ == '__main__':
    pass
