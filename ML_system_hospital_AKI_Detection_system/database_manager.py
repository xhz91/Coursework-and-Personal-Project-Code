import sqlite3
import pandas as pd
import csv
import numpy as np
from patient import Patient
import os
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_filepath):
        self.db_filepath = db_filepath

    def _init_database(self, del_existing=True):
        """ Initialize an empty table in the db file with pre-defined columns.
        Works only if the db file does not already exist.

        Returns: None"""
        if del_existing:
            self.delete_database_file()

        try:
            with sqlite3.connect(self.db_filepath) as conn:
                cursor = conn.cursor()
                create_table_command = """CREATE TABLE main (
                mrn INTEGER PRIMARY KEY,
                sex TEXT,
                dob DATE,
                num_tests INTEGER,
                mean_creatinine FLOAT,
                latest_creatinine FLOAT
                );"""
                cursor.execute(create_table_command)
                conn.commit()
            print("Tables created successfully.")
        except sqlite3.OperationalError as e:
            print("Failed to create tables:", e)

    def _validate_table_exists(self):
        """ Check if the table called 'main' exists.

        Returns: bool
        """
        try:
            with sqlite3.connect(self.db_filepath) as conn:
                validation_command = """
                SELECT name FROM sqlite_master
                WHERE type='table'
                AND name='main';
                """
                cursor = conn.cursor()
                # Check if the table 'main' exists
                cursor.execute(validation_command)
                table_exists = cursor.fetchone()
                if not table_exists:
                    return False
                else:
                    return True
        except sqlite3.OperationalError as e:
            print("Failed to create or check table:", e)

    def delete_database_file(self):
        """ Delete the database by deleting its file.

        Returns: None
        """
        if os.path.exists(self.db_filepath):
            os.remove(self.db_filepath)

    def process_csv(self, file_path):
        """ Extract relevant patient information from historical .csv data

        Args:
            file_path (str): file path to historical csv data

        Returns:
            pd.Dataframe: Pandas dataframe containing mrn, mean_creatinine,
                          latest_creatinine and num_tests
                          for each row in the csv file.
        """
        # Empty list to store processed data
        result = []
        # Open and read the CSV file
        with open(file_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)

            fieldnames = reader.fieldnames  # Header columns from the file
            creatinine_result_keys = [key for key in fieldnames if
                                      key.startswith('creatinine_result_')]
            # Process each row
            for row in reader:
                mrn = row['mrn']
                # Initialize the list to store creatinine data
                creatinine_results = []
                for result_key in creatinine_result_keys:
                    if row[result_key]:
                        # Append only the result value if it exists
                        creatinine_results.append(float(row[result_key]))
                # Append the row data as a dictionary to the result list
                result.append({
                    'mrn': mrn,
                    'mean_creatinine': np.average(creatinine_results),
                    'latest_creatinine': creatinine_results[-1],
                    'num_tests': len(creatinine_results)
                })

        return pd.DataFrame(result)

    def _load_historic_data(self, csv_filepath):
        """ Load historic data in a provided raw CSV file into the database.

        Assumes data format is identical to that provided in courseworks
        (data, creatinine_result_n, creatinine_result_datetime_n) format.

        Returns: None
        """
        if not self._validate_table_exists():
            self._init_database()

        processed_df = self.process_csv(file_path=csv_filepath)

        # Insert into table
        insert_query = """
        INSERT INTO main (mrn, sex, dob, num_tests, mean_creatinine,
        latest_creatinine)
        VALUES (?, NULL, NULL, ?, ?, ?);
        """
        data_to_insert = processed_df[[
            "mrn",
            "num_tests",
            "mean_creatinine",
            "latest_creatinine"]].values.tolist()

        with sqlite3.connect(self.db_filepath) as conn:
            cursor = conn.cursor()
            cursor.executemany(insert_query, data_to_insert)
            conn.commit()

        return None

    def patient_exists(self, mrn):
        """ Check if a patient's MRN exists in the database

        Args:
            mrn (str): Patient's MRN identifier

        Returns:
            bool
        """
        query = "SELECT 1 FROM main WHERE mrn = ? LIMIT 1"
        with sqlite3.connect(self.db_filepath) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (mrn, ))
            patient_exists = cursor.fetchone() is not None

        return patient_exists

    def get_patient_info(self, mrn):
        """ Retrieve patient info if patient exists in database.

        Args:
            mrn (str): Patient's MRN identifier
        Returns:
            Patient() object with patient information.
            None if mrn does not exist.
        """
        if self.patient_exists(mrn):
            query = """SELECT mrn, dob, sex, mean_creatinine,
            latest_creatinine, num_tests FROM main WHERE mrn = ?"""

            with sqlite3.connect(self.db_filepath) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (mrn, ))
                row = cursor.fetchone()
                if row:
                    return Patient(
                        mrn=row[0],
                        date_of_birth=row[1],
                        sex=row[2],
                        mean_creatinine=row[3],
                        latest_creatinine=row[4],
                        num_tests=row[5]
                    )

        return None

    def to_datetime(self, date_str):
        """Converts date string into date.time object
        Args:
            date_str (str): String representing date in format "YYYY-MM-DD"

        Returns:
            datetime.date
        """
        try:
            # Try to parse the string with the expected format
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return None

    def create_record_from_pas(self, mrn, sex, dob):
        """Adds new patient record with sex and dob info to database

        Args:
            mrn (int): Patient's unique MRN identifier
            sex (str): Patient's sex ('M' for male or 'F' for female)
            dob (str): Patient's date of birth in format: "YYYY-MM-DD"
        """
        if not self.patient_exists(mrn):
            insert_query = """
            INSERT INTO MAIN (
            mrn,
            sex,
            dob,
            num_tests,
            mean_creatinine,
            latest_creatinine)
            VALUES (?, ?, ?, 0, NULL, NULL)"""

            dob_datetime = self.to_datetime(
                date_str=dob)

            with sqlite3.connect(self.db_filepath) as conn:
                cursor = conn.cursor()
                cursor.execute(insert_query, (mrn, sex, dob_datetime))
                conn.commit()

    def create_record_from_lims(self, mrn, new_creatinine_value):
        if not self.patient_exists(mrn):
            insert_query = """
            INSERT INTO MAIN (
            mrn,
            sex,
            dob,
            num_tests,
            mean_creatinine,
            latest_creatinine)
            VALUES (?, NULL, NULL, 1, ?, ?)
            """
            with sqlite3.connect(self.db_filepath) as conn:
                cursor = conn.cursor()
                cursor.execute(insert_query, (mrn,
                                              new_creatinine_value,
                                              new_creatinine_value))
                conn.commit()

    def update_dob_sex(self, mrn, dob, sex):
        """ Update the patient's row with features required from the model.

        Updates the patient's date of birth and sex.
        Args:
            mrn (str): Patient MRN
            dob (str): Patient date of birth
            sex (str): Patient sex
        """
        update_query = """
        UPDATE main
        SET dob = ?, sex = ?
        WHERE mrn = ?;
        """

        dob_datetime = self.to_datetime(date_str=dob)

        with sqlite3.connect(self.db_filepath) as conn:
            cursor = conn.cursor()
            cursor.execute(update_query, (dob_datetime, sex, mrn))
            conn.commit()

    def update_creatinine(self, mrn, new_creatinine_value):
        """ Update the patient's row given the latest blood test from LIMS.

        Increment the number of tests by 1, and recompute the new
        mean creatinine value. Also, add the latest creatinine value.

        Args:
            mrn (str): Patient MRN
            new_creatinine_value (str): Latest creatinine level
        Returns:
            None
        """
        patient = self.get_patient_info(mrn)

        new_num_tests, new_mean_creatinine = patient.calculate_mean_creatinine(
            new_creatinine_value=new_creatinine_value)

        update_query = """
        UPDATE main
        SET num_tests = ?, mean_creatinine = ?, latest_creatinine = ?
        WHERE mrn = ?"""
        with sqlite3.connect(self.db_filepath) as conn:
            cursor = conn.cursor()
            cursor.execute(update_query, (
                new_num_tests,
                new_mean_creatinine,
                new_creatinine_value,
                mrn))
            conn.commit()

    def check_has_sex_dob(self, mrn):
        """ Check if a given patient has sex and date of birth info.

        Returns:
            bool
        """
        if self.patient_exists(mrn):
            patient = self.get_patient_info(mrn)
            return patient.has_sex_and_dob()
        else:
            raise ValueError('Patient does not exist.')

    def check_has_complete_information(self, mrn):
        """Checks whether the patient record contains enough information
        to make an AKI prediction

        Args:
            mrn (int): Patient's unique MRN identifier
        Returns:
            bool
        """
        if self.patient_exists(mrn):
            patient = self.get_patient_info(mrn)
            return patient.has_complete_information()
        else:
            raise ValueError('Patient does not exist.')
