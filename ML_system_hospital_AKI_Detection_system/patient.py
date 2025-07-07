class Patient:
    """ Contains information about a patient."""
    def __init__(self, mrn, date_of_birth=None, sex=None, mean_creatinine=None,
                 latest_creatinine=None, num_tests=None):
        """
        Args:
            mrn (int): Patient's unique MRN identifier
            date_of_birth (str, optional): Patient's date of birth in
            format YYYY-MM-DD. Defaults to None.
            sex (str, optional): Patient's sex ('M' for male or 'F' for
            female). Defaults to None.
            mean_creatinine (float, optional): Average creatinine
            value over all of the patient's blood tests. Defaults to None.
            latest_creatinine (float, optional): Creatinine value from
            latest blood test. Defaults to None.
            num_tests (int, optional): Total number of creatine blood tests.
            Defaults to None.
        """
        self.mrn = mrn
        self.date_of_birth = date_of_birth
        self.sex = sex
        self.mean_creatinine = mean_creatinine
        self.latest_creatinine = latest_creatinine
        self.num_tests = num_tests

    def __eq__(self, other):
        if not isinstance(other, Patient):
            return False
        return (
            self.mrn == other.mrn and
            self.date_of_birth == other.date_of_birth and
            self.sex == other.sex and
            self.mean_creatinine == other.mean_creatinine and
            self.latest_creatinine == other.latest_creatinine and
            self.num_tests == other.num_tests
        )

    def __str__(self):
        """Return a readable string representation of the patient."""
        return (f"""Patient(MRN: {self.mrn}, DOB: {self.date_of_birth},
                Sex: {self.sex}, Mean Creatinine: {self.mean_creatinine},
                Latest Creatinine: {self.latest_creatinine},
                Number of Tests: {self.num_tests})""")

    def calculate_mean_creatinine(self, new_creatinine_value):
        """ Calculate the new mean given a new test, and update num_tests.

        Args:
            mrn (int): Patient's unique MRN identifier
            new_creatinine_value (float): Creatinine val from a new blood test
        Returns:
            new_num_tests: The new number of tests
            new_mean_creatinine: The new number of tests
        """
        new_num_tests = self.num_tests + 1

        if self.mean_creatinine is None:
            new_mean_creatinine = new_creatinine_value
        else:
            new_mean_creatinine = ((self.mean_creatinine * self.num_tests) +
                                   new_creatinine_value) / new_num_tests

        return new_num_tests, new_mean_creatinine

    def has_sex_and_dob(self):
        """ Check if a Patient has sex and dob information.

        Returns:
            bool
        """
        return all([
            self.date_of_birth,
            self.sex
        ])

    def has_complete_information(self):
        """ Check if Patient info is all present for AKI prediction.

        Returns:
            bool
        """
        return all([
            self.has_sex_and_dob(),
            self.num_tests > 0,
            self.mean_creatinine,
            self.latest_creatinine])
