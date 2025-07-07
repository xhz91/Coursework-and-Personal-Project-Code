# run pytest -s tests/test_database_manager.py

import pytest
from database_manager import DatabaseManager
from patient import Patient


@pytest.fixture
def db_manager():
    return DatabaseManager(db_filepath="tests/test.db")


@pytest.fixture
def mrn_from_historical_data():
    return 178994518


@pytest.fixture
def new_mrn():
    return 1234567890


@pytest.fixture
def new_mrn_sex():
    return 'F'


@pytest.fixture
def new_mrn_dob():
    return '2001-10-15'


def test_init_database(db_manager):
    db_manager._init_database()

    assert db_manager._validate_table_exists()


def test_get_patient_info(db_manager, mrn_from_historical_data):
    db_manager._init_database()
    db_manager._load_historic_data(csv_filepath='history.csv')
    assert db_manager.get_patient_info(mrn_from_historical_data) is not None


def test_load_historic_data(db_manager, mrn_from_historical_data):
    db_manager._init_database()
    db_manager._load_historic_data(csv_filepath='history.csv')
    expected = Patient(
        mrn=mrn_from_historical_data,
        date_of_birth=None,
        sex=None,
        mean_creatinine=51.37,
        latest_creatinine=51.37,
        num_tests=1
    )
    actual = db_manager.get_patient_info(mrn=mrn_from_historical_data)

    assert actual == expected


def test_create_record_from_pas(db_manager, mrn_from_historical_data,
                                new_mrn_sex, new_mrn_dob):
    db_manager._init_database()
    assert not db_manager.patient_exists(mrn=mrn_from_historical_data)
    db_manager.create_record_from_pas(
        mrn=mrn_from_historical_data, sex=new_mrn_sex, dob=new_mrn_dob)
    assert db_manager.patient_exists(mrn=mrn_from_historical_data)
    expected = Patient(
        mrn=mrn_from_historical_data,
        sex=new_mrn_sex,
        date_of_birth=new_mrn_dob,
        mean_creatinine=None,
        latest_creatinine=None,
        num_tests=0
    )
    actual = db_manager.get_patient_info(mrn=mrn_from_historical_data)

    assert expected == actual


def test_update_dob_sex(db_manager, mrn_from_historical_data,
                        new_mrn_sex, new_mrn_dob):
    db_manager._init_database()

    db_manager._load_historic_data(csv_filepath='history.csv')

    assert db_manager.patient_exists(mrn=mrn_from_historical_data)

    db_manager.update_dob_sex(
        mrn=mrn_from_historical_data,
        dob=new_mrn_dob,
        sex=new_mrn_sex
    )

    expected = Patient(
        mrn=mrn_from_historical_data,
        sex=new_mrn_sex,
        date_of_birth=new_mrn_dob,
        mean_creatinine=51.37,
        latest_creatinine=51.37,
        num_tests=1
    )

    actual = db_manager.get_patient_info(mrn=mrn_from_historical_data)

    assert expected == actual


def test_whether_first_creatinine_result_gets_added_correctly(
        db_manager, 
        new_mrn, 
        new_mrn_sex, 
        new_mrn_dob):
    db_manager._init_database()
    # Create record from PAS
    db_manager.create_record_from_pas(
        mrn=new_mrn,
        sex=new_mrn_sex,
        dob=new_mrn_dob)
    # Simulate updating record from LIMS
    db_manager.update_creatinine(
        mrn=new_mrn,
        new_creatinine_value=101.1
    )
    # Retrieve the actual patient info
    actual = db_manager.get_patient_info(mrn=new_mrn)

    expected = Patient(
        mrn=new_mrn,
        date_of_birth=new_mrn_dob,
        sex=new_mrn_sex,
        num_tests=1,
        latest_creatinine=101.1,
        mean_creatinine=101.1
    )

    assert actual == expected


def test_whether_second_creatinine_result_gets_added_correctly(
        db_manager, 
        mrn_from_historical_data):
    db_manager._init_database()
    db_manager._load_historic_data(csv_filepath='history.csv')

    actual = db_manager.get_patient_info(mrn=mrn_from_historical_data)
    # Check the loading in was done properly
    assert actual.latest_creatinine == 51.37
    assert actual.num_tests == 1

    existing_mean_creatinine_result = actual.mean_creatinine
    existing_num_tests = actual.num_tests

    new_creatinine_value = 53.37

    db_manager.update_creatinine(
        mrn=mrn_from_historical_data,
        new_creatinine_value=new_creatinine_value)

    actual = db_manager.get_patient_info(mrn=mrn_from_historical_data)

    expected = Patient(
        mrn=mrn_from_historical_data,
        sex=None,
        date_of_birth=None,
        latest_creatinine=new_creatinine_value,
        mean_creatinine=(existing_mean_creatinine_result * existing_num_tests
                         + new_creatinine_value) / (existing_num_tests + 1),
        num_tests=existing_num_tests + 1
    )

    assert actual == expected


def test_whether_third_creatinine_result_gets_added_correctly(
        db_manager,
        mrn_from_historical_data):
    db_manager._init_database()
    db_manager._load_historic_data(csv_filepath='history.csv')

    actual = db_manager.get_patient_info(mrn=mrn_from_historical_data)
    # Check the loading in was done properly

    existing_latest_creatinine_result = actual.latest_creatinine

    new_creatinine_values = [53.37, 52.37]

    for new_creatinine_value in new_creatinine_values:
        db_manager.update_creatinine(
            mrn=mrn_from_historical_data,
            new_creatinine_value=new_creatinine_value)

    actual = db_manager.get_patient_info(mrn=mrn_from_historical_data)

    expected = Patient(
        mrn=mrn_from_historical_data,
        sex=None,
        date_of_birth=None,
        latest_creatinine=new_creatinine_values[-1],
        mean_creatinine=(sum([existing_latest_creatinine_result,
                              *new_creatinine_values]) / 3),
        num_tests=3
    )

    assert actual == expected


def test_able_to_correctly_label_patient_record_as_complete(
        db_manager,
        mrn_from_historical_data,
        new_mrn_dob,
        new_mrn_sex):
    db_manager._init_database()
    db_manager._load_historic_data(csv_filepath='history.csv')
    db_manager.update_dob_sex(
        mrn=mrn_from_historical_data,
        dob=new_mrn_dob,
        sex=new_mrn_sex
    )
    actual = db_manager.get_patient_info(mrn=mrn_from_historical_data)
    expected = Patient(
        mrn=mrn_from_historical_data,
        date_of_birth=new_mrn_dob,
        sex=new_mrn_sex,
        num_tests=1,
        latest_creatinine=51.37,
        mean_creatinine=51.37
    )
    assert actual == expected
    assert db_manager.check_has_complete_information(
        mrn=mrn_from_historical_data) is True


def test_able_to_correctly_patient_record_as_incomplete(
        db_manager,
        mrn_from_historical_data):
    db_manager._init_database()
    db_manager._load_historic_data(csv_filepath='history.csv')
    actual = db_manager.get_patient_info(mrn=mrn_from_historical_data)
    expected = Patient(
        mrn=mrn_from_historical_data,
        date_of_birth=None,
        sex=None,
        num_tests=1,
        latest_creatinine=51.37,
        mean_creatinine=51.37
    )
    assert actual == expected
    assert db_manager.check_has_complete_information(
        mrn=mrn_from_historical_data) is False