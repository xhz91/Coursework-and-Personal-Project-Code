from datetime import datetime

class MessageInterpreter:
    def __init__(self):
        pass

    def identify_message_type(self, message_lst):
        """ Identify whether the HL7 message is PAS or LIMS and if PAS, determine admission or discharge.

        Args:
            message_lst (list): Parsed HL7 message as a list of fields.

        Returns:
            str: "PAS - Admission" if it's an admission message.
                "PAS - Discharge" if it's a discharge message.
                "PAS - Other" if it's a different ADT message.
                "LIMS" if it's a LIMS message.
                "UNKNOWN" if type cannot be determined.
        """
        if len(message_lst) < 10:
            return "UNKNOWN"

        # Extract message type from MSH.9
        message_type = message_lst[5]

        if message_type.startswith("ADT"):
            if message_type == "ADT^A01":
                return "PAS - Admission"
            elif message_type == "ADT^A03":
                return "PAS - Discharge"
            else:
                return "PAS - Other"
        elif message_type.startswith("ORU"):
            return "LIMS"
        else:
            return "UNKNOWN"

    def interpret_PAS_admission_message(self, message_lst):
        """_summary_

        Args:
            message (_type_): _description_

        Returns:
            _type_: _description_
        """
        index_of_mrn_in_message = 9
        index_of_dob_in_messsage = 11
        index_of_sex_in_message = 12

        return PASMessage(
            mrn=message_lst[index_of_mrn_in_message],
            dob=message_lst[index_of_dob_in_messsage],
            sex=message_lst[index_of_sex_in_message])
    
    def interpret_LIMS_message(self, message_lst):
        """_summary_

        Args:
            message_lst (_type_): _description_
        """

        index_of_mrn_in_message = 9
        index_of_test_date_and_time = 12
        index_of_result_type = 16
        index_of_result_value = 17

        return LIMSMessage(
            mrn=message_lst[index_of_mrn_in_message],
            test_date_and_time=message_lst[index_of_test_date_and_time],
            result_type=message_lst[index_of_result_type],
            result_value=message_lst[index_of_result_value])


class PASMessage:
    def __init__(self, mrn, dob, sex):
        self.mrn = mrn
        self.dob = self.recast_date(dob)
        self.sex = sex

    def recast_date(self, date_str):
        # Parse the string into a datetime object
        date_obj = datetime.strptime(date_str, "%Y%m%d")

        # Format the datetime object into the desired format
        return date_obj.strftime("%Y-%m-%d")


class LIMSMessage:
    def __init__(self, mrn, result_type, test_date_and_time, result_value):
        """_summary_

        Args:
            mrn (_type_): _description_
            result_type (_type_): _description_
            test_date_and_time (_type_): _description_
            result_value (_type_): _description_
        """
        self.mrn = mrn
        self.result_type = result_type
        self.test_date_and_time = test_date_and_time
        self.result_value = float(result_value)
