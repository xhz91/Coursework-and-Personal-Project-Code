import socket
import datetime
from message_interpreter import MessageInterpreter
from database_manager import DatabaseManager
from preprocessor import PreProcessor
from model import Model
import os
from time import time

# MLLP protocol constants
MLLP_START_OF_BLOCK = 0x0b
MLLP_END_OF_BLOCK = 0x1c
MLLP_CARRIAGE_RETURN = 0x0d
MLLP_BUFFER_SIZE = 1024


def generate_ack_message(ack_code="AA"):
    """ Generate a properly formatted HL7 acknowledgement message.

    Returns:
        mllp_message: A MLLP message in bytes
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Create the segments separately
    msh_segment = f"MSH|^~\\&|||||{timestamp}||ACK|||2.5"
    msa_segment = f"MSA|{ack_code}"

    # Join segments with carriage return
    message = f"{msh_segment}\r{msa_segment}\r"

    # MLLP wrapping
    mllp_message = bytes([MLLP_START_OF_BLOCK]) + \
                   message.encode('ascii') + \
                   bytes([MLLP_END_OF_BLOCK, MLLP_CARRIAGE_RETURN])

    return mllp_message


def parse_mllp_messages(buffer):
    """ Parse MLLP messages from the received byte stream.

    Returns:
        messages: A list of parsed HL7 messages (each message as a list of fields)
        buffer: Remaining unprocessed data
    """
    messages = []
    consumed = 0
    expect = MLLP_START_OF_BLOCK
    i = 0

    while i < len(buffer):
        if expect is not None:
            if buffer[i] != expect:
                raise Exception(f"Bad MLLP encoding: expected {hex(expect)}, found {hex(buffer[i])}")
            if expect == MLLP_START_OF_BLOCK:
                expect = None
                consumed = i
            elif expect == MLLP_CARRIAGE_RETURN:
                hl7_message = buffer[consumed + 1:i - 1].decode("ascii")
                segments = hl7_message.split("\r")

                # Split each segment by '|' and remove empty fields
                parsed_message = [field for segment in segments for field in segment.split("|") if field]
                messages.append(parsed_message)
                expect = MLLP_START_OF_BLOCK
                consumed = i + 1
        else:
            if buffer[i] == MLLP_END_OF_BLOCK:
                expect = MLLP_CARRIAGE_RETURN
        i += 1

    return messages, buffer[consumed:]


def receive_hl7_messages(host, port):
    """
    Receive HL7 messages over MLLP, process them, and send ACK responses.
    """
    model = Model('decision_tree_model.pkl')
    preprocessor = PreProcessor()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connected to MLLP server at {host}:{port}")

        buffer = b""
        while True:
            data = s.recv(MLLP_BUFFER_SIZE)
            if not data:
                print("Connection closed by server")
                break

            buffer += data
            messages, buffer = parse_mllp_messages(buffer)
            for message in messages:

                check_tuple = write_message_to_database(message_lst=message)

                if check_tuple is not None:
                    mrn, check_patient_info_required = check_tuple

                    if check_patient_info_required:
                        patient_df = preprocessor.process_data(
                            mrn=mrn,
                            db_filepath='database.db')
                        prediction = model.predict(test_data=patient_df)

                        if prediction == 1:
                            print(f'Time before sending POSITIVE PREDICTION: {time()}')
                            if message[5].startswith('ORU'):
                                timestamp_update = message[12]
                                model.send_aki_alert(mrn=mrn, timestamp=timestamp_update)
                                print(f'Time after sending POSITIVE PREDICTION: {time()}')
                                print('Sending positive prediction information')

                print("Received HL7 message")
                print(message)
                print("|".join(message))

                print(f'Time before sending ACKNOWLEDGEMENT message: {time()}')
                ack_message = generate_ack_message("AA")
                s.sendall(ack_message)
                print(f'Time after sending ACKKNOWLEDGEMENT message: {time()}')

                # print("Final ACK message (decoded):")
                # message_parts = ack_message[1:-2].decode("ascii").split("\r")
                # for part in message_parts:
                #     print(part)


def write_message_to_database(message_lst):
    """ Handle writing a message to the database given a list containing
    message information from the initial parser. If the message does not
    require updating the database, no writing occurs.

    Then, return a tuple containing the patient MRN that has been evaluated
    along with whether a check for full features for AKI prediction is needed.

    Args:
        message_lst: A list of information from the message

    Returns:
        Tuple (int, bool): (mrn, and check requirement for AKI prediction.)
    """
    interpreter = MessageInterpreter()
    db_manager = DatabaseManager(db_filepath='database.db')
    message_type = interpreter.identify_message_type(message_lst)

    if message_type == "PAS - Admission":
        pas_message = interpreter.interpret_PAS_admission_message(message_lst)
        if not db_manager.patient_exists(mrn=pas_message.mrn):
            
            print(f'Time before creating ')
            db_manager.create_record_from_pas(
                mrn=pas_message.mrn,
                sex=pas_message.sex,
                dob=pas_message.dob)

            return (pas_message.mrn, False)
        else:
            if not db_manager.check_has_sex_dob(mrn=pas_message.mrn):
                db_manager.update_dob_sex(
                    mrn=pas_message.mrn,
                    dob=pas_message.dob,
                    sex=pas_message.sex)

                return (pas_message.mrn, True)

    elif message_type == 'LIMS':
        lims_message = interpreter.interpret_LIMS_message(message_lst)
        if lims_message.result_type == 'CREATININE':
            if not db_manager.patient_exists(mrn=lims_message.mrn):
                db_manager.create_record_from_lims(
                    mrn=lims_message.mrn,
                    new_creatinine_value=lims_message.result_value
                )

                return (lims_message.mrn, False)
            else:
                db_manager.update_creatinine(
                    mrn=lims_message.mrn,
                    new_creatinine_value=lims_message.result_value)

                return (lims_message.mrn, True)

    elif message_type == 'UNKNOWN' or message_type == 'PAS - Discharge':
        return None
    else:
        raise Exception('Unable to identify message type.')


def main():
    # Load in historic data
    db_manager = DatabaseManager('database.db')
    db_manager._init_database(del_existing=True)
    db_manager._load_historic_data('/data/history.csv')
    # Get env MLLP SERVER HOST and MLLP SERVER PORT
    mllp_address = os.getenv('MLLP_ADDRESS', 'localhost:8440')
    host, port = mllp_address.split(':')
    # Begin receiving HL7 messages
    receive_hl7_messages(host, int(port))


if __name__ == "__main__":
    main()
