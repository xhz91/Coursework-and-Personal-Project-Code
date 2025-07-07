# To build docker image: docker build -t test_1 .
# To run docker image: docker run -v ${PWD}/data:/data --env MLLP_ADDRESS=host.docker.internal:8440 --env PAGER_ADDRESS=host.docker.internal:8441 test_1

# Use Ubuntu Oracular as the base image
FROM ubuntu:oracular

# Install Python, pip, and venv
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -yq install python3-pip python3-venv

# Set the working directory
WORKDIR /simulator

# Copy Python source files
COPY simulator.py /simulator/
COPY simulator_test.py /simulator/
COPY client.py /simulator/
COPY database_manager.py /simulator/
COPY model.py /simulator/
COPY patient.py /simulator/
COPY preprocessor.py /simulator/
COPY message_interpreter.py /simulator/
COPY decision_tree_model.pkl /simulator/

# Copy and install dependencies
COPY requirements.txt /simulator/
RUN python3 -m venv /simulator
RUN /simulator/bin/pip install -r requirements.txt

# Expose the necessary ports
EXPOSE 8440
EXPOSE 8441

# Set entrypoint script as the default command
ENTRYPOINT ["/simulator/bin/python", "/simulator/client.py"]

# Default arguments for simulator.py
CMD ["--messages=/data/messages.mllp"]

