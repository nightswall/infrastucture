#!/bin/bash

# Set the directory of the script as an environment variable
SCRIPT_DIR=$(dirname "$0")

# Check if the sshpass command is installed
if ! command -v sshpass > /dev/null; then
  # Install sshpass if it is not installed
  echo "sshpass is not installed. Installing sshpass..."
  sudo apt-get update
  sudo apt-get install sshpass -y
fi

# Prompt the user for the MQTT server IP address
echo "Enter the IP address of the MQTT broker: "
read SERVER_IP

# Set the SERVER_IP environment variable
export SERVER_IP

echo "Enter the number of IoT nodes desired"
read IOT_NODES

export IOT_NODES
# Define a function to run the script for each sensor
run_sensor_script() {
  sensor_number=$1
  port=$((5022+sensor_number))
  docker_name="sensor$sensor_number"
  script_name="sensor${sensor_number}_client.py"

  # Run the docker image
  echo "Running the docker image for sensor $sensor_number..."
  docker run -d -p $port:5022 --name $docker_name ghcr.io/carlosperate/qemu-rpi-os-lite:buster-legacy-2022-04-07-mu

  # Wait for the installation to finish
  echo "Waiting for the installation to finish for sensor $sensor_number..."
  for i in {180..1}; do
    printf "\rWaiting for %d seconds..." "$i"
    sleep 1
  done

  # Copy the script to the emulated machine
  echo "Copying the script to the emulated machine for sensor $sensor_number..."
  sshpass -p 'raspberry' scp -o StrictHostKeyChecking=no -P $port "$SCRIPT_DIR/sensor_scripts/$script_name" pi@localhost:/home/pi
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    tput setaf 1
    echo "Error: scp command failed with exit code $exit_code"
    tput sgr0
  fi

  # Install the paho.mqtt library
  sshpass -p raspberry ssh -o StrictHostKeyChecking=no -p $port pi@localhost "sudo apt-get update && sudo apt-get install python3-pip -y && pip3 install paho-mqtt"
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    tput setaf 1
    echo "Error: ssh command failed with exit code $exit_code"
    tput sgr0
  fi

  # Run the script on the emulated machine
  echo "Running the script on the emulated machine for sensor $sensor_number..."
  sshpass -p 'raspberry' ssh -o StrictHostKeyChecking=no -p $port pi@localhost "python3 /home/pi/$script_name $SERVER_IP" &
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    tput setaf 1
    echo "Error: ssh command failed with exit code $exit_code"
    tput sgr0
  fi
}



# Run the script for each sensor
for sensor_number in {1..$IOT_NODES}; do
  run_sensor_script $sensor_number
done
