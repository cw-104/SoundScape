#!/bin/bash

# Check if the virtual environment directory exists
has_init=true
if [ ! -d env ]; then
    has_init=false
    echo "Creating virtual environment..."
    python3.9 -m venv env
fi

# Activate the virtual environment
source env/bin/activate

# Install dependencies if the environment was just created
if [ "$has_init" = false ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Python Environment is ready. (source env/bin/activate)"

# Prompt for additional dependencies
echo 
echo "Some additional dependencies may be required for the project to run properly."
echo
echo "bzip2 libgmp-dev libblas-dev libffi-dev libgfortran5 libsqlite3-dev libz-dev libmpc-dev libmpfr-dev libncurses5-dev libopenblas-dev libssl-dev libreadline-dev tk-dev xz-utils"
echo 

echo "mac command to install these packates globally: "
echo
echo "brew install bzip2 libgmp-dev libblas-dev libffi-dev libgfortran5 libsqlite3-dev libz-dev libmpc-dev libmpfr-dev libncurses5-dev libopenblas-dev libssl-dev libreadline-dev tk-dev xz-utils"
echo "linux command"
echo 
echo "sudo apt install bzip2 libgmp-dev libblas-dev libffi-dev libgfortran5 libsqlite3-dev libz-dev libmpc-dev libmpfr-dev libncurses5-dev libopenblas-dev libssl-dev libreadline-dev tk-dev xz-utils libsox-dev"