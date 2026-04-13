#!/bin/bash

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No virtual environment is currently activated."
    echo "Please activate your venv first, then re-run this script."
    exit 1
fi

VENV_NAME=$(basename "$VIRTUAL_ENV")

echo "Detected venv: $VENV_NAME"

if ! python -c "import ipykernel" &>/dev/null; then
    echo "ipykernel not found. Installing..."
    pip install ipykernel
else
    echo "ipykernel already installed."
fi

python -m ipykernel install --user --name "$VENV_NAME" --display-name "$VENV_NAME"

echo "Kernel '$VENV_NAME' registered successfully."
