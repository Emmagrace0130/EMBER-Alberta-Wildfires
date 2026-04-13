#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <venv-name>"
    exit 1
fi

VENV_NAME="$1"

python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "Virtual environment '$VENV_NAME' created and activated."
echo "To activate it later, run: source $VENV_NAME/bin/activate"
