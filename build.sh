#!/usr/bin/env bash
# Force Python 3.10 on Render (Render defaults to 3.13)
pyenv install -s 3.10.13
pyenv global 3.10.13

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
