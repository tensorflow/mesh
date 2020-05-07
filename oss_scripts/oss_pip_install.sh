#!/bin/bash

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

pip install -q "tf-nightly"

# First ensure that the base dependencies are sufficient for a full import
pip install -q -e .
python -c "import mesh_tensorflow as mtf"
