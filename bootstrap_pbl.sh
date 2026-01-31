#!/bin/bash

# A bootstrap scrip that:
# - Creates a new Python 3.6 virtual environment, if not exists already
# - Installs/Updates the dependencies of a specific version (commit,branch,tag)
# - Installs/Updates the software itself with that version

# exit when any command fails
set -e
 
display_usage() { 
    echo "Creates or updates a Python virtual environment with pbl installed"
    echo ""
    echo "Usage: bootstrap_pbl.sh ENV_DIR [PBL_COMMIT]"
    echo ""
    echo "  If PBL_COMMIT is not given, the latest master version is used"
}

# if less than two arguments supplied, display usage 
if [  $# -le 0 ]; then
    display_usage
    exit 1
fi

# check whether user had supplied -h or --help . If yes display usage 
if [[ ( $# == "--help") ||  $# == "-h" ]]; then
    display_usage
    exit 0
fi

# arguments
ENV=$1
COMMIT=$2

if [[ -z "$COMMIT" ]]; then
    echo "Using master branch"
    COMMIT="master"
fi

# environment
if [[ ! -f "$ENV/bin/activate" ]]; then
    echo "Environment does not exist, creating new one."
    
    # determine appropriate python executable
    PYTHON36=`which python3.6`
    PYTHON3=`which python3`
    PYTHON=`which python`
    if [[ -n "$PYTHON36" ]]; then
        PYTHON=$PYTHON36
    elif [[ -n "$PYTHON3" ]]; then
        PYTHON=$PYTHON3
    elif [[ -n "$PYTHON" ]]; then
        PYTHON=$PYTHON
    else
        echo "Failed to find a suitable Python version!"
        exit 1
    fi
    
    PYTHON_VERSION=`$PYTHON --version`
    
    if [[ "$PYTHON_VERSION" != "Python 3.6"* ]]; then
        echo "Need Python 3.6, found $PYTHON_VERSION"
        exit 1
    fi
    
    $PYTHON -m venv "$ENV"
fi

# activate environment
echo "Activating environment"
. "$ENV/bin/activate"

# upgrade pip
echo "Updating pip"
pip install --upgrade pip

TMP_DIR=$(mktemp -d -t pbl-bootstrap-XXXXXXXXXX)
PBL_DIR="$TMP_DIR/pbl"

# Cloning pbl
echo "Cloning pbl"
git clone git@github.com:fhkiel-mlaip/pbl.git "$PBL_DIR"
git -C "$PBL_DIR" checkout --quiet $COMMIT

echo "Installing publing requirements"
pip install -r "$PBL_DIR/requirements.txt"
pip install https://github.com/b52/opengm/releases/download/v2.5/opengm-2.5-py3-none-manylinux1_x86_64.whl

echo "Installing private requirements"
pip install -e 'git+ssh://git@github.com/fhkiel-mlaip/hiwi.git@master#egg=hiwi'
pip install -e 'git+ssh://git@github.com/fhkiel-mlaip/rfl.git@master#egg=rfl'

echo "Installing pbl"
pip install -e 'git+ssh://git@github.com/fhkiel-mlaip/pbl.git@'$COMMIT'#egg=pbl'

rm -rf "$TMP_DIR"
