#!/bin/bash
set -e


CARS196_ROOT='data/cars196/'
CARS196_DATA='1se8UcOW1hu752bXCw4GyYzkJkCO4Kt6Z'

if [[ ! -d "${CARS196_ROOT}" ]]; then
    mkdir -p data/
    pushd data/
    echo "Downloading Cars196 data-set..."
    gdown "${CARS196_DATA}" -O cars196.zip
    unzip cars196.zip
    rm cars196.zip
    popd
fi

