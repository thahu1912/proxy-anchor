#!/bin/bash
set -e


CARS196_ROOT='data/cars196/'
CARS196_DATA='https://www.dropbox.com/scl/fi/0a7bi36c55qb9bhm4pzzu/cars196.tar?rlkey=8rld9uelq39iw8r5moootq19g&dl=0'
CARS196_DEVKIT='1rdETvCItDkiWecqL_L2F_qQOSNUcmdSd'


if [[ ! -d "${CARS196_ROOT}" ]]; then
    mkdir -p data/
    pushd data/
    echo "Downloading Cars196 data-set..."
    wget -O cars196.tar "${CARS196_DATA}"
    gdown "${CARS196_DEVKIT}" -O cars_annos.zip
    tar -xvf cars196.tar
    unzip cars_annos.zip
    rm cars196.tar
    rm cars_annos.zip
    mv cars_annos.mat cars196/

    popd
fi

