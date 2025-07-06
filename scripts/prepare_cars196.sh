#!/bin/bash
set -e


CARS196_ROOT='data/cars196/'
CARS196_DATA='https://www.dropbox.com/scl/fi/0a7bi36c55qb9bhm4pzzu/cars196.tar?rlkey=8rld9uelq39iw8r5moootq19g&dl=0'
CARS196_DEVKIT='1t2CzZy3oEpZnb7vPwp68Z7AwyI43osuS'


if [[ ! -d "${CARS196_ROOT}" ]]; then
    mkdir -p data/
    pushd data/
    echo "Downloading Cars196 data-set..."
    wget -O cars196.tar "${CARS196_DATA}"
    gdown "${CARS196_DEVKIT}" -O cars_devkit.tgz
    tar -xvf cars196.tar
    tar -xvf cars_devkit.tgz
    rm cars196.tar
    rm cars_devkit.tgz

    popd
fi

