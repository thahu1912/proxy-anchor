#!/bin/bash
set -e


CARS196_ROOT='data/cars196/'
CARS196_DATA='199pRuHUlDdtnvOQm-drmj-20P4GrN43T'


if [[ ! -d "${CARS196_ROOT}" ]]; then
    mkdir -p data/cars196/
    pushd data/cars196/
    echo "Downloading Cars196 data-set..."
    gdown "${CARS196_DATA}" -O cars196.tar
    tar -xvf cars196.tar
    rm cars196.tar
    popd
fi

