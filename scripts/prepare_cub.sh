#!/bin/bash
set -e

CUB_ROOT='data/CUB_200_2011/'
CUB_DATA='https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'


if [[ ! -d "${CUB_ROOT}" ]]; then
    mkdir -p data/
    pushd data/
    echo "Downloading CUB_200_2011 data-set..."
    wget ${CUB_DATA}
    tar -zxf CUB_200_2011.tgz
    rm CUB_200_2011.tgz
    popd
fi


