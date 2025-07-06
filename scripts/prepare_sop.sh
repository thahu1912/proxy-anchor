#!/bin/bash
set -e

# Prepare Stanford Online Products (SOP) dataset for CBML training
SOP_ROOT='data/online_products/'
SOP_DATA='https://www.dropbox.com/scl/fi/7icj466ds04ex7rd7kxxs/online_products.tar?rlkey=c2tp644h3uzui38tpu3l8z2uq&e=1&dl=0'

echo "Preparing Stanford Online Products dataset..."

# Create directories
mkdir -p data/

if [[ ! -d "${SOP_ROOT}" ]]; then
    mkdir -p data/
    pushd data/
    echo "Downloading Stanford Online Products dataset..."
    wget -O online_products.tar "${SOP_DATA}"
    tar -xvf online_products.tar
    rm online_products.tar
    popd
fi
