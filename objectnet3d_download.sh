#!/bin/bash

# Get parent directory of this file
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Relevant strings
OBJECT_NET_PATH=${DIR}/ObjectNet3D
FTP_URL="ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D"

IMAGES="ObjectNet3D_images.zip"
CADS="ObjectNet3D_cads.zip"
ANNOTATIONS="ObjectNet3D_annotations.zip"
IMAGE_SETS="ObjectNet3D_image_sets.zip"

# Make ObjectNet3D directory (exists ok)
mkdir -p ${OBJECT_NET_PATH}

if [ -d "${OBJECT_NET_PATH}/Image_sets" ]; then
    echo "Splits Already Downloaded"
else
    wget -O "${OBJECT_NET_PATH}/Image_sets.zip" "${FTP_URL}/${SPLITS}"
    unzip -d "${OBJECT_NET_PATH}/Image_sets" "${OBJECT_NET_PATH}/Image_sets.zip" && rm "${OBJECT_NET_PATH}/Image_sets.zip"
fi


if [ -d "${OBJECT_NET_PATH}/annotations" ]; then
    echo "Annotations Already Downloaded"
else
    wget -O "${OBJECT_NET_PATH}/annotations.zip" "${FTP_URL}/${ANNOTATIONS}"
    unzip -d "${OBJECT_NET_PATH}/annotations" "${OBJECT_NET_PATH}/annotations.zip"
fi
