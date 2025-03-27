#!/bin/bash

# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="${DIR}/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

rm -rf ${DIR}/build
source ${DIR}/.venv/bin/activate
mkdir -p ${DIR}/build/release/build && cd ${DIR}/build/release/build
 

cmake \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=../ \
-DPY_INSTALL_PATH=$DIR/.venv/lib/python3.11/site-packages \
-DLIB_TESTS=OFF \
-DBUILD_LIB_ACS_INT=ON \
-DUSE_STDCXXFS=ON \
../../../cpp

make -j nproc
make install 


