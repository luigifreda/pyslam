#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${SCRIPT_DIR}
cd modules/dbow3
rm -Rf build 
rm -Rf install

cd ${SCRIPT_DIR}
rm -Rf build 
rm -Rf lib