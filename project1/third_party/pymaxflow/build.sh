#!/bin/bash

if [ ! -f maxflow.cpp ]; then
    cd third_party/pymaxflow
fi
python setup.py build_ext --inplace
mv project1/third_party/pymaxflow/pymaxflow.so .
rm -rf project1/