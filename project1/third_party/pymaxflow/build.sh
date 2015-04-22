#!/bin/bash

if [ ! -f maxflow.cpp ]; then
    cd third_party/pymaxflow
fi
output="$(python setup.py build_ext --inplace)"
mv `echo $output | grep "pymaxflow.so" | rev | cut -d' ' -f 1 | rev` .
rm -rf project1/