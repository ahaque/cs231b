#!/bin/bash
#
rm -rf submission
mkdir submission
mkdir -p submission/experiments/
mkdir -p submission/third_party/
mkdir -p submission/third_party/pymaxflow

cp README.md submission/
cp __init__.py submission/
cp check_dependencies.py submission/
cp compute_bounding_boxes.py submission/
cp experiments/__init__.py submission/experiments/
cp experiments/combine_experiment.py submission/experiments/
cp experiments/components_experiment.py submission/experiments/
cp experiments/iteration_experiment.py submission/experiments/
cp gaussian.py submission/
cp gmm.py submission/
cp grabcut.py submission/
cp ml.py submission/
cp ml_parallel.py submission/
cp ml_remote.py submission/
cp third_party/__init__.py submission/third_party/
cp third_party/pymaxflow/.gitignore submission/third_party/pymaxflow/
cp third_party/pymaxflow/CHANGES.txt submission/third_party/pymaxflow/
cp third_party/pymaxflow/MANIFEST.in submission/third_party/pymaxflow/
cp third_party/pymaxflow/README.txt submission/third_party/pymaxflow/
cp third_party/pymaxflow/__init__.py submission/third_party/pymaxflow/
cp third_party/pymaxflow/block.h submission/third_party/pymaxflow/
cp third_party/pymaxflow/build.sh submission/third_party/pymaxflow/
cp third_party/pymaxflow/graph.cpp submission/third_party/pymaxflow/
cp third_party/pymaxflow/graph.h submission/third_party/pymaxflow/
cp third_party/pymaxflow/instances.inc submission/third_party/pymaxflow/
cp third_party/pymaxflow/maxflow.cpp submission/third_party/pymaxflow/
cp third_party/pymaxflow/pymaxflow.pyx submission/third_party/pymaxflow/
cp third_party/pymaxflow/setup.py submission/third_party/pymaxflow/
cp third_party/pymaxflow/test.py submission/third_party/pymaxflow/