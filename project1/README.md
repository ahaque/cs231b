Requirements
------------
- cython (version 0.19.1 works, version 0.16 not)
- numpy
- matplotlib
- sklearn
- argparse
- pymaxflow (src included in third_party/)

Dependencies Check
------------------
To check if the necessary dependencies are installed, please run `python check_dependencies.py`. This script will check if you have the necessary packages, and also build pymaxflow if it does not already exist on your system.

Running Grabcut
---------------
There are two ways to run the algorithm
- Use `python ml.py` to evaluate the algorithm on all the images
    - You can also use `python ml.py banana1` to evaluate specifically on *banana1*
    - You can also use the parallel implementation by running `python ml_parallel.py`
    - You can also use the distributed implementation by running `python ml_remote.py`
    - Please pass in the `-h` flag to see requirements for the distributed implementation
- Use `python grabcut.py -i data_GT/book.jpg`. You can also pass in a bounding box using the `-b` argument. Please use `-h` to see all available options.

Experiments
-----------
Several experiments were performed for hyperparameter tuning. Note that these take a while to finish as we are trying various number of iterations and components. Use `python -m experiments.components_experiment` or `python -m experiments.iteration_experiment` to run the experiments on all images. You can also optionally pass in an image name (without the extension) to experiment only on that image.