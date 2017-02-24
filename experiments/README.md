
This page describes how to reproduce the experimental results reported in Bolt's KDD paper.

## Install Dependencies

To run the experiments, you will first need to obtain the following tools / libraries, and datasets.

### C++ Code

- [Bazel](http://bazel.build), Google's open-source build system

### Python Code:
- [Joblib](https://github.com/joblib/joblib) - for caching function output
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn) - for k-means
- [Kmc2](https://github.com/obachem/kmc2) - for k-means seeding
- [Pandas](http://pandas.pydata.org) - for storing results and reading in data
- [Seaborn](https://github.com/mwaskom/seaborn) - for plotting, if you want to reproduce our figures

### Datasets:
- [Sift1M](http://corpus-texmex.irisa.fr) - 1 million Sift descriptors. This page also hosts the Sift10M, Sift100M, Sift1B, and Gist1M datasets (not used in the paper)
- [Convnet1M](https://github.com/una-dinosauria/stacked-quantizers) - 1 million convnet image descriptors
- [MNIST](http://yann.lecun.com/exdb/mnist/) - 60,000 images of handwritten digits; perhaps the most widely used dataset in machine learning
- [LabelMe](http://www.cs.toronto.edu/~norouzi/research/mlh/) - 22,000 GIST desciptors of images


## Reproduce Timing / Throughput results

The timing scripts available include:
 - `time_bolt.sh`: Computes Bolt's time/throughput when encoding data vectors and queries, scanning through a dataset given an already-encoded query, and answering queries (encode query + scan).
 - `time_pq_opq.sh`: This is the same as `time_bolt.sh`, but the results are for Product Quantization (PQ) and Optimized Product Quantization (OPQ). There is no OPQ scan because the algorithm is identical to PQ once the dataset and query are both encoded.
 - `time_popcount.sh`: Computes the time/throughput for Hamming distance computation using the `popcount` instruction.
 - `time_matmul_square.sh`: Computes the time/throughput for matrix multiplies using square matrices.
 - `time_matmul_tall.sh`: Computes the time/throughput for matrix multiplies using tall, skinny matrices.

To reproduce the timing experiments using these scripts:

1. Run the appropriate shell script in this directory. E.g.:
    ```
    $ bash time_bolt.sh
    ```

2. Since we want to be certain that the compiler unrolls all the relevant loops / inlines code equally for all algorithms, the number of bytes used in encodings and lengths of vectors are constants at the tops of the corresponding files. To assess all the combinations of encoding / vector lengths used in the paper, you will have to modify these constants and rerun the tests multiple times. Ideally, this would be automated, but it isn't; pull requests welcome.

| Test          |   File        | Constants and Experimental Values
|:----------    |:----------    |:----------
| time_bolt.sh  | cpp/test/quantize/profile_bolt.cpp | M = {8,16,32}, ncols={64, 128, 256, 512, 1024}
| time_pq_opq.sh  | cpp/test/quantize/profile_pq.cpp | M = {8,16,32}, ncols={64, 128, 256, 512, 1024}
| time_popcount.sh  | cpp/test/quantize/profile_multicodebook.cpp | M = {8,16,32}
| time_matmul_square.sh  | cpp/test/quantize/profile_bolt.cpp | sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
| time_matmul_tall.sh  | cpp/test/quantize/profile_bolt.cpp | sizes = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]

The `sizes` for the `time_matmul_*` scripts only need to be modified once, since the code iterates over all provided sizes; the tests are just set to run a subset of the sizes at the moment so that one can validate that it runs without having to wait for minutes.


## Reproduce Accuracy Results

### Clean Datasets

Unlike the timing experiments, the accuracy experiments depend on the aforementioned datasets. We do not own these datasets, so instead of distributing them ourselves, we provide code to convert them to the necessary formats and precompute the true nearest neighbors.

After downloading any dataset from the above links, start an IPython notebook:

```bash
    cd python/
    ipython notebook &
```

then open `munge_data.ipynb` and execute the code under the dataset's heading. This code will save a query matrix, a database matrix, and a ground truth nearest neighbor indices matrix. For some datasets, it will also save a training database matrix. Computing the true nearest neighbors will take a while for datasets wherein these neighbors are not already provided.

Once you have the above matrices for the datasets saved, go to python/datasets.py and fill in the appropriate path constants so that they can be loaded on your machine.

### Run the Accuracy Experiments

To compute the correlations between the approximate and true dot products:
```bash
$ bash correlations_dotprod.sh
```

To compute the correlations between the approximate and true squared Euclidean distances:
```bash
$ bash correlations_l2.sh
```

To compute the recall@R:
```bash
$ bash recall_at_r.sh
```

## Compare to Our Results

All raw results are in the `results/` directory. `summary.csv` files store aggregate metrics, and `all_results.csv` files store metrics for individual queries. The latter are mostly present so that we can easily plot confidence intervals with Seaborn.


## Notes

At present, Bolt has only been tested with Clang on OS X. The timing / throughput results in the paper are based on compiling with Xcode, not Bazel, although this shouldn't make a difference.

Feel free to contact us with any and all questions.

<!-- Also, because this code is a cleaned-up version of what we originally ran, there is a small but nonzero chance we've subtly broken something or diminished the performance. Please don't hesitate to contact us if you believe this is the case. -->