# MDFT_NN

A recurrent neural network for learning MDFT parameters.

## Install package

Run the following line for installation

```bash
pip install git+https://github.com/Rahgooy/MDFT.git@master
```

## Reproducing the results of my thesis

### Data generation

The generated data are available in the `/data` folder. However if you want to regenerate them use the following instructions. The data generation method is written in Matlab.

- Run `./matlab/BuildsimMultiMDF.m` to build the `mex` files if you are not using a mac system.

- Run `./matlab/data_generator.m` file to generate the data.

### Maximum Likelihood (MLE) results

The results are saved in the `./results/MLE` folder.
If you want to run the model run `./matlab/learn_multi.m` file.

### Neural Net (NN) results

The results are saved in the `./results/NN` folder.
If you want to run the `NN` model in the root folder run `bash ./scripts/learn.sh`.

### Summary of the results

To see the summary of the results saved in the `./results/MLE` and `./results/NN` folders run:

```bash
python summarize.py
```
