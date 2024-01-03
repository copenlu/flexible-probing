# A latent-variable model for intrinsic probing

This repository contains code accompanying the paper: [A latent-variable model for intrinsic probing](https://arxiv.org/abs/2201.08214).

## Setup

These instructions assume that conda is already installed on your system.

1. Clone this repository. *NOTE: We recommend keeping the default folder name when cloning.*
2. First run `conda env create -f environment.yml`.
3. Activate the environment with `conda activate multilingual-typology-probing`.
4. (Optional) Setup wandb, if you want live logging of your runs.

### Generate data

You will also need to generate the data.

1. First run `mkdir unimorph && cd unimorph && wget https://raw.githubusercontent.com/unimorph/um-canonicalize/master/um_canonicalize/tags.yaml`
2. Download [UD 2.1 treebanks](https://universaldependencies.org/) and put them in `data/ud/ud-treebanks-v2.1`
3. Clone the modified [UD converter](git@github.com:ltorroba/ud-compatibility.git) to this repo's parent folder and then convert the treebank annotations to the UniMorph schema `./scripts/ud_to_um.sh`.
4. Run `./scripts/preprocess_bert.sh` to preprocess all the relevant treebanks using relevant embeddings. This may take a while.

## Getting Started

The `run.py` script can be used to invoke the experiments.
Commands are of the format `python run.py [ARGS] MODE [MODE-ARGS]`.
First you supply some general arguments (e.g., probe size), which are independent of the experiment.
Next you inform the type of experiment you want to run, and pass specific arguments to it.
For example, `python run.py --language eng --trainer poisson --gpu --attribute Number manual --dimensions 1 2 3` will train a probe (with the default hyperparameters) on the English (number) dataset using the Poisson variational family.

Alternatively, you can run `make experiments_01`, `make experiments_02`, `make experiments_deep_01`, etc. to reproduce the results. Look at the makefile for other options.
Plots for the paper were generated with `01_make_bar_graphs_all.py`, etc.
Hyperparameter tuning is done using Weights & Biases sweeps. To begin one, run `wandb sweep sweep.yml`

## Extra Information

### Citation

If this code or the paper were useful to you, consider citing it:


```
@article{
    stanczak-etal-2023-latent-variable,
    title={A Latent-Variable Model for Intrinsic Probing},
    volume={37},
    url={https://ojs.aaai.org/index.php/AAAI/article/view/26593},
    DOI={10.1609/aaai.v37i11.26593},
    number={11},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Stańczak, Karolina and Torroba Hennigen, Lucas and Williams, Adina and Cotterell, Ryan and Augenstein, Isabelle},
    year={2023},
    month={Jun.},
    pages={13591-13599}
}
```

### Contact

To ask questions or report problems, please open an [issue](https://github.com/copenlu/flexible-probing/issues).
