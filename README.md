<div align="center">

# Disease severity estimation from multispectral data of Table Beets

<!-- Python Version Badge -->
[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://python.org)

</div>

## **Overview**
Code for the processing of multispectral data to obtain cercospora leaf spot disease severity.

## **Installation**
### Prerequisites
- Python 3.9
- Conda
### Steps
- Clone/download the repository 
- Set up and activate the environment 
- Download the dataset in the directory
```shell
# Downloading the directory
git clone git@github.com:saif8091/disease_multispec.git
cd disease_multispec

# Setting up environment
conda env create -f environment.yml python=3.9
conda activate cls_multispec
```
**Note**: The file should be downloaded and placed as data directory in the project root. Otherwise the code will break.

## **Preprocessing and feature generation**
This code performs the required preprocessing and oraganises the data into predict and target variable form with train, validation and test split
```shell
python make.py
```
<br>

## Project Structure

The directory structure of new project looks like this:

```
├───data
│    ├──
│
├───figures
│                  
├───preprocess           <- preprocessing and feature generation directory
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

<br>