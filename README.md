# FakeNews

The overall goal of this project is to be able to detect fake news. For this, we have created and trained models in order to classify news articles.

## Access dataset

The dataset used in the project are scraped articles into a large .csv file (>32 GB, 8+ mio. rows). There exists also a sample .csv file, which only contains a small amount of articles (250 rows).
For the large dataset, all the chunks, needs to be downloaded, and 7-zip can be used to unpack the dataset.

- Sample: https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv
- Large: https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0

## Getting started

1. Install miniconda (or Anaconda)
   https://docs.conda.io/en/latest/miniconda.html
   Mark the option _Add to path_.
2. Create conda enviroment
   ```powershell
   conda create --name fake-news-env python=3.10.9
   ```
3. Activate enviroment
   ```powershell
   conda activate fake-news-env
   ```
4. Install pip
   ```powershell
   conda install pip
   ```
5. Find path to miniconda
6. Install packages via conda enviroment installed pip (DO NOT use the typically globally installed pip)
   ```powershell
   <path-to-miniconda>/miniconda3/envs/fake-news-env/Scripts/pip install -r requirements.txt
   ```
7. Add the enviroment as a jupyter kernel
   ```powershell
   ipython kernel install --name "fake-news-env" --user
   ```
8. You're now able to run python scripts and open `jupyter lab` via:
   ```powershell
   python <script-name>
   ```
   and
   ```powershell
   jupyter lab
   ```

## Run in VS Code

Nice-to-have extensions:

- Python Extension Pack
- Jupyter

## Enabling conda in Windows Powershell

1. First, in an **administrator** command prompt, enable unrestricted Powershell script execution (see [About Execution Policies](https://docs.microsoft.com/en-ca/powershell/module/microsoft.powershell.core/about/about_execution_policies)):
   ```powershell
   set-executionpolicy unrestricted
   ```
2. Make sure that the conda Script directory in is your Path.
   For instance, with miniconda: `%USERPROFILE%\Miniconda3\Scripts`.
   In a regular Powershell prompt check if conda is working, and update to latest version:
   ```powershell
   conda update conda
   conda --version
   ```
3. Setup conda for Powershell using the following command:
   ```powershell
   conda init powershell
   ```
4. Finally, restart powershell. An initialization script is run every time Powershell starts.
   You should now be able to activate environment with:
   ```powershell
   conda activate <my-env>
   ```

## Project

The core of the project is made in python scripts mainly due to collabration and version control.
Where all the modelling and statistics are made in a Jupyter Notebook.

### Placement of data set files

The data set files should be placed in `datasets/large` and `datasets/sample`.

### Source code

#### _filehandling.py_

This code in overall create a randomly sampled data set from the raw data set file. It has a function to just create one single randomly data set file and a function to split the data set into training, validation, and testing data set. These to functions has helpter functions which for example remove unwanted rows, such as those with empty content or types which is not of interest.

#### _pipeline.py_

TODO

#### _preprocessing.py_

TODO

#### _simple_model.py_

TODO

#### _models.py_

TODO

#### _book.ipynb_

TODO

#### _main.py_

TODO

### Tests

TODO
