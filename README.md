# Fake News Corpus

The overall goal of this project is to be able to detect fake news. For this, we have created and trained models to classify news articles.

## Access the Fake News Corpus dataset

The dataset used in the project is scraped articles into a large .csv file (>32 GB, 8+ mio. rows). There exists also a sample .csv file, which only contains a small number of articles (250 rows).
For the large dataset, all the chunks need to be downloaded, and 7-zip can be used to unpack the dataset.

- Sample: https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv
- Large: https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0

## Access the LIAR dataset

https://paperswithcode.com/dataset/liar

## Getting started

1. Install miniconda (or Anaconda)
   https://docs.conda.io/en/latest/miniconda.html
   Mark the option _Add to path_.
2. Create conda environment
   ```powershell
   conda create --name fake-news-env python=3.10.9
   ```
3. Activate environment
   ```powershell
   conda activate fake-news-env
   ```
4. Install pip
   ```powershell
   conda install pip
   ```
5. Find path to miniconda
6. Install packages via conda environment installed pip (DO NOT use the typically globally installed pip)
   ```powershell
   <path-to-miniconda>/miniconda3/envs/fake-news-env/Scripts/pip install -r requirements.txt
   ```
7. Add the environment as a jupyter kernel
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

## Nice-to-have extensions in VS Code

- Python Extension Pack
- Jupyter Notebook

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

The core of the project is made in python scripts mainly due to collaboration and version control.
Where all the modeling and statistics are made in a Jupyter Notebook.

### Placement of data set files

The data set files should be placed in `datasets/large` and `datasets/sample`.

### Source code

#### _filehandling.py_

This Python module in overall creates a randomly sampled data set from the raw data set file. To achieve this the module have different helper functions. Fist of all the csv_to_h5() function is converting a csv file to h5 and a function to shuffle the entries in h5 format. Secondly a function to convert back again called h5_to_csv().

The code (correct sequence of functions) to generate the randomly shuffled data set can be done by running the script:

```powershell
python pipeline.py
```

Where choices can be made to either generate a cleaned sample or cleaned large data set of a size specified.

#### _pipeline.py_

This Python module contains the entire preprocessing pipeline. The different functionalities are structured in classes, which makes it possible to chain multiple classes. E.g. Clean_data(), Stem(), Remove_stopwords(), etc. In this way we can easily combine and test multiple preprocessing techniques. This module also contain the function get_dataframe_with_distribution(), where you can specify a distribution and a split (how train, validation and test classes should be split). In addition the module also contains classes that is used for statistics.

The code (correct sequence of functions) can be run by running the script:

```powershell
python filehandling.py
```

#### _simple_model.py_

TODO

#### _advanced_model.ipynb_

The jupyter notebook contains the code for testing and training different neural networks for the fake news detection.

### _logistic_model.ipynb_

This Jupyter Notebook contains the code for testing and training different logistic and simple models. It uses extensively the model_tests.py for creating different vector representations and to get evaluation metrics of the different models.

### _model_tests.py_

This Python module contains functions to create different vector representations, such as count-vectors, td-idf vectors etc. It also has functions that should make it easier to test many different models and input representations. It also contain functions to find different evaluation metrics.

### _liar.py_

This Python module contains the correct pipeline to generate a cleaned liar dataset.

The code (correct sequence of functions) can be run by running the script:

```powershell
python liar.py
```

### _stats.py_

This Python module contains the code for the statistics of the project.

The Statistics class contains general functions for getting the data and plotting the data.

The FakeNewsCorpus class contains the interface for getting the data from the Fake News Corpus dataset and plotting the data using the Statistics class.

The Liar class contains the interface for getting the data from the Liar dataset and plotting the data using the Statistics class.

The Statistics_FakeNewsCorpus_vs_Liar uses the FakeNewsCorpus and Liar classes to get the data and plot the data for the comparison between the two datasets.

### _stats.ipynb_

This Jupyter Notebook contains plot of the statistics of the project. It uses the stats.py file to get the data.
