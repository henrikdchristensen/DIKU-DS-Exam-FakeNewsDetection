# FakeNews

## Access Data files

### Sample dataset

https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv

### Big dataset

https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0

## Getting started

1. Install miniconda (or Anaconda)
   https://docs.conda.io/en/latest/miniconda.html
   Mark the option to Add to path.
2. Create conda enviroment
   conda create --name fake-news-env python=3.10.9
3. Activate enviroment
   conda activate fake-news-env
4. Install pip
   conda install pip
5. Find path to miniconda
6. Install packages via conda enviroment installed pip (DO NOT use the typically globally installed pip)
   /miniconda3/envs/fake-news-env/Scripts/pip install -r requirements.txt
7. Add the enviroment as a jupyter kernel
   ipython kernel install --name "fake-news-env" --user

# Enabling conda in Windows Powershell

First, in an **administrator** command prompt, enable unrestricted Powershell script execution
(see [About Execution Policies](https://docs.microsoft.com/en-ca/powershell/module/microsoft.powershell.core/about/about_execution_policies)):

```powershell
set-executionpolicy unrestricted
```

Make sure that the conda Script directory in is your Path.
For instance, with miniconda: `%USERPROFILE%\Miniconda3\Scripts`.

In a regular Powershell prompt check if conda is working, and update to latest version:

```powershell
conda update conda
conda --version
```

Setup conda for Powershell using the following command:

```powershell
conda init powershell
```

Finally, restart powershell. An initialization script is run every time Powershell starts.
You should now be able to activate environment with:

```powershell
conda activate <my-env>
```
