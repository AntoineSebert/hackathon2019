# hackathon2019

## Prerequisites

### Python

version : [3.6.8](https://www.python.org/downloads/release/python-368/)

#### Checking

To check the installation & version :
```
python --version
> Python 3.6.8
```

### pip

#### Installation/Upgrade

https://pip.pypa.io/en/stable/installing/

#### Checking

To check the installation & version :
```
pip --version
> pip 19.0.3
```

#### Python Packages Index

https://pypi.org/

### Development Workflow

#### Installation/Upgrade

https://pipenv.readthedocs.io/en/latest/install/

#### Checking

To check the installation & version :
```
pipenv --version
> pipenv, version 2018.11.26
```

### Libraries

```
cd path/to/project
> pipenv lock && pipenv sync
```

## Dataset

https://www.kaggle.com/kashnitsky/medium-articles

## Run project

```python src/main.py```

## Other commands

Security checks
```pipenv check```

pylint
```pylint src/main.py```
