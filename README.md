# Overview

This project implements image compression using Singular Value Decomposition (SVD). 
It takes an image as input, compresses it, and outputs the reconstructed image while 
demonstrating compression trade-offs.


# Installation

## Prerequisites

- Python 3.10+ installed
- Git installed
- pip available in your PATH

To check:

```bash
python --version
pip --version
git --version
```


## Linux
```bash
git clone git@github.com:nickgibson1000/Image-Compression.git
cd Image-Compression

python3 -m venv venv
source venv/bin/activate 

pip install -r requirements.txt

cd src
```


## Windows (Powershell)

```bash
git clone git@github.com:nickgibson1000/Image-Compression.git
cd Image-Compression

python -m venv venv
venv\Scripts\Activate.ps1

pip install -r requirements.txt

cd src
```


## Mac

```bash
git clone git@github.com:nickgibson1000/Image-Compression.git
cd Image-Compression

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cd src
```


