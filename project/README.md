# CZ4003

Nanyang Technological University  
School of Computer Science and Engineering

Academic Year 2020-2021 Semester 1

source code for CZ4003 Computer Vision project

This README is best viewed using a text editor that supports Markdown.

All commands are for Unix-based systems.

---

## files

- ocr.ipynb: notebook that shows usage of Otsu algorithm
- otsu.py: file that defines Otsu algorithm and its variants as functions

---

## set-up

1. [install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it
2. create a virtual environment with the necessary libraries: `conda env create --name cv_venv --file cv_venv.txt`
3. install Tesseract: `sudo apt-get install tesseract-ocr libtesseract-dev`
4. you are done setting up - deactivate the virtual environment if you want: `conda deactivate`

---

## view / run code

1. change directory to this repository: `cd <path_to_this_repo>`
2. activate the virtual environment: `conda activate cv_venv`
3. start jupyter lab: `jupyter lab`
4. click on ocr.ipynb to view it
    - if there is an issue with python path related to 'otsu', create a new cell in the notebook, then copy and paste the Otsu functions into the cell and run it

---

## dependencies / libraries used

- Python libraries listed in `cv_venv.txt`

