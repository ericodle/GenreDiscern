<p align="center">
  <img src="imgs/gd_logo.jpeg" width="350" title="logo">
</p>

# GenreDiscern
Music genre discriminator via artifical neural network modeling

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

## Prerequisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Enter this into your computer's command line interface (terminal, control panel, etc.) to check the version:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.

## Setup

Here is an easy way to use our GitHub repository.

### Step 1: Clone the repository


Open the command line interface and run:
  ```sh
  git clone https://github.com/ericodle/GenreDiscern.git
  ```

### Step 2: Navigate to the project directory
Find where your computer saved the project, then enter:

  ```sh
  cd /path/to/project/directory
  ```

If performed correctly, your command line interface should resemble

```
user@user:~/GenreDiscern-main$
```

### Step 3: Create a virtual environment: 
I like to use a **virtual environment**.
Let's make one called "gd-env"


```sh
python3 -m venv gd-env
```

A virtual environment named "gd-env" has been created. 
Let's enter the environment to do our work:


```sh
source gd-env/bin/activate
```

When performed correctly, your command line interface prompt should look like 

```
(gd-env) user@user:~/GenreDiscern-main$
```

### Step 3: Install requirements.txt

Next, let's install specific software versions so everything works properly.

  ```sh
pip3 install -r requirements.txt
  ```

### Step 4: Run GenreDiscern

This project has a GUI for easy use.
Activate the GUI by running the following terminal command:

  ```sh
python3 run_GenreDiscern.py
  ```

#### Pre-process sorted music dataset

Simply click "MFCC Extraction" from the Hub Window.

### Train a model

Simply click "Train Model" from the Hub Window.

#### Sort music using trained model

(Feature coming soon)

## Repository Files

- [ ] train_model.py

This script is called to train a neural network class on labeled MFCC data.
- [ ] models.py

This script defines the artificial neural network architectures.

- [ ] MFCC_extraction.py

This script extracts MFCCs from a curated dataset.

- [ ] model_sort.py

This script is called to sort songs by genre (future project).

## Citing Our Research

Our research paper provides a comprehensive overview of the methodology, results, and insights derived from this repository. You can access the full paper by following this link: []()

<!-- LICENSE -->

## License
This project is open-source and is released under the [MIT License](LICENSE). Feel free to use and build upon our work while giving appropriate credit.


