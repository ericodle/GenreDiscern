<p align="center">
  <img src="img/gd_logo.png" width="350" title="logo">
</p>

# GenreDiscern
Music genre discriminator via artificial neural network modeling

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

### For Mac users
Newer versions of Python lack certain SSL (secure sockets layer) certificates on MacOS.
To correct this, perform the following steps in your working environment:
#### Step 1: Install certifi
  ```sh
  python3 -m pip3 install --upgrade pip3 
  pip3 install --upgrade certifi
  ```
#### Step 2: Configure system SSL certificates
Find the path to your Python distribution's Certificates.command, then enter the following command.
Our system used Python 3.12, therefore the command looked like this:

  ```sh
  sudo /Applications/Python\ 3.12/Install\ Certificates.command
  ```
Your system should now have the required SSL certificates

Here is an easy way to use our GitHub repository.

### Install with pip

This is the easiest way to install GenreDiscern.

#### Step 1: Set up a virtual environment.

  ```sh
  python3 -m venv env
  ```

#### Step 2: Activate the virtual environment.

  ```sh
  source env/bin/activate
  ```

#### Step 3: Install GenreDiscern using pip

  ```sh
  pip3 install genrediscern
  ```

### Step 4: Run GenreDiscern

This project has a GUI for easy use.
Activate the GUI by running the following terminal command:

  ```sh
genrediscern
  ```

#### Pre-process sorted music dataset

Click "MFCC Extraction" from the Hub Window.

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


