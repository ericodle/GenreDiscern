<p align="center">
  <img src="img/gd_logo.png" width="350" title="logo">
</p>

# GenreDiscern
Music genre discriminator via artificial neural network modeling

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

## Prerequisite

- Python3
- VSCode
- an Nvidia GPU
- Preferably a Linux system

## Setup

#### Step 1: Clone the repo and move into the working directory

  ```sh
  git clone https://github.com/ericodle/GenreDiscern
  cd GenreDiscern
  ```

Bonus tip: open in VSCode.

  ```sh
  code .
  ```

#### Step 2: Set up a virtual environment.

  ```sh
  python3 -m venv env
  ```

#### Step 3: Activate the virtual environment.

  ```sh
  source env/bin/activate
  ```

#### Step 4: Install dependencies using pip

  ```sh
pip3 install -r requirements.txt
```

### Step 4: Run GenreDiscern

  ```sh
python3 ./src/run_genrediscern.py
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


