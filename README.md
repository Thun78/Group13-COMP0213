# Group13-COMP0213

> Object-Oriented Programming Coursework - Group 13

## Project Description

This project implements a complete pipeline for dataset generation, classifier training, and testing within an object-oriented programming Python framework.  
It supports:

- Generating datasets with customizable parameters  
- Training multiple classifier models on the generated data  
- Testing and evaluating the trained classifiers  

All functionalities are accessed through a single entry point script `main.py` using command-line arguments to specify the mode of operation.

---

## Installation

### Prerequisites

- Python 3.6 or higher  
- `pip` package manager

### Install Dependencies

If a `requirements.txt` file is provided, install dependencies via:

```bash
pip install -r requirements.txt
```



## Usage

Run the main program with different modes via command-line arguments:

```bash
python main.py --mode generate_data --num_samples 150
python main.py --mode train_classifier
python main.py --mode test_planner
```

## Project Structure

/
├── CW/                # Core coursework source code  
├── eval/              # Evaluation and testing scripts  
├── models/            # Class and data model definitions  
├── data/              # (Optional) Dataset storage directory  
├── main.py            # Main script with mode-based CLI interface  
├── requirements.txt   # Python package dependencies  
└── README.md          # This file  

## Contributing

If you would like to contribute, please feel free to:

- Submit issues or bug reports.
- Create pull requests with suggested improvements.
- Contact the project maintainers for questions or collaboration.

## Contributors

- Thun78
- TedTang124 (Ted Tang)



