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


```plaintext
/
├── CW/                     # Core coursework source code  
│   ├── main.py             # Main entry script for coursework  
│   ├── train.py            # Training script for classifiers  
│   ├── model_test.py       # Script for model testing and evaluation  
│   ├── grippers/           # Modules related to gripper models  
│   └── objects/            # Modules related to object models  
├── eval/                   # Evaluation and testing scripts  
│                           # Used to validate and analyze model performance  
├── models/                 # Model and data structure definitions  
│                           # Contains class and data structure definitions  
├── three_finger_cube.csv   # Dataset CSV files  
├── three_finger_cylinder.csv  
├── two_finger_cube.csv  
├── two_finger_cylinder.csv  
├── README.md               # This README file  
```

## Contributing

If you would like to contribute, please feel free to:

- Submit issues or bug reports.
- Create pull requests with suggested improvements.
- Contact the project maintainers for questions or collaboration.

## Contributors

- TedTang124 (Ted Tang)
- Thun78(Thun Sahacharoen)







