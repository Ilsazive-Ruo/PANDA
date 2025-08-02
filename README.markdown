# PANDA

## Requirements

- **Python version**: Python 3.8+
- **Dependencies**: Listed in `requirements.txt`. Install with:
  ```bash
  pip install -r requirements.txt
  ```
- **Hardware**: Recommended NVIDIA GPU with at least 8GB of VRAM for training
- **Other tools**: Conda

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ilsazive-Ruo/PANDA.git
   cd PANDA
   ```
2. Create and activate a virtual environment:
   ```bash
   conda create -n env python=3.9
   conda activate env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

##Task Definitions
This repository includes multiple prediction tasks aligned with those described in the associated publication. The tasks and their corresponding labels in the code are as follows:

- **SYN and PV**:
Correspond to the Pharmacological Prediction tasks, referred to as the PP set in the paper.
- **SA**:
Refers to the API–Excipient Design task.
- **FSA**:
Refers to the Manufacturing Optimization task.

## Usage
Explain how to run the code, including examples:

- **Training**:
  ```bash
  python train.py  -b batchsize -e epochs
  ```
- **Evaluation**:
  ```bash
  python CV.py  -b batchsize -e epochs
  ```
- **Prediction with demo data**:
  ```bash
  python predictor.py -i data/demo.csv -p predicted -m weights/PANDA.pth -t SA
  ```

## Project Structure
```
project/
├── data/               # Input/output data
├── weights/            # Pre-trained models
├── predicted/          # Demo outputs
├── tool/               # scripts for data preprocessing, analysis, etc.
├── CV.py               # 10 folds cross validation
├── PANDA.py            # model structures
├── train.py            # model training
├── predictor.py        # script for predicting custom data
├── requirements.txt    # Dependencies
├── README.md           # This file
└── LICENSE             # License file
```

## License

- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📦 Supplementary Data (Molecule graph)
The full dataset for this study is available at:
🔗 https://doi.org/10.5281/zenodo.16038069
