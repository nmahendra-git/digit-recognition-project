# Handwritten Digit Recognition (MNIST)

Machines often struggle to recognize handwritten digits due to variation in writing style. This project trains and compares multiple models — a classical SVM, an MLP, and a small CNN — to classify handwritten digits (0–9) from the MNIST dataset.

## Project Structure

```
digit-recognition-project/
├── notebooks/
│   └── 01_digit_recognition.ipynb   # Main notebook: run this end-to-end
├── src/
│   ├── data_loader.py               # Loads & preprocesses MNIST
│   ├── train_baseline.py            # SVM + MLP training script
│   ├── train_cnn.py                 # CNN training script
│   └── visualize.py                 # Plotting helpers
├── models/                          # Saved trained models (created after training)
├── reports/
│   └── report.md                    # Report template — fill in your results
├── data/                            # (MNIST downloads here automatically)
├── requirements.txt
├── .gitignore
└── README.md
```

## Tech Stack
- **Language:** Python
- **Libraries:** NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/digit-recognition-project.git
   cd digit-recognition-project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Option A — Jupyter Notebook (recommended for learning)
```bash
jupyter notebook notebooks/01_digit_recognition.ipynb
```
Run the cells top to bottom. This covers data loading, EDA, training all three models, evaluation, and visualization.

### Option B — Standalone scripts
```bash
# Train SVM + MLP baselines
python src/train_baseline.py

# Train the CNN
python src/train_cnn.py
```
Both scripts save trained models into the `models/` folder.

## Learning Outcomes
- How to load and preprocess an image dataset for both classical ML and deep learning
- CNN fundamentals: convolution, pooling, dropout
- How classical ML baselines (SVM/MLP) compare to a CNN on an image task
- Evaluating classifiers with accuracy and macro F1, and reading a confusion matrix

## Results
See [`reports/report.md`](reports/report.md) for the filled-in results table, sample predictions, and error analysis once you've run the notebook.

## Deliverables Checklist
- [x] Training notebook (`notebooks/01_digit_recognition.ipynb`)
- [x] Saved model / weights (generated in `models/` after running training)
- [x] Report with metrics + sample predictions (`reports/report.md`)

## License
This project is for educational purposes.
