# Report: Handwritten Digit Recognition

## 1. Objective
Train and compare models that classify handwritten digits (0-9) from the MNIST dataset.

## 2. Dataset
- **Source:** MNIST, loaded via `tensorflow.keras.datasets.mnist`
- **Training images:** 60,000 (28x28 grayscale)
- **Test images:** 10,000 (28x28 grayscale)
- **Classes:** 10 (digits 0-9)

## 3. Models Trained
| Model | Type | Notes |
|---|---|---|
| SVM (RBF kernel) | Classical ML | Trained on 8,000-image subsample for speed |
| MLP | Classical ML (neural net) | 2 hidden layers (256, 128 units), trained on full 60k |
| CNN | Deep learning | 2 conv blocks + dense head, trained on full 60k |

## 4. Results
_Fill in after running the notebook — copy the numbers printed in Section 6 ("Compare all models")._

| Model | Accuracy | Macro F1 |
|---|---|---|
| SVM | _e.g. 0.97_ | _e.g. 0.97_ |
| MLP | _e.g. 0.97_ | _e.g. 0.97_ |
| CNN | _e.g. 0.99_ | _e.g. 0.99_ |

## 5. Sample Predictions
_Paste/insert the screenshots from the notebook's "Visualize predictions & errors" section here._

## 6. Error Analysis
_Describe the digits the model confuses most often, based on the confusion matrix (e.g. 4 vs 9, 3 vs 5, 7 vs 1). Note any patterns you observe in the misclassified samples._

## 7. Conclusion
_Summarize which model performed best and why (e.g. the CNN's convolutional filters capture spatial structure that classical models can't)._
