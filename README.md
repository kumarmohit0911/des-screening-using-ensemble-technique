# Deep Eutectic Solvent (DES) Formation & Solubility Prediction

**Educational / Portfolio Reproduction** of a machine learning pipeline for binary classification of Deep Eutectic Solvent (DES) formation and solubility prediction.

**Important Notice**  
This repository contains a **cleaned, anonymized, and educational version** of work originally developed during a paid remote ML internship (Aug–Oct 2025).  
- No proprietary datasets, internal company code, or confidential information is included.  
- All data used here is either **synthetic**, **publicly available benchmark molecules**, or generated via RDKit for demonstration.  
- The methodology and performance are inspired by the original project but re-implemented independently for portfolio and learning purposes.  
The original work belongs to the commissioning organization (CWEK). This repo is shared only to demonstrate personal technical skills in cheminformatics + ML.

## Project Overview

Binary classification task: predict whether a molecule is likely to form a **Deep Eutectic Solvent (DES)** or exhibit high solubility characteristics based on molecular structure.

### Key Achievements (on held-out test set – educational reproduction)
- **Accuracy**: 92.20%  
- **F1-Score**: 0.933  
- **ROC-AUC**: 0.972  
- Ensemble meta-classifier (stacking) with calibrated threshold optimization (best threshold ≈ 0.604)  
- Strong class separation (balanced precision-recall across DES and non-DES classes)

### Models Used (Stacked Ensemble)
1. **Graph Isomorphism Network (GIN)** – GNN on molecular graphs (atom features + bond types) fused with molecular descriptors  
2. **ANN** – Simple feed-forward neural network on Mordred/RDKit descriptors  
3. **XGBoost** – Gradient boosting on descriptors  
4. **LightGBM** – Another gradient boosting model on descriptors  
5. **Meta-learner** – Logistic Regression on base model probabilities (stacking)

### Tools & Libraries
- RDKit (2025.9.3) – Molecular manipulation & descriptor generation  
- Mordred – Additional 2D/3D molecular descriptors  
- PyTorch + PyTorch Geometric – GNN implementation (GINConv)  
- XGBoost & LightGBM – Tree-based models  
- scikit-learn – Metrics, scaling, stacking  
- Pandas, NumPy, Matplotlib/Seaborn – Data handling & visualization

## Results Visualizations (from test set)

## Results Visualizations (from test set)

Click on any image to view it full size.

| Plot                              | Description                                      | Preview (click to enlarge)                                                                 |
|-----------------------------------|--------------------------------------------------|--------------------------------------------------------------------------------------------|
| ROC Curve                         | AUC = 0.972                                      | [![ROC Curve](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/raw/main/roc_curve.png)](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/blob/main/roc_curve.png) |
| Precision-Recall Curve            | High precision at reasonable recall              | [![Precision-Recall Curve](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/raw/main/precision_recall_curve.png)](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/blob/main/precision_recall_curve.png) |
| Confusion Matrix (raw)            | [[54  2] [ 9 76]]                                | [![Confusion Matrix Raw](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/raw/main/confusion_matrix_raw.png)](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/blob/main/confusion_matrix_raw.png) |
| Normalized Confusion Matrix       | Shows class-wise performance                     | [![Normalized Confusion Matrix](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/raw/main/confusion_matrix_normalized.png)](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/blob/main/confusion_matrix_normalized.png) |
| Threshold vs F1 Score             | Optimal threshold ≈ 0.604                        | [![Threshold vs F1](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/raw/main/threshold_vs_f1.png)](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/blob/main/threshold_vs_f1.png) |
| Prediction Probability Distribution | Clear separation between classes                | [![Probability Distribution](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/raw/main/probability_distribution.png)](https://github.com/kumarmohit0911/des-screening-using-ensemble-technique/blob/main/probability_distribution.png) |

## How to Run (Demonstration Only)

1. Install dependencies
   ```bash
   pip install rdkit-pypi torch torchvision torchaudio torch-geometric pandas scikit-learn xgboost lightgbm joblib matplotlib seaborn

2. Use a public / synthetic SMILES dataset
  Example: Create a small demo_data.csv with columns smiles and form_DES (0/1)
  Or generate synthetic molecules via RDKit (e.g., random SMILES from PubChem subsets or RDKit's MolFromSmiles generators)

1. Run the main notebook/script :
   ```bash
   jupyter notebook ensemble_with_gin_lightgbm_stacking.ipynb

*Note: This is for educational/portfolio demonstration only. Do not use for production without proper validation and domain expert review.*

License

MIT License – for personal/educational use only.

The underlying methodology and original paid work remain the intellectual property of the commissioning organization.


**Developed by Kumar Mohit**
