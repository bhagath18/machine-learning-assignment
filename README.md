## Title: How Dropout Rate Influences Representation Learning in Deep Neural Networks

This repository contains the full implementation, analysis, and tutorial corresponding to the report titled “How Dropout Rate Influences Representation Learning in Deep Neural Networks.” The project investigates how different dropout rates affect optimisation behaviour, generalisation performance, and the internal representations learned by a multilayer perceptron trained on the MNIST dataset.

## Repository Structure
README.md – Project overview and instructions
24082046_ML_BD.ipynb – Jupyter notebook with full code
24082046_ML – Final tutorial/report (PDF)
figures/ – Folder containing plots generated during experiments
• loss_curves.png
• accuracy_curves.png
• sparsity_plot.png
• cm_p0.png
• cm_p03.png
• cm_p07.png
license.txt – Usage license
## Project Overview

Dropout is widely used to reduce overfitting in deep neural networks. This project evaluates its effect by testing five dropout rates: 0.0, 0.1, 0.3, 0.5, and 0.7.

Under controlled experimental conditions, the following metrics were collected:

• Training and validation loss
• Validation accuracy
• Test confusion matrices
• Activation sparsity for each hidden layer
• Qualitative differences in internal representations

The project demonstrates not only whether dropout improves performance, but also how it shapes the structure of learned features.

## How to Run the Notebook

## Install required dependencies:
pip install torch torchvision matplotlib seaborn numpy scikit-learn

## Open the notebook:
jupyter notebook 24082046_ML_BD.ipynb

Run all notebook cells. The notebook will:
• Load MNIST
• Build a 3-layer MLP with configurable dropout
• Train models for each dropout setting
• Log losses, accuracy, and sparsity
• Generate all plots and confusion matrices

## Key Findings (Summary)

• Dropout rate 0.1 achieves the best generalisation (around 97.15% accuracy).
• Dropout rate 0.0 overfits and produces dense, co-adapted activations.
• Dropout rate 0.3 maintains strong generalisation with balanced sparsity.
• Dropout rates above 0.5 cause underfitting and representation collapse.
• Dropout meaningfully shapes internal feature representations, not just accuracy.

Full analysis is included in the report.

## License

This repository is released under the MIT License. Users may reuse or extend the code with appropriate attribution.

References

## A full references list is included in the report, covering key works such as:

• Srivastava et al. (2014) – Dropout
• Hinton et al. (2012) – Co-adaptation
• Gal and Ghahramani (2016) – Bayesian interpretation of dropout
• Goodfellow et al. (2016) – Deep Learning textbook
