# LocalCP4PINN

## Abstract
**LocalCP4PINN** is a framework for **local conformal prediction (Local CP)**, a distribution-free calibration method designed to quantify *spatially adaptive uncertainty* in physics-informed neural networks (PINNs) when solving forward problems.

For the detailed algorithm of the local conformal prediction, please take a look at the article below. Please remember to cite the paper if the algorithm helps :)

- 📄 **Reference paper:** [A Conformal Prediction Framework for Uncertainty Quantification in Physics-Informed Neural Networks](https://arxiv.org/abs/2509.13717)


## Example: 2D Allen–Cahn Equation
The figure below illustrates **geometric-distance-based uncertainty calibration** for the 2D Allen–Cahn equation under various heteroskedastic noise patterns, at a significance level of $\alpha = 0.05$.  
Dashed lines mark irregular, noisy regions.  

- **Rows:** Different noise patterns  
- **Columns:**  
  1. Referenced true absolute error distribution
  2. Baseline PINN interval width  
  3. CP interval width  
  4. Local CP interval width 

<p align="center">
  <img width="1803" height="1229" alt="Allen–Cahn Example" src="https://github.com/user-attachments/assets/3de9e23a-0f83-49f7-aa1b-32ba501a7f03" />
</p>

---

## Repository Structure & Usage
Each Jupyter notebook corresponds to a numerical experiment reported in the paper.  
👉 We recommend starting with the **1D Poisson equation** example to become familiar with the repository structure and workflow.

### Environment
- Recommended: **Python 3.10.17**


