# LocalCP4PINN (Unpublished)

## Abstract
**LocalCP4PINN** is a framework for **local conformal prediction (Local CP)**, a distribution-free calibration method for quantifying *spatially adaptive uncertainty* in physics-informed neural networks (PINNs) when solving forward problems.

<img width="983" height="654" alt="image" src="https://github.com/user-attachments/assets/47b5acf9-ef7c-45fe-b3b6-ffe0da69629a" />

## Example: 2D Allen–Cahn Equation
The figure below demonstrates **geometric-distance-based uncertainty calibration** for the 2D Allen–Cahn equation under different heteroskedastic noise patterns.  
Dashed lines highlight irregular, noisy regions.  

- **Rows:** Different noise patterns  
- **Columns:** (1) True absolute error, (2) baseline PINN interval width, (3) CP interval width, and (4) Local CP interval width at significance level $\alpha = 0.05$  

<img width="1803" height="1229" alt="image" src="https://github.com/user-attachments/assets/3de9e23a-0f83-49f7-aa1b-32ba501a7f03" />

## Repository Structure & Usage
Each Jupyter notebook corresponds to a numerical experiment reported in the paper.  
We recommend starting with the **1D Poisson equation** example to become familiar with the repository structure and workflow.  

### Notes
- Local CP examples are still being integrated into the manuscript. For detailed results and presentations, please refer to the upcoming article update.  
- Recommended environment: **Python 3.10.17**  
