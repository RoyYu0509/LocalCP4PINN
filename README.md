# LocalCP4PINN

## Abstract
**LocalCP4PINN** is a framework for **local conformal prediction**, a distribution-free conformal calibration method designed to quantify local uncertainty in physics-informed neural networks (PINNs) when solving the forward problem.

<img width="983" height="654" alt="image" src="https://github.com/user-attachments/assets/47b5acf9-ef7c-45fe-b3b6-ffe0da69629a" />

### UQ on 2D Allen-Cahn Example
Geometric-distance PINN uncertainty calibration for the 2D Allen–Cahn equation under different heteroskedastic noise patterns. The irregular noisy regions are distinguished with dashed lines.
- **Row:** Different noise patterns. 
- **Column:** The true absolute-error distributions and the interval widths for the baseline model, CP, and local CP at $\alpha=0.05$, respectively, from the first column to the fourth column. 

<img width="1803" height="1229" alt="image" src="https://github.com/user-attachments/assets/3de9e23a-0f83-49f7-aa1b-32ba501a7f03" />


## Reading Guide
Each Jupyter notebook in this repository corresponds to a numerical experiment presented in the paper.
You can start with the simplest 1D Poisson example to get yourself familiar with the structure of this repository.

***NOTE:***
- The local CP examples are not yet updated in the article. For more information, please refer to the updated article.
- Use Python 3.10.17.
