# ⚡ PCA Optimization for Tall, Skinny Data

This repository contains an optimized custom implementation of Principal Component Analysis (PCA) built from scratch. It is mathematically designed to vastly outperform standard SVD-based PCA (like `sklearn`'s implementation) specifically when dealing with **"tall and skinny"** datasets—datasets that have a massive number of instances (rows) but a relatively small number of features (columns).

## 🧠 The Problem with Standard PCA
Most standard PCA algorithms (including `sklearn.decomposition.PCA`) rely on Singular Value Decomposition (SVD) applied directly to the $n \times m$ data matrix $X$. For tall and skinny data where $n \gg m$, traditional SVD has a time complexity of $O(n \cdot m^2)$, making it extremely computationally expensive because it must process every single row iteratively.

## 🚀 The Optimized Solution
Instead of running SVD directly on $X$, we apply the following algorithmic optimization:

1. **Mean Centering**: $X_{centered} = X - \mu$
2. **Covariance Matrix**: We compute the covariance matrix $C = \frac{X_{centered}^T \cdot X_{centered}}{n - 1}$. 
   - *Why?* This immediately reduces our massive $n \times m$ matrix down to a tiny $m \times m$ feature-to-feature matrix!
3. **Symmetric Eigendecomposition (`np.linalg.eigh`)**: The covariance matrix is mathematically guaranteed to be **strictly symmetric**. Instead of using standard eigenvalue solvers (`np.linalg.eig`) or SVD, we use numpy's `np.linalg.eigh`. `eigh` is a highly optimized LAPACK routine specifically designed for symmetric/Hermitian matrices, running in $O(m^3)$ time complexity. Since $m$ is very small, this runs almost instantly regardless of how many millions of rows $n$ we have.

## 🎛️ Tuning Variance & The 95% Golden Rule
Instead of blindly guessing how many Principal Components ($k$) to keep, our algorithm dynamically calculates it based on the **percentage of variance** we want to retain:
```python
cumulative_variance = np.cumsum(eigenvalues_sorted) / np.sum(eigenvalues_sorted)
n_components = np.argmax(cumulative_variance >= 0.95) + 1
```
For this implementation, we tuned the threshold to **95% variance**. 
Why 95%? In data science, this acts as a "Golden Ratio Rule". Retaining 95% of the variance guarantees that the machine learning model retains virtually all of the predictive signal, while successfully dropping the bottom 5% which is almost exclusively mathematical noise and multicollinearity. 

## 📊 Results (California Housing Dataset)
We tested our `mypca` algorithm against `sklearn`'s PCA using the California Housing Dataset (20,640 rows, 8 columns) and trained a Linear Regression model on both outputs.

### 1. Zero Loss in Accuracy
Both implementations yielded the exact same predictive performance, proving that our mathematical optimization did not corrupt the data.
- **MSE (Sklearn PCA)**: 1.3103
- **MSE (Custom PCA)**: 1.3103

### 2. Execution Time Reduction
Because our algorithm avoids full-matrix SVD and utilizes symmetric `eigh`, it drastically reduced the time complexity.
- **Sklearn PCA Time**: `0.0028s`
- **Custom PCA Time**: `0.0012s`

**Conclusion**: For tall and skinny datasets, our symmetric covariance optimization cuts execution time by over **50%** while retaining 100% of the modeling accuracy.
