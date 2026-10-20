# Far-bin mass (exploratory, Block 4)

Probability on bins at least 10 from the top bin, window mean. Same-task AUROC, failures above successes, 99.17% CIs, bootstrap over episodes, 2000 draws, seed 0. Beyond exp(H): far-bin mass minus its least-squares line on the window's exp(H). Harness `143729b`.

**all steps**

| Dim | window | mean far mass (fail / succ) | r with exp(H) | AUROC | beyond exp(H) |
|---|---|---|---|---|---|
| x | last 40 | 0.094 / 0.085 | 0.72 | 0.63* | 0.57 |
| x | last 50 | 0.089 / 0.081 | 0.72 | 0.64* | 0.57 |
| x | last 75 | 0.089 / 0.080 | 0.73 | 0.63* | 0.58 |
| y | last 40 | 0.054 / 0.057 | 0.72 | 0.55 | 0.46 |
| y | last 50 | 0.058 / 0.058 | 0.72 | 0.58 | 0.51 |
| y | last 75 | 0.058 / 0.058 | 0.74 | 0.58 | 0.54 |
| z | last 40 | 0.070 / 0.062 | 0.72 | 0.54 | 0.60 |
| z | last 50 | 0.069 / 0.057 | 0.75 | 0.56 | 0.57 |
| z | last 75 | 0.066 / 0.055 | 0.80 | 0.61 | 0.55 |
| roll | last 40 | 0.048 / 0.042 | 0.91 | 0.55 | 0.47 |
| roll | last 50 | 0.047 / 0.043 | 0.91 | 0.53 | 0.45 |
| roll | last 75 | 0.048 / 0.041 | 0.92 | 0.55 | 0.44 |
| pitch | last 40 | 0.075 / 0.063 | 0.87 | 0.64* | 0.57 |
| pitch | last 50 | 0.076 / 0.063 | 0.88 | 0.65* | 0.58 |
| pitch | last 75 | 0.073 / 0.061 | 0.89 | 0.67* | 0.56 |
| yaw | last 40 | 0.045 / 0.041 | 0.79 | 0.58 | 0.51 |
| yaw | last 50 | 0.042 / 0.038 | 0.76 | 0.58 | 0.49 |
| yaw | last 75 | 0.041 / 0.035 | 0.74 | 0.59 | 0.55 |
| gripper | last 40 | 0.007 / 0.008 | 0.99 | 0.43 | 0.44 |
| gripper | last 50 | 0.006 / 0.007 | 0.99 | 0.43 | 0.43 |
| gripper | last 75 | 0.005 / 0.007 | 0.99 | 0.42 | 0.45 |

**stop not in top two**

| Dim | window | mean far mass (fail / succ) | r with exp(H) | AUROC | beyond exp(H) |
|---|---|---|---|---|---|
| x | last 40 | 0.106 / 0.095 | 0.68 | 0.57 | 0.52 |
| x | last 50 | 0.102 / 0.090 | 0.68 | 0.62* | 0.56 |
| x | last 75 | 0.101 / 0.089 | 0.70 | 0.61 | 0.58 |
| y | last 40 | 0.062 / 0.068 | 0.61 | 0.55 | 0.50 |
| y | last 50 | 0.065 / 0.068 | 0.66 | 0.59 | 0.54 |
| y | last 75 | 0.066 / 0.066 | 0.67 | 0.62* | 0.59 |
| z | last 40 | 0.069 / 0.063 | 0.71 | 0.55 | 0.60 |
| z | last 50 | 0.070 / 0.061 | 0.74 | 0.56 | 0.58 |
| z | last 75 | 0.071 / 0.060 | 0.77 | 0.65* | 0.61 |
| roll | last 40 | 0.111 / 0.078 | 0.72 | 0.70* | 0.58 |
| roll | last 50 | 0.106 / 0.079 | 0.75 | 0.69* | 0.53 |
| roll | last 75 | 0.106 / 0.075 | 0.79 | 0.73* | 0.58 |
| pitch | last 40 | 0.108 / 0.088 | 0.79 | 0.62* | 0.52 |
| pitch | last 50 | 0.110 / 0.085 | 0.79 | 0.66* | 0.56 |
| pitch | last 75 | 0.103 / 0.084 | 0.83 | 0.65* | 0.53 |
| yaw | last 40 | 0.056 / 0.046 | 0.74 | 0.60 | 0.49 |
| yaw | last 50 | 0.050 / 0.042 | 0.71 | 0.59 | 0.45 |
| yaw | last 75 | 0.047 / 0.040 | 0.72 | 0.55 | 0.50 |

\* CI excludes 0.5. CIs: `far_bins.json`.
