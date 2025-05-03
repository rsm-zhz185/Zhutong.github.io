# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import pandas as pd

# Load the Blueprinty customer data
df = pd.read_csv("blueprinty.csv")

# Show basic info and first few rows
df.info(), df.head()

#
#
#
#
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("blueprinty.csv")


sns.set(style="whitegrid")


mean_patents = df.groupby("iscustomer")["patents"].mean()


plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="patents", hue="iscustomer", bins=20, kde=False, multiple="dodge")
plt.xlabel("Number of Patents")
plt.ylabel("Count")
plt.title("Distribution of Patents by Customer Status")
plt.legend(title="Customer", labels=["Non-customer", "Customer"])
plt.tight_layout()
plt.show()

mean_patents
#
#
#
#
#
#
#
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("blueprinty.csv")

sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="iscustomer", y="age")
plt.xticks([0, 1], ["Non-customer", "Customer"])
plt.xlabel("Customer Status")
plt.ylabel("Firm Age")
plt.title("Firm Age Distribution by Customer Status")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="region", hue="iscustomer")
plt.xlabel("Region")
plt.ylabel("Count")
plt.title("Regional Distribution by Customer Status")
plt.legend(title="Customer", labels=["Non-customer", "Customer"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.groupby("iscustomer")["age"].mean()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import numpy as np

# 定义 log-likelihood 函数（不含 beta，直接用 lambda 和 Y）
def poisson_log_likelihood(y, lam):
    """Compute the log-likelihood of Poisson model given y and lambda."""
    return np.sum(y * np.log(lam) - lam - np.log(np.math.factorial(y)))
#
#
#
#
#
#
#
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

df = pd.read_csv("blueprinty.csv")

y_obs = df["patents"].values


def poisson_log_likelihood(y, lam):
    return np.sum(y * np.log(lam) - lam - sp.gammaln(y + 1))

lam_range = np.linspace(0.1, 10, 200)
log_liks = [poisson_log_likelihood(y_obs, lam) for lam in lam_range]

plt.figure(figsize=(8, 5))
plt.plot(lam_range, log_liks, label="Log-Likelihood")
plt.xlabel("Lambda")
plt.ylabel("Log-Likelihood")
plt.title("Poisson Log-Likelihood vs. Lambda")
plt.grid(True)
plt.tight_layout()
plt.show()
#
#
#
#
#
#
import pandas as pd
from scipy.optimize import minimize

# 构造 X 和 y
X = df["iscustomer"].values  # predictor
y = df["patents"].values     # response
X_design = np.column_stack((np.ones_like(X), X))  # Add intercept column

# 定义 log-likelihood 的负值（用于最小化）
def neg_log_likelihood(beta, X, y):
    lam = np.exp(X @ beta)
    return -np.sum(y * np.log(lam) - lam - sp.gammaln(y + 1))

# 初始猜测
beta_init = np.array([0.1, 0.1])

# 最优化
result = minimize(neg_log_likelihood, beta_init, args=(X_design, y), method='BFGS')

# 提取估计值和结果状态
beta_hat = result.x
result_success = result.success
beta_hat, result_success

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
