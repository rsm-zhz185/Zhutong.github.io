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
import numpy as np
import scipy.special as sp
from scipy.optimize import minimize

df = pd.read_csv("blueprinty.csv")


X = df["iscustomer"].values  # predictor
y = df["patents"].values     # response
X_design = np.column_stack((np.ones_like(X), X))  # Add intercept column


def neg_log_likelihood(beta, X, y):
    lam = np.exp(X @ beta)
    return -np.sum(y * np.log(lam) - lam - sp.gammaln(y + 1))


beta_init = np.array([0.1, 0.1])


result = minimize(neg_log_likelihood, beta_init, args=(X_design, y), method='BFGS')

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
import pandas as pd
import numpy as np
import scipy.special as sp
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("blueprinty.csv")

df["age_squared"] = df["age"] ** 2

# One-hot encode region
region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)

X_covariates = pd.concat([
    df[["age", "age_squared", "iscustomer"]],
    region_dummies
], axis=1)

X_mat = X_covariates.values
y_vec = df["patents"].values

# update log-likelihood
def poisson_regression_loglik(beta, X, y):
    lam = np.exp(X @ beta)
    return -np.sum(y * np.log(lam) - lam - sp.gammaln(y + 1))
#
#
#
#
#
import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.optimize import minimize


df = pd.read_csv("blueprinty.csv")

df["age_std"] = (df["age"] - df["age"].mean()) / df["age"].std()
df["age_squared_std"] = df["age_std"] ** 2


region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)


X_df = pd.concat([df[["age_std", "age_squared_std", "iscustomer"]], region_dummies], axis=1)
X_np = X_df.to_numpy(dtype=np.float64)  
X_np = np.column_stack((np.ones(X_np.shape[0]), X_np))  
y_np = df["patents"].to_numpy(dtype=np.float64)         
beta_init = np.zeros(X_np.shape[1])

def poisson_loglik(beta, X, y):
    beta = np.asarray(beta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lin_pred = X @ beta
    lam = np.exp(lin_pred)
    return -np.sum(y * np.log(lam) - lam - sp.gammaln(y + 1))


result = minimize(poisson_loglik, beta_init, args=(X_np, y_np), method="BFGS")
beta_hat = result.x
hessian_inv = result.hess_inv
standard_errors = np.sqrt(np.diag(hessian_inv))

var_names = ["Intercept"] + list(X_df.columns)
coef_table = pd.DataFrame({
    "Coefficient": beta_hat,
    "Std. Error": standard_errors
}, index=var_names)

coef_table
#
#
#
#
#
#
import statsmodels.api as sm
import pandas as pd
import numpy as np

df = pd.read_csv("blueprinty.csv")
df["age_std"] = (df["age"] - df["age"].mean()) / df["age"].std()
df["age_squared_std"] = df["age_std"] ** 2


region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)

X_df = pd.concat([df[["age_std", "age_squared_std", "iscustomer"]], region_dummies], axis=1)
X_df = X_df.astype(np.float64)  
X_df_with_const = sm.add_constant(X_df)  

y_vec = df["patents"].astype(np.float64).values  


glm_poisson = sm.GLM(y_vec, X_df_with_const, family=sm.families.Poisson())
glm_results = glm_poisson.fit()


glm_summary = glm_results.summary2().tables[1]
glm_summary
#
#
#
#
#
#
#
#
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Step 1: Read and preprocess the data
df = pd.read_csv("blueprinty.csv")
df["age_std"] = (df["age"] - df["age"].mean()) / df["age"].std()
df["age_squared_std"] = df["age_std"] ** 2
region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)

# Step 2: Build the base design matrix
X_df = pd.concat([df[["age_std", "age_squared_std", "iscustomer"]], region_dummies], axis=1)
X_df = X_df.astype(np.float64)  # Ensure numeric format
y_vec = df["patents"].astype(np.float64).values

# Step 3: Fit the Poisson model using statsmodels
X_df_with_const = sm.add_constant(X_df, has_constant='add')
glm_poisson = sm.GLM(y_vec, X_df_with_const, family=sm.families.Poisson())
glm_results = glm_poisson.fit()

# Step 4: Create counterfactual design matrices
X_df_0 = X_df.copy()
X_df_0["iscustomer"] = 0
X_0 = sm.add_constant(X_df_0, has_constant='add').astype(np.float64)

X_df_1 = X_df.copy()
X_df_1["iscustomer"] = 1
X_1 = sm.add_constant(X_df_1, has_constant='add').astype(np.float64)

# Step 5: Generate predictions under each scenario
beta_hat_glm = glm_results.params.values
y_pred_0 = np.exp(X_0 @ beta_hat_glm)
y_pred_1 = np.exp(X_1 @ beta_hat_glm)

# Step 6: Compute average treatment effect (ATE)
diff = y_pred_1 - y_pred_0
ate = diff.mean()
ate

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

import pandas as pd

df_airbnb = pd.read_csv("airbnb.csv")

df_airbnb.info(), df_airbnb.head()

#
#
#
#
#

# 查看每列缺失情况
missing_summary = df_airbnb.isnull().sum()

# 删除包含关键变量缺失的行
df_airbnb_clean = df_airbnb.dropna(subset=[
    "bathrooms", "bedrooms",
    "review_scores_cleanliness", "review_scores_location", "review_scores_value"
])

# 保留感兴趣的列用于建模
df_model = df_airbnb_clean[[
    "number_of_reviews", "days", "room_type", "bathrooms", "bedrooms", "price",
    "review_scores_cleanliness", "review_scores_location", "review_scores_value",
    "instant_bookable"
]]

df_model.shape, df_model.isnull().sum()

#
#
#
#
#
#
