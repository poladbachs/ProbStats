import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

data = pd.read_csv("housing_data.csv")
print("Data Loaded:")
print(data.info())
print(data.head())

# Exploratory Data Analysis
sns.boxplot(x=data['medv'])
plt.title("Boxplot of Housing Prices (medv)")
plt.show()

sns.histplot(data['lstat'], bins=30, kde=True)
plt.title("Histogram of LSTAT")
plt.show()

sns.scatterplot(x=data['rm'], y=data['medv'])
plt.title("Scatter Plot: RM vs MEDV")
plt.xlabel("Average Number of Rooms (rm)")
plt.ylabel("Median Value of Homes (medv)")
plt.show()

# Data Splitting
student_number = 22980056
seed = int(str(student_number)[-4:]) if str(student_number)[-4] != '0' else int(str(student_number)[-3:])
np.random.seed(seed)

total_rows = len(data)
train_size = int(0.9 * total_rows)

all_indices = list(range(total_rows))
train_indices = np.random.choice(all_indices, train_size, replace=False)
test_indices = [i for i in all_indices if i not in train_indices]

train_data = data.iloc[train_indices]
test_data = data.iloc[test_indices]
print(f"Training Set Size: {len(train_data)}, Test Set Size: {len(test_data)}")

# Linear Regression Model Fitting
X_train = sm.add_constant(train_data[['rm', 'lstat', 'crim', 'indus']])
y_train = train_data['medv']
X_test = sm.add_constant(test_data[['rm', 'lstat', 'crim', 'indus']])
y_test = test_data['medv']

model = sm.OLS(y_train, X_train).fit()
print("Model Summary:")
print(model.summary())

# Bootstrap
n_iterations = 1000
bootstrap_coeffs = []

for _ in range(n_iterations):
    sample_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_resampled = X_train.iloc[sample_indices]
    y_resampled = y_train.iloc[sample_indices]

    bootstrap_model = sm.OLS(y_resampled, X_resampled).fit()
    bootstrap_coeffs.append(bootstrap_model.params)

bootstrap_coeffs = np.array(bootstrap_coeffs)
conf_intervals = np.percentile(bootstrap_coeffs, [2.5, 97.5], axis=0)

# Confidence interval for indus
indus_index = list(X_train.columns).index('indus')
indus_conf_int = conf_intervals[:, indus_index]
print(f"Confidence Interval for indus: {indus_conf_int}")

# Confidence interval for rm
rm_index = list(X_train.columns).index('rm')
rm_conf_int = conf_intervals[:, rm_index]
print(f"Confidence Interval for rm: {rm_conf_int}")

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Predicted AND Actual Values
plt.scatter(y_test, y_pred)
plt.title("Predicted vs Actual Home Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()