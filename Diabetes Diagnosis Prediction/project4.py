import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

student_number = 22980056
seed = int(str(student_number)[-4:]) if str(student_number)[-4] != '0' else int(str(student_number)[-3:])
np.random.seed(seed)
tf.random.set_seed(seed)

data = pd.read_csv("diabetes.csv")
print(data.head())

train_data = data.sample(frac=0.9, random_state=seed)
test_data = data.drop(train_data.index)

X_train = train_data.drop(columns=["Outcome"])
y_train = train_data["Outcome"]
X_test = test_data.drop(columns=["Outcome"])
y_test = test_data["Outcome"]

log_model = LogisticRegression(max_iter=250)
log_model.fit(X_train, y_train)

coefficients = pd.DataFrame({
    "Feature": train_data.drop(columns=["Outcome"]).columns,
    "Coefficient": log_model.coef_[0]
})
print("Logistic Regression Coefficients:\n", coefficients)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]
print("Logistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
beta0 = log_model.intercept_[0]
baseline_probability = np.exp(beta0) / (1 + np.exp(beta0))
print(f"Intercept (beta0): {beta0}")
print(f"Baseline Probability (P(Outcome = 1)): {baseline_probability:.4f}")

print("Classification Report:\n", classification_report(y_test, y_pred))

sns.pairplot(data[["Glucose", "BMI", "Age", "Outcome"]], diag_kind="kde", hue="Outcome")
plt.show()

correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

NN_model = Sequential([
    Dense(1, activation="sigmoid")
])

NN_model.compile(optimizer=Adam(learning_rate=0.05),
                 loss="binary_crossentropy", 
                 metrics=["accuracy"])

NN_model.fit(X_train, y_train, epochs=100, verbose=1, batch_size=16)

y_pred_nn = (NN_model.predict(X_test) > 0.5).astype(int)
print("Neural Network Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))

print("\nComparison:")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
