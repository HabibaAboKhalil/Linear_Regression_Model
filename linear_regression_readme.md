# 💼 Linear Regression Model — Salary Prediction  

## 📌 Overview  
This project demonstrates how to use **Linear Regression** to predict an employee’s salary based on their years of experience.  
It’s a simple yet powerful example of applying **Supervised Machine Learning** using Python and scikit-learn.  

---

## 🧠 What is Linear Regression?  
**Linear Regression** is a statistical model used to predict a continuous value (like salary) based on one or more input features (like years of experience).  
It finds the **best-fitting straight line** that represents the relationship between the input (X) and the output (Y).  

🧾p Formula:  
\[
y = aX + b
\]  
Where:  
- **a** → Slope (how much salary increases per year of experience)  
- **b** → Intercept (the base salary when experience = 0)  

---

## 🧬 Steps in the Code  

### 1. Importing the Dataset  
```python
import pandas as pd
dataset = pd.read_csv("/content/Salary_Data.csv")
x = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values
```
We load the dataset and separate features (**x = Years of Experience**) and target (**y = Salary**).

---

### 2. Splitting Data into Training and Testing Sets  
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
```
This splits the data into:
- **80%** for training  
- **20%** for testing  

---

### 3. Training the Linear Regression Model  
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
```
We create a regression object and train it using the training data.

---

### 4. Making Predictions  
```python
y_pred = regressor.predict(x_test)
print(y_pred)
```
We predict salaries for the test set and compare them with actual values.

---

### 5. Model Evaluation  
```python
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))
```
We use **Mean Absolute Error (MAE)** to check how accurate the predictions are.

---

### 6. Regression Equation  
```python
A = regressor.coef_   # Slope
b = regressor.intercept_  # Intercept
print("y =", A, "X +", b)
```
This gives the final regression line equation used to make predictions.

Example:
```python
print("The salary of the employee is:", regressor.predict([[10.5]]))
```
Predicts the salary for **10.5 years of experience**.

---

### 7. Visualizing the Results  

**Training Data Visualization**
```python
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
```

**Testing Data Visualization**
```python
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
```
- 🔴 Red dots → Actual salaries  
- 🔵 Blue line → Predicted regression line  

---

## 🏁 Conclusion  
This model successfully predicts employee salaries based on experience using **Linear Regression**.  
It’s a foundational step for understanding predictive analytics and machine learning in Python.  

