# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

def logistic(data):
    X = data.drop('approved',axis=1)
    y = data['approved']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42)

    model.fit(X_train_scaled, y_train)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    threshold = 0.75
    y_pred    = (y_pred_proba > threshold).astype(int)

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def drawing(data):
    approved_data = data[data['approved'] == 1]
    rejected_data = data[data['approved'] == 0]

    plt.scatter(approved_data['income'], approved_data['price'], color='blue', label='Approved')
    plt.scatter(rejected_data['income'], rejected_data['price'], color='red', label='Rejected')
    plt.xlabel('Income')
    plt.ylabel('Price')
    plt.title('Scatter Plot of Price vs Income')
    plt.legend()
    plt.show()
    
    corr_matrix = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    
    return print('This is a plot of price vs income')

def statistics(data):
    print(data.describe())
    print("\nCorrelation Matrix:")
    print(data.corr())

    print("\nValue Counts for Categorical Columns:")
    print("m_s:\n", data['m_s'].value_counts())
    print("\nsp_prog:\n", data['sp_prog'].value_counts())
    print("\napproved:\n", data['approved'].value_counts())
    
def predict_approval(price, year, age, income, m_s, sp_prog):
    model = LogisticRegression(random_state=42)

    data = pd.read_excel("data.xlsx")

    X = data.drop('approved', axis=1)
    y = data['approved']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model.fit(X_scaled, y)

    input_data = [[price, year, age, income, m_s, sp_prog]]
    input_scaled = scaler.transform(input_data)

    approval_prediction = model.predict(input_scaled)[0]

    if approval_prediction == 1:
        return "Approved"
    else:
        return "Not Approved"
def plot_learning_curve(data):
    X = data.drop('approved', axis=1)
    y = data['approved']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(model, X_scaled, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', label='Cross-Validation Score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='orange')
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    
#def model()


data = pd.read_excel("data.xlsx")
print(1)

drawing(data)
plot_learning_curve(data)
statistics(data)
logistic(data)

price_input = float(input("Enter price: "))
year_input = int(input("Enter year: "))
age_input = int(input("Enter age: "))
income_input = int(input("Enter income: "))
m_s_input = int(input("Enter m_s (0 or 1): "))
sp_prog_input = int(input("Enter sp_prog (0 or 1): "))

result = predict_approval(price_input, year_input, age_input, income_input, m_s_input, sp_prog_input)
print(f"The prediction is: {result}")