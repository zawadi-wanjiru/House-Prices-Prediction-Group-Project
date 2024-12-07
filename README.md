# Predicting House Prices Using Regression Analysis

## Project Overview

This project is aimed at predicting house prices based on various features such as size, location, and number of rooms using regression analysis. We employed Python and libraries like pandas, scikit-learn, matplotlib, and seaborn for data analysis and visualization. The project emphasizes the application of data science concepts, including data cleaning, feature engineering, and model evaluation.
--
## Key Features

Data Cleaning: Handled missing values and outliers to ensure dataset quality.
Feature Engineering: Created new predictive features such as price per square foot and conducted feature selection.
Model Building: Trained and evaluated a linear regression model using metrics like MAE, MSE, and R-squared.
Visualization: Presented insights through detailed charts and plots.
--
## Group Members

1. Nixon Masanya
2. Glen Otieno
3. Gachunga Gift

## Setup and Installation
### Prerequisites
Ensure you have the following installed:
* Python 3.8 or higher
* iii. Jupyter Notebook or Google Colab
* Git

##Installation
Clone the repository:
bash
git clone https://github.com/your-username/house-prices-regression.git  
cd house-prices-regression  
Install dependencies:

bash
Copy code
pip install -r requirements.txt  
Open the Jupyter Notebook:

bash
Copy code
jupyter notebook  
Functionality and Features
1. Data Exploration and Cleaning
Performed exploratory data analysis to understand the dataset.
Identified and addressed missing values and outliers.
Code Snippet:
python
Copy code
# Checking for missing values  
missing_values = data.isnull().sum()  
print(missing_values)

# Filling missing values  

</pre><code class = python>
data['feature'].fillna(data['feature'].mean(), inplace=True)  
</pre></code>

Visuals
Distribution of House Prices: Place a histogram showing the price distribution of houses here.
Correlation Heatmap: Add a heatmap visualization of feature correlations.
2. Feature Engineering
Created additional features like price_per_sqft.
Selected features using correlation analysis and variance inflation factor (VIF).
Code Snippet:
python
Copy code
# Creating a new feature  
data['price_per_sqft'] = data['price'] / data['sqft']  

# Correlation analysis  
import seaborn as sns  
sns.heatmap(data.corr(), annot=True)  
3. Model Building
Built a linear regression model.
Evaluated the model with metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
Code Snippet:
python
Copy code
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  

# Training the model  
model = LinearRegression()  
model.fit(X_train, y_train)  

# Predictions  
y_pred = model.predict(X_test)  

# Evaluation  
print("MAE:", mean_absolute_error(y_test, y_pred))  
print("MSE:", mean_squared_error(y_test, y_pred))  
print("R-squared:", r2_score(y_test, y_pred))  
Visuals
Prediction vs. Actual Scatter Plot: Add a scatter plot comparing actual and predicted prices.
4. Deployment
The model is deployed using Flask for a simple web interface.
Code Snippet:
python
Copy code
from flask import Flask, request, jsonify  
import joblib  

app = Flask(__name__)  
model = joblib.load('model.pkl')  

@app.route('/predict', methods=['POST'])  
def predict():  
    data = request.json  
    prediction = model.predict([data['features']])  
    return jsonify({'prediction': prediction[0]})  

if __name__ == '__main__':  
    app.run(debug=True)  
Additional Information
Dataset: The dataset is sourced from an open Kaggle dataset for house prices.
Limitations: Model performance can be improved with additional features or alternative algorithms like Random Forest or Gradient Boosting.
Visual Summary
House Price Distribution: Histogram placed in Data Exploration and Cleaning.
Feature Correlation Heatmap: Heatmap placed in Feature Engineering.
Prediction vs. Actual Scatter Plot: Scatter plot placed in Model Building.
Feel free to reach out for any clarifications or additional details!
