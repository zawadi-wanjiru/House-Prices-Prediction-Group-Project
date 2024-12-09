# Predicting House Prices Using Regression Analysis Group Project

![Homepage Profile](https://github.com/zawadi-wanjiru/House-Prices-Prediction-Group-Project/blob/main/Background%20Image.jpg)

## Project Overview
Find the dataset [here](https://github.com/zawadi-wanjiru/House-Prices-Prediction-Group-Project/blob/main/house_prices_dataset.csv)

This project is aimed at predicting house prices based on various features such as size, location, and number of rooms using regression analysis. We employed Python and libraries like pandas, scikit-learn, matplotlib, and seaborn for data analysis and visualization. The project emphasizes the application of data science concepts, including data cleaning, feature engineering, and model evaluation.

## Key Features

*Data Cleaning: Handled missing values and outliers to ensure dataset quality.
*Feature Engineering: Created new predictive features such as price per square foot and conducted feature selection.
*Model Building: Trained and evaluated a linear regression model using metrics like MAE, MSE, and R-squared.
*Visualization: Presented insights through detailed charts and plots.

## Group Members

1. Nixon Masanya
2. Glen Otieno
3. Gachunga Gift

## Setup and Installation
### Prerequisites
Ensure you have the following installed:
* Python 3.8 or higher
* Jupyter Notebook or Google Colab
* Git

##Installation
1. Clone the repository:

</pre><code class = bash>
git clone https://github.com/your-username/house-prices-regression.git  
cd house-prices-regression  
</code></pre>

2. Install dependencies:

</pre><code class = bash>
pip install -r requirements.txt  
</code></pre>

3. Open the Jupyter Notebook:
</pre><code class = bash>
jupyter notebook
</code></pre>

**##Functionality and Features**

**1. Data Exploration and Cleaning** 

* Performed exploratory data analysis to understand the dataset.
* Identified and addressed missing values and outliers.

# Checking for missing values  

</pre><class code = python>
missing_values = data.isnull().sum()  
print(missing_values)
</code></pre>

# Filling missing values  
</pre><code class = python>
data['feature'].fillna(data['feature'].mean(), inplace=True)  
</code></pre>

**Visuals**
* Distribution of House Prices: Place a histogram showing the price distribution of houses here.
![](https://github.com/zawadi-wanjiru/House-Prices-Prediction-Group-Project/blob/main/Hl.png)

* Correlation Heatmap: Add a heatmap visualization of feature correlations.
![](https://github.com/zawadi-wanjiru/House-Prices-Prediction-Group-Project/blob/main/Hc.png)

**2. Feature Engineering**

* Created additional features like price_per_sqft.
  ![](https://github.com/zawadi-wanjiru/House-Prices-Prediction-Group-Project/blob/main/Hp.png)

* Selected features using correlation analysis and variance inflation factor (VIF).

# Creating a new feature  
</pre><code class = python>
data['price_per_sqft'] = data['price'] / data['sqft']  
</code></pre>

# Correlation analysis  
</pre><code class = python>
import seaborn as sns  
sns.heatmap(data.corr(), annot=True)  
</code></pre>

**3. Model Building**

* Built a linear regression model.
* Evaluated the model with metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

</pre><code class = python>
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
</code></pre>

# Training the model  
</pre><code class = python>
model = LinearRegression()  
model.fit(X_train, y_train)
</code></pre>

# Predictions  
</pre><code class = python>
y_pred = model.predict(X_test)  
</code></pre>

# Evaluation  
</pre><code class = python>
print("MAE:", mean_absolute_error(y_test, y_pred))  
print("MSE:", mean_squared_error(y_test, y_pred))  
print("R-squared:", r2_score(y_test, y_pred))  
</code></pre>

**Visuals**

* Prediction vs. Actual Scatter Plot: Add a scatter plot comparing actual and predicted prices.

**4. Deployment** 

* The model is deployed using Flask for a simple web interface.

</pre><code class = python>
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
  </code></pre>
  
**Additional Information**

* Dataset: The dataset is sourced from an open Kaggle dataset for house prices.
* Limitations: Model performance can be improved with additional features or alternative algorithms like Random Forest or Gradient Boosting.

**Visual Summary**
* House Price Distribution: Histogram placed in Data Exploration and Cleaning.
* Feature Correlation Heatmap: Heatmap placed in Feature Engineering.
* Prediction vs. Actual Scatter Plot: Scatter plot placed in Model Building.
Feel free to reach out for any clarifications or additional details! 😄 😃
