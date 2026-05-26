

<h1>Startup Profit Prediction using Machine Learning</h1>

<p>
This project predicts startup profits using 
<b>Multiple Linear Regression</b> with the help of 
Python and Machine Learning libraries.
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
The goal of this project is to predict the profit of startups 
based on different features such as:
</p>

<ul>
<li>R&D Spend</li>
<li>Administration</li>
<li>Marketing Spend</li>
<li>State</li>
</ul>

<p>
The project uses the <span class="highlight">Linear Regression Algorithm</span>
from Scikit-learn.
</p>

<hr>

<h2>📌 Libraries Used</h2>

<div class="code-box">
<code>
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
</code>
</div>

<h3>Explanation</h3>

<ul>
<li><b>NumPy:</b> Used for numerical operations</li>
<li><b>Pandas:</b> Used for data handling and analysis</li>
<li><b>train_test_split:</b> Splits data into training and testing sets</li>
<li><b>LinearRegression:</b> Machine Learning model</li>
<li><b>mean_squared_error:</b> Calculates prediction loss</li>
<li><b>r2_score:</b> Calculates model accuracy</li>
</ul>

<hr>

<h2>📌 Creating Class</h2>

<div class="code-box">
<code>
class StartupPrediction:
</code>
</div>

<p>
A class is created to organize the complete Machine Learning workflow.
</p>

<hr>

<h2>📌 Constructor Method</h2>

<div class="code-box">
<code>
def __init__(self, file_name):

    self.file_name = file_name
    self.df = None
    self.model = LinearRegression()
</code>
</div>

<h3>Explanation</h3>

<ul>
<li>Initializes dataset file path</li>
<li>Creates empty dataframe</li>
<li>Initializes Linear Regression model</li>
</ul>

<hr>

<h2>📌 Loading Dataset</h2>

<div class="code-box">
<code>
def load_data(self):

    try:
        self.df = pd.read_csv(self.file_name)

        print("Dataset loaded successfully")

        print(self.df.head())

    except FileNotFoundError:
        print("File not found")

    except Exception as e:
        print("Error:", e)
</code>
</div>

<h3>Explanation</h3>

<ul>
<li>Reads CSV dataset using Pandas</li>
<li>Displays first 5 rows</li>
<li>Uses Exception Handling for errors</li>
</ul>

<hr>

<h2>📌 Data Preprocessing</h2>

<div class="code-box">
<code>
self.df['State'] = self.df['State'].map({
    'New York': 0,
    'California': 1,
    'Florida': 2
}).astype(int)
</code>
</div>

<h3>Explanation</h3>

<p>
Machine Learning models cannot understand text data directly.
So categorical values are converted into numerical values.
</p>

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse:collapse; color:white; margin-top:20px;">
<tr>
<th>State</th>
<th>Encoded Value</th>
</tr>

<tr>
<td>New York</td>
<td>0</td>
</tr>

<tr>
<td>California</td>
<td>1</td>
</tr>

<tr>
<td>Florida</td>
<td>2</td>
</tr>

</table>

<hr>

<h2>📌 Feature and Target Selection</h2>

<div class="code-box">
<code>
X = self.df.iloc[:, :-1]

y = self.df.iloc[:, -1]
</code>
</div>

<h3>Explanation</h3>

<ul>
<li><b>X:</b> Independent variables (input features)</li>
<li><b>y:</b> Dependent variable (target/profit)</li>
</ul>

<hr>

<h2>📌 Train Test Split</h2>

<div class="code-box">
<code>
return train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)
</code>
</div>

<h3>Explanation</h3>

<ul>
<li>70% data used for training</li>
<li>30% data used for testing</li>
<li>random_state ensures reproducible results</li>
</ul>

<hr>

<h2>📌 Training the Model</h2>

<div class="code-box">
<code>
def train_model(self, X_train, y_train):

    self.model.fit(X_train, y_train)

    print("Model trained successfully")
</code>
</div>

<h3>Explanation</h3>

<p>
The Linear Regression model learns patterns from training data.
</p>

<hr>

<h2>📌 Model Evaluation</h2>

<div class="code-box">
<code>
y_train_pred = self.model.predict(X_train)

y_test_pred = self.model.predict(X_test)
</code>
</div>

<p>
Predictions are generated for both training and testing data.
</p>

<hr>

<h2>📌 Calculating Loss</h2>

<div class="code-box">
<code>
train_loss = np.sqrt(mean_squared_error(
    y_train,
    y_train_pred
))
</code>
</div>

<h3>Explanation</h3>

<p>
Root Mean Squared Error (RMSE) measures prediction error.
Lower RMSE indicates better performance.
</p>

<hr>

<h2>📌 Calculating Accuracy</h2>

<div class="code-box">
<code>
train_accuracy = r2_score(
    y_train,
    y_train_pred
)
</code>
</div>

<h3>Explanation</h3>

<p>
R² Score measures how well the model fits the data.
</p>

<ul>
<li>1 = Perfect prediction</li>
<li>0 = Poor prediction</li>
<li>Higher score = Better model</li>
</ul>

<hr>

<h2>📌 Object Creation</h2>

<div class="code-box">
<code>
obj = StartupPrediction(
    "50_Startups.csv"
)
</code>
</div>

<p>
Creates object for StartupPrediction class.
</p>

<hr>

<h2>📌 Calling Methods</h2>

<div class="code-box">
<code>
obj.load_data()

X_train, X_test, y_train, y_test =
obj.preprocess_data()

obj.train_model(X_train, y_train)

obj.evaluate_model(
    X_train,
    X_test,
    y_train,
    y_test
)
</code>
</div>

<h3>Workflow</h3>

<ul>
<li>Load dataset</li>
<li>Preprocess data</li>
<li>Train model</li>
<li>Evaluate model</li>
</ul>

<hr>

<h2>📌 Machine Learning Workflow</h2>

<ul>
<li>Import Libraries</li>
<li>Load Dataset</li>
<li>Data Preprocessing</li>
<li>Feature Selection</li>
<li>Train Test Split</li>
<li>Train Linear Regression Model</li>
<li>Predict Outputs</li>
<li>Evaluate Performance</li>
</ul>

<hr>

<h2>📌 Technologies Used</h2>

<ul>
<li>Python</li>
<li>NumPy</li>
<li>Pandas</li>
<li>Scikit-learn</li>
<li>Machine Learning</li>
</ul>

<hr>

<h2>🎯 Conclusion</h2>

<p>
This project demonstrates how Multiple Linear Regression can be used
to predict startup profits using historical business data.
</p>

<p>
It also explains:
</p>

<ul>
<li>Object Oriented Programming (OOPs)</li>
<li>Data Preprocessing</li>
<li>Model Training</li>
<li>Model Evaluation</li>
<li>Machine Learning Workflow</li>
</ul>

<div class="footer">
Made with ❤️ using Python and Machine Learning
</div>

</div>

</body>
</html>
