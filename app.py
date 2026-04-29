import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r"C:\Users\RAMADEVI SURIPAKA\Downloads\PythonProject1\50_Startups.csv")

# Convert State column
df['State'] = df['State'].map({
    'New York': 0,
    'California': 1,
    'Florida': 2
}).astype(int)

# Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    rd = float(request.form['rd'])
    admin = float(request.form['admin'])
    marketing = float(request.form['marketing'])
    state = int(request.form['state'])

    prediction = model.predict([[rd, admin, marketing, state]])

    return render_template(
        "index.html",
        prediction_text=f"Predicted Profit: {prediction[0]:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)