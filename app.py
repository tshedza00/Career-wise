from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load recommendations from the updated CSV file
data = pd.read_csv('recommendations.csv')

# Normalize the 'Interests' column by converting to lowercase
data['Interests'] = data['Interests'].str.lower()

# Prepare data for machine learning (LabelEncoder for NearestNeighbors model)
label_encoder = LabelEncoder()
data['Interests_encoded'] = label_encoder.fit_transform(data['Interests'])

# Prepare data for deep learning (first 3 letters as input)
data['Interest_prefix'] = data['Interests'].apply(lambda x: x[:3])

# One-hot encode the first 3 letters
# One-hot encode the first 3 letters with handle_unknown='ignore'
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


X = onehot_encoder.fit_transform(data['Interest_prefix'].values.reshape(-1, 1))

# Encode the full interests for deep learning output
y = label_encoder.transform(data['Interests'])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an MLP Classifier (Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Nearest Neighbors model
features = data[['Grade', 'Interests_encoded']]
model_knn = NearestNeighbors(n_neighbors=3)  # Adjust number of neighbors if necessary
model_knn.fit(features)

# DataFrame to store timestamps
interest_log = pd.DataFrame(columns=['Interest', 'Timestamp'])
import pandas as pd
from flask import jsonify, request
@app.route('/interest_trends')
def interest_trends():
    # Load the interest log from the CSV file
    interest_log = pd.read_csv('interest_log.csv')

    # Convert the Timestamp column to datetime
    interest_log['Timestamp'] = pd.to_datetime(interest_log['Timestamp'])

    # Group by interest and count the number of entries
    interest_trend = interest_log.groupby(['Interest']).size().reset_index(name='Count')

    # Sort by the most common interests
    interest_trend = interest_trend.sort_values(by='Count', ascending=False)

    # Render the trends in a table format
    return render_template('interest_trends.html', trends=interest_trend)
@app.route('/suggest')
def suggest():
    interest = request.args.get('interest')
    # Perform logic here based on the interest and return a response
    return f"Suggestions for {interest}"


@app.route('/update_graph', methods=['POST'])
def update_graph():
    # Get the data from the request (example: interests or grades)
    user_input = request.json.get('interest')

    # Here, you can update your data processing based on the user input
    # Simulate some time series data for this example
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    values = pd.Series([i * 1.5 for i in range(10)])  # Example trend data

    # Convert the data to JSON format for use in the graph
    data = {
        'dates': dates.strftime('%Y-%m-%d').tolist(),
        'values': values.tolist()
    }

    return jsonify(data)

@app.route('/')
@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None  # Initialize an error message variable

    if request.method == 'POST':
        user_interest = request.form.get('interest', '').strip().lower()  # Get the user input

        # Check if the interest is in the CSV file
        if user_interest not in data['Interests'].str.lower().values:
            error_message = "The interest you entered is not recognized. Please try again."

    return render_template('index.html', error=error_message)

@app.route('/timeseries')
def timeseries():
    return render_template('timeseries.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    global interest_log  # Make sure to update the global log
    grade = int(request.form['grade'])
    interests = request.form['interests'].lower()  # Convert input to lowercase

    # If the interest entered is less than 3 letters, show an error
    if len(interests) < 3:
        return render_template('recommend.html', error="Please enter at least 3 letters.")

    # Log the timestamp and interest input
    current_time = datetime.now()
    new_log_entry = pd.DataFrame([[interests, current_time]], columns=['Interest', 'Timestamp'])
    interest_log = pd.concat([interest_log, new_log_entry], ignore_index=True)

    # Save the log to a CSV file
    interest_log.to_csv('interest_log.csv', index=False)

    # Get the first 3 letters of the interest
    interest_prefix = interests[:3]

    # One-hot encode the input interest prefix
    interest_encoded_input = onehot_encoder.transform([[interest_prefix]])

    # Use the MLP classifier to predict the most relevant full interest
    predicted_interest_encoded = mlp.predict(interest_encoded_input)
    predicted_interest = label_encoder.inverse_transform(predicted_interest_encoded)[0]

    try:
        if predicted_interest not in data['Interests'].values:
            raise ValueError(f"The predicted interest '{predicted_interest}' is not recognized. Available interests: {', '.join(data['Interests'].unique())}")

        interest_encoded = label_encoder.transform([predicted_interest])
        input_data = [[grade, interest_encoded[0]]]
        distances, indices = model_knn.kneighbors(input_data)

        recommended_courses = data.loc[indices[0], 'Course'].tolist()
        recommended_universities = data.loc[indices[0], 'University'].tolist()
        recommended_careers = data.loc[indices[0], 'Career'].tolist()
        recommended_university_urls = data.loc[indices[0], 'University_URL'].tolist()

        # Perform Time Series Analysis (basic visualization of interest trends)
        interest_log.set_index('Timestamp', inplace=True)
        interest_log['Interest'] = interest_log['Interest'].astype(str)
        interest_trend = interest_log.resample('D').size()  # Group by day

        # Plot the time series data
        plt.figure(figsize=(10, 5))
        interest_trend.plot()
        plt.title('Interest Entry Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Entries')
        plt.grid(True)
        plt.savefig('static/interest_trend.png')
        plt.close()

        return render_template('recommend.html', interests=predicted_interest, 
                               courses=recommended_courses, 
                               universities=recommended_universities, 
                               careers=recommended_careers,
                               university_urls=recommended_university_urls,
                               trend_image='static/interest_trend.png')

    except ValueError as e:
        return render_template('recommend.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
