from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_pipeline = joblib.load('exam_score_predictor.pkl')

# Define valid categories for the inputs that need to match the training data
VALID_PARENTAL_EDUCATION_LEVEL = ['High School', 'College', 'Graduate']
VALID_ACCESS_TO_RESOURCES = ['Low', 'Medium', 'High']
VALID_EXTRACURRICULAR_ACTIVITIES = ['Yes', 'No']
VALID_INTERNET_ACCESS = ['Yes', 'No']
VALID_MOTIVATION_LEVEL = ['Low', 'Medium', 'High']
VALID_SCHOOL_TYPE = ['Public', 'Private']
VALID_PEER_INFLUENCE = ['Negative', 'Neutral', 'Positive']


# Home route for the web interface
@app.route('/')
def home():
    return render_template('index.html')  # This will display the input form


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture user input from the form
        user_input = {
            'Hours_Studied': float(request.form['Hours_Studied']),
            'Attendance': float(request.form['Attendance']),
            'Parental_Involvement': request.form['Parental_Involvement'].strip(),
            'Access_to_Resources': request.form['Access_to_Resources'].strip(),
            'Extracurricular_Activities': request.form['Extracurricular_Activities'].strip(),
            'Sleep_Hours': float(request.form['Sleep_Hours']),
            'Previous_Scores': float(request.form['Previous_Scores']),
            'Motivation_Level': request.form['Motivation_Level'].strip(),
            'Internet_Access': request.form['Internet_Access'].strip(),
            'Tutoring_Sessions': float(request.form['Tutoring_Sessions']),
            'Family_Income': request.form['Family_Income'].strip(),
            'Teacher_Quality': request.form['Teacher_Quality'].strip(),
            'School_Type': request.form['School_Type'].strip(),
            'Peer_Influence': request.form['Peer_Influence'].strip(),
            'Physical_Activity': float(request.form['Physical_Activity']),
            'Learning_Disabilities': request.form['Learning_Disabilities'].strip(),
            'Parental_Education_Level': request.form['Parental_Education_Level'].strip(),
            'Distance_from_Home': request.form['Distance_from_Home'].strip(),
            'Gender': request.form['Gender'].strip()
        }

        # Validate 'Parental_Education_Level'
        if user_input['Parental_Education_Level'] not in VALID_PARENTAL_EDUCATION_LEVEL:
            raise ValueError(f"Invalid Parental Education Level: {user_input['Parental_Education_Level']}")

        # Validate other categorical inputs similarly (e.g., Access_to_Resources)
        if user_input['Access_to_Resources'] not in VALID_ACCESS_TO_RESOURCES:
            raise ValueError(f"Invalid Access to Resources: {user_input['Access_to_Resources']}")

        if user_input['Extracurricular_Activities'] not in VALID_EXTRACURRICULAR_ACTIVITIES:
            raise ValueError(f"Invalid Extracurricular Activities: {user_input['Extracurricular_Activities']}")

        if user_input['Internet_Access'] not in VALID_INTERNET_ACCESS:
            raise ValueError(f"Invalid Internet Access: {user_input['Internet_Access']}")

        if user_input['Motivation_Level'] not in VALID_MOTIVATION_LEVEL:
            raise ValueError(f"Invalid Motivation Level: {user_input['Motivation_Level']}")

        if user_input['School_Type'] not in VALID_SCHOOL_TYPE:
            raise ValueError(f"Invalid School Type: {user_input['School_Type']}")

        if user_input['Peer_Influence'] not in VALID_PEER_INFLUENCE:
            raise ValueError(f"Invalid Peer Influence: {user_input['Peer_Influence']}")

        # Convert the input into a DataFrame
        input_df = pd.DataFrame([user_input])

        # Make prediction
        prediction = model_pipeline.predict(input_df)[0]

        # Return the result
        return render_template(
            'index.html',
            prediction_text=f'Predicted Exam Score: {prediction:.2f}',
            user_input=user_input
        )

    except Exception as e:
        # Print any error that occurs during processing
        #print(f"Error: {e}")
        return render_template(
            'index.html',
            prediction_text=f"An error occurred: {e}",
            user_input=user_input
        )


if __name__ == "__main__":
    app.run(debug=True)
