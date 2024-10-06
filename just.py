import joblib
import pandas as pd


# Load the trained model
model_pipeline = joblib.load('exam_score_predictor.pkl')


def predict_exam_score(user_input):
    # Convert user input to DataFrame (assuming user_input is a dictionary)
    input_df = pd.DataFrame([user_input])

    # Predict using the loaded model
    prediction = model_pipeline.predict(input_df)
    return prediction[0]


# Example user input
user_input = {
    'Hours_Studied': 20,
    'Attendance': 90,
    'Parental_Involvement': 'High',
    'Access_to_Resources': 'Medium',
    'Extracurricular_Activities': 'Yes',
    'Sleep_Hours': 8,
    'Previous_Scores': 85,
    'Motivation_Level': 'High',
    'Internet_Access': 'Yes',
    'Tutoring_Sessions': 2,
    'Family_Income': 'Medium',
    'Teacher_Quality': 'High',
    'School_Type': 'Public',
    'Peer_Influence': 'Positive',
    'Physical_Activity': 4,
    'Learning_Disabilities': 'No',
    'Parental_Education_Level': 'College',
    'Distance_from_Home': 'Near',
    'Gender': 'Male'
}


# Make prediction
predicted_score = predict_exam_score(user_input)
print(f"Predicted Exam Score: {predicted_score:.2f}")
