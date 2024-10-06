import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import joblib
# Load the data
data = pd.read_csv('StudentPerformanceFactors.csv')

# 1. Understanding the Data Structure
print("Data Info:")
print(data.info())

# Explanation: This shows information about the data. It tells you how many rows and columns are present,
# which columns contain strings (categorical data), and which are numbers (numerical data).

# 2. Separating Features (X) and Target (Y)
X = data.drop('Exam_Score', axis=1)  # Features
Y = data['Exam_Score']               # Target (what we want to predict)

# Explanation: X contains all columns except 'Exam_Score' (the score students got). Y contains the 'Exam_Score' which we want to predict.

# 3. Detecting Categorical (string) and Numerical Columns Automatically
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(exclude=['object']).columns.tolist()

# Explanation: Here we automatically separate columns that contain text (like 'Gender', 'Parental_Involvement') from columns that contain numbers (like 'Hours_Studied', 'Attendance').

# 4. Preprocessing: OneHotEncoder for categorical columns, StandardScaler for numerical columns
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numerical_columns),
                   ('cat', OneHotEncoder(), categorical_columns) ])

# Explanation: This step sets up the preprocessing.
# - 'StandardScaler()' is applied to numerical columns to normalize them (mean=0, standard deviation=1).
# - 'OneHotEncoder()' converts text into numbers so that the model can use it.

# 5. Building the Pipeline: Preprocessing followed by Linear Regression
model_pipeline = make_pipeline(preprocessor, LinearRegression())

# Explanation: The pipeline first applies the transformations (scaling and encoding) and then fits a Linear Regression model.
# This keeps everything organized and consistent.

# 6. Splitting Data: Train on 80%, Test on 20%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Explanation: We split the data into a training set (80%) and a test set (20%).
# - Training set: Used to teach the model.
# - Test set: Used to see how well the model performs on unseen data.

# 7. Fitting the Model
model_pipeline.fit(x_train, y_train)

# Explanation: The model is trained (fitted) on the training data. It learns patterns from the data.

# 8. Evaluating the Model (Checking Accuracy)
accuracy = model_pipeline.score(x_test, y_test)
print(f"The accuracy of this model is: {accuracy}")

# Explanation: We check how well the model performs by using the test data (the data the model hasn’t seen before).
# 'score' tells us the accuracy (how close the predictions are to the actual results).
joblib.dump(model_pipeline, 'exam_score_predictor.pkl')
'''yo joblib vanne le data euta file ma save garxa jasle ahmi arkho file ma kaam garna sakhau'''