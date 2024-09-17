import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of students to generate data for
num_students = 100

# Generate data for each feature
age = np.random.randint(16, 22, num_students)
gender = np.random.choice(['Male', 'Female'], num_students)
attendance_rate = np.random.uniform(60, 100, num_students)  # 60% to 100%
homework_completion_rate = np.random.uniform(50, 100, num_students)  # 50% to 100%
test_scores = np.random.uniform(50, 100, num_students)  # 50 to 100
disciplinary_incidents = np.random.poisson(0.5, num_students)  # Incident count, using Poisson distribution
participation = np.random.choice(['Low', 'Medium', 'High'], num_students)
parental_support = np.random.choice(['Low', 'Medium', 'High'], num_students)
family_income = np.random.randint(20000, 100000, num_students)  # Income level in dollars
extracurriculars = np.random.randint(0, 5, num_students)  # Number of extracurriculars
mental_health_score = np.random.uniform(1, 10, num_students)  # Self-reported mental health score (1 to 10)
health_issues = np.random.choice(['Yes', 'No'], num_students)

# Generate the target variable "at_risk" based on some conditions
at_risk = (attendance_rate < 75) | (test_scores < 60) | (disciplinary_incidents > 3)

# Create a DataFrame to hold the dataset
data = pd.DataFrame({
    'age': age,
    'gender': gender,
    'attendance_rate': attendance_rate,
    'homework_completion_rate': homework_completion_rate,
    'test_scores': test_scores,
    'disciplinary_incidents': disciplinary_incidents,
    'participation': participation,
    'parental_support': parental_support,
    'family_income': family_income,
    'extracurriculars': extracurriculars,
    'mental_health_score': mental_health_score,
    'health_issues': health_issues,
    'at_risk': at_risk.astype(int)  # Convert boolean to 0/1
})

# Save the dataset to a CSV file
data.to_csv('student_risk_data.csv', index=False)

# Display the first few rows of the dataset
print(data.head())
