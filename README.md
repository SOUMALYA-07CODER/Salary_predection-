Salary Prediction using Machine Learning
========================================

This project builds a simple Linear Regression model to predict the salary of an employee based on their years of experience. It's intended for HR departments, recruiters, and data science learners who want to understand the basics of supervised learning.

------------------------------------------------------------
Project Structure
------------------------------------------------------------
Salary_Prediction/
├── dataset.csv
├── Salary_Prediction.ipynb
├── README.txt
└── model.pkl (optional)

------------------------------------------------------------
Problem Statement
------------------------------------------------------------
Accurately predicting employee salary based on work experience is important for fair and data-driven decision-making in HR processes. Manual estimation methods are often subjective and inconsistent.

This project uses a supervised learning approach (Linear Regression) to model the relationship between Years of Experience and Salary based on a real dataset of 30 records.

------------------------------------------------------------
Technologies & Libraries Used
------------------------------------------------------------
- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib (for saving the model)

------------------------------------------------------------
Dataset Information
------------------------------------------------------------
- File Name: dataset.csv
- Size: 30 records
- Features:
  - YearsExperience: Number of years the employee has worked
  - Salary: Corresponding annual salary in ₹

Stats:
- Mean Experience: 5.31 years
- Mean Salary: ₹76,003
- Salary Range: ₹37,731 – ₹1,22,391

------------------------------------------------------------
How it Works
------------------------------------------------------------
1. Load Dataset
   df = pd.read_csv("dataset.csv")

2. Visualize Relationship
   sns.scatterplot(x='YearsExperience', y='Salary', data=df)

3. Train the Model
   from sklearn.linear_model import LinearRegression
   model = LinearRegression()
   model.fit(X_train, y_train)

4. Make Predictions
   salary_pred = model.predict([[5]])  # Predict salary for 5 years of experience

5. Save the Model (optional)
   import joblib
   joblib.dump(model, "model.pkl")

------------------------------------------------------------
Results
------------------------------------------------------------
- The Linear Regression model was trained successfully.
- It shows a strong linear correlation between experience and salary.
- Simple and efficient for small-scale predictions.

------------------------------------------------------------
Conclusion
------------------------------------------------------------
This model demonstrates how a basic ML technique can automate salary prediction using a clean dataset. While the dataset is small and contains a single feature, it provides a foundation for more complex models in HR analytics.

------------------------------------------------------------
Future Scope
------------------------------------------------------------
- Add more features like education, location, or role
- Train with larger, real-world datasets
- Deploy as a web app using Streamlit or Flask
- Include multiple regression models for comparison

------------------------------------------------------------
References
------------------------------------------------------------
- scikit-learn Documentation
- pandas Documentation
- Towards Data Science - Linear Regression articles
- Dataset inspired by publicly available examples

------------------------------------------------------------
Acknowledgments
------------------------------------------------------------
This project was created by Soumalya Ganguly as a capstone mini-project to demonstrate machine learning application in salary prediction.
