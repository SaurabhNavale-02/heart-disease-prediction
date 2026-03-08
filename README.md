Project Title:>> Heart Disease Prediction using Machine Learning



>> Overview
This project focuses on predicting the likelihood of heart disease using several machine learning classification models. The goal is to analyze different health parameters and determine whether a person may have a risk of heart disease.
The project includes exploratory data analysis (EDA), model comparison, and a simple web interface where users can enter medical information and get a prediction.



>> Dataset
The dataset used in this project contains medical attributes that are commonly used in heart disease analysis such as:
 Age
 Sex
 Chest pain type
 Resting blood pressure
 Cholesterol level
 Fasting blood sugar
 Resting ECG results
 Maximum heart rate achieved
 Exercise induced angina
 ST depression
 Slope of ST segment
 Number of major vessels
 Thalassemia

These features are used to train machine learning models to classify whether heart disease is present or not.



<< Exploratory Data Analysis >>

Before building the models, exploratory data analysis was performed to understand the dataset better. This included:

 Checking missing values

 Understanding feature distributions

 Analyzing correlations between variables

 Visualizing relationships between important health indicators


These steps helped in preparing the data for model training.



>> Models Used

Several classification algorithms were tested to compare their performance:

 Logistic Regression

 K-Nearest Neighbors (KNN)

 Gaussian Naive Bayes

 Support Vector Machine (SVM)

 Decision Tree



Each model was trained and evaluated using accuracy and F1 score.

>> Model Performance <<



|      Model          | Accuracy | F1 Score |

| ------------------- | -------- | -------- |

| Logistic Regression | 0.8696   | 0.8846   |

| KNN                 | 0.8641   | 0.8815   |

| Naive Bayes         | 0.8478   | 0.8614   |

| SVM                 | 0.8478   | 0.8667   |

| Decision Tree       | 0.788    | 0.8079   |



Based on the results, << Logistic regression >> was selected as the final model for prediction.



<< Model Deployment >>

A simple web interface was built using Streamlit where users can input health parameters and receive a prediction about the risk of heart disease.

The trained model and preprocessing steps are saved and loaded in the application to generate predictions.



>> Tools Used

 Python

 Pandas

 NumPy

 Scikit-learn

 Matplotlib

 Seaborn

 Streamlit



>> Project Structure

heart-disease-prediction

│

├── data

├── notebooks

├── models

├── app.py

├── requirements.txt

└── README.md



<< How to Run the Project >>



1\. Clone the repository

2\. Install the required libraries



pip install -r requirements.txt

3\. Run the Streamlit app

streamlit run app/app.py

This will start the web application in your browser.




