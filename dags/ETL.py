import pandas as pd
import logging
import joblib
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

log_file = '/Users/pahan/Desktop/Basics-of-Static-Training/reports/titanic_log.txt'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_data():
    logging.info("Extracting data...")

    train_data = pd.read_csv('/Users/pahan/Desktop/Basics-of-Static-Training/data/train.csv')
    test_data = pd.read_csv('/Users/pahan/Desktop/Basics-of-Static-Training/data/test.csv')
    gender_submission_data = pd.read_csv('/Users/pahan/Desktop/Basics-of-Static-Training/data/gender_submission.csv')

    logging.info(f"Train Data:\n{train_data.head()}")
    logging.info(f"Test Data:\n{test_data.head()}")
    logging.info(f"Gender Submission Data:\n{gender_submission_data.head()}")

    return train_data, test_data, gender_submission_data

def transform_data(**context):
    logging.info("Transforming data...")
    
    train_data, test_data, gender_submission_data = context['ti'].xcom_pull(task_ids='extract_data')

    train_data = train_data.dropna(subset=['Age', 'Embarked'])

    train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch']
    
    logging.info(f"Transformed Train Data:\n{train_data.head()}")

    return train_data, test_data, gender_submission_data

def train_model(**context):
    logging.info("Training model...")

    train_data, test_data, gender_submission_data = context['ti'].xcom_pull(task_ids='transform_data')

    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

    train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    X_train = train_data[['Pclass', 'Age', 'Fare', 'Sex', 'Embarked']]
    X_test = test_data[['Pclass', 'Age', 'Fare', 'Sex', 'Embarked']]

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    y_train = train_data['Survived']
    y_test = gender_submission_data['Survived']  # Используем данные из gender_submission_data

    model = LogisticRegression(max_iter=200)  # Определяем модель
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Для ROC-кривой

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    logging.info(f"Model accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Survived', 'Not Survived'], yticklabels=['Survived', 'Not Survived'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    confusion_matrix_filename = '/Users/pahan/Desktop/Basics-of-Static-Training/reports/confusion_matrix.png'
    plt.savefig(confusion_matrix_filename)
    logging.info(f"Confusion matrix saved to {confusion_matrix_filename}")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_curve_filename = '/Users/pahan/Desktop/Basics-of-Static-Training/reports/roc_curve.png'
    plt.savefig(roc_curve_filename)
    logging.info(f"ROC curve saved to {roc_curve_filename}")

    model_filename = '/Users/pahan/Desktop/Basics-of-Static-Training/reports/titanic_model.pkl'
    joblib.dump(model, model_filename)
    logging.info(f"Model saved to {model_filename}")

    return {
        'model_path': model_filename,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix_filename,
        'roc_curve': roc_curve_filename
    }


def load_model():
    model_filename = '/Users/pahan/Desktop/Basics-of-Static-Training/reports/titanic_model.pkl'
    model = joblib.load(model_filename)
    logging.info(f"Model loaded from {model_filename}")
    return model

dag = DAG(
    'titanic_dag',
    description='ETL process for Titanic dataset and model training',
    schedule_interval=None,
    start_date=datetime(2024, 12, 22),
    catchup=False,
)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    provide_context=True,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

extract_task >> transform_task >> train_task
