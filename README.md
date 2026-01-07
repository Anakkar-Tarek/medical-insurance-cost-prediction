# Medical Insurance Cost Prediction

## Overview

This project aims to predict medical insurance costs based on individual-level data such as age, gender, body mass index (BMI), smoking status, and geographical region. The goal is to build a regression model that estimates the **charges** for a given person based on their attributes. The project follows industry best practices, using state-of-the-art machine learning models and techniques for analysis, model selection, and deployment.

## Project Structure

```
medical-insurance-cost-prediction/
├── data/                # Dataset and associated files
│   ├── insurance.csv    # Medical insurance dataset
│   └── README.md        # Dataset description
├── notebooks/           # Jupyter notebooks for EDA and experiments
│   ├── 01_eda.ipynb     # Exploratory Data Analysis
├── src/                 # Source code for model training, prediction, etc.
│   ├── train.py         # Script for training models
│   ├── predict.py       # Script for making predictions
│   └── utils.py         # Helper functions
├── models/              # Saved models and artifacts
│   └── best_model.pkl   # Trained model
├── app.py               # Flask application for predictions
├── Dockerfile           # Docker configuration for containerization
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Dataset

### Description

The dataset used in this project is derived from a real-world medical insurance cost dataset. It includes both numerical and categorical features, such as age, BMI, smoking status, and region, along with the target variable: **charges** (the amount billed by the insurance company).

* **`age`**: Age of the primary beneficiary
* **`sex`**: Gender (male/female)
* **`bmi`**: Body Mass Index (BMI)
* **`children`**: Number of dependents
* **`smoker`**: Smoking status (yes/no)
* **`region`**: U.S. region (northeast, southeast, southwest, northwest)
* **`charges`**: Medical charges (TARGET VARIABLE)

### Files

* `insurance.csv`: Contains the full dataset with 1,338 rows and 7 features.
* `data/README.md`: Explanation of the dataset, its source, and its features.

### Dataset Source

This dataset can be found on [Kaggle](https://www.kaggle.com/).

## Dependencies

The project uses a virtual environment to manage dependencies, ensuring that the required Python libraries are isolated from your global environment. To set up the virtual environment and install the dependencies, follow these steps:

### Setting Up Virtual Environment

1. Clone the repository:

   ```
   git clone https://github.com/YOUR_USERNAME/medical-insurance-cost-prediction.git
   cd medical-insurance-cost-prediction
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. To freeze the environment later, you can run:

   ```
   pip freeze > requirements.txt
   ```

### Required Libraries

* `pandas`: Data manipulation
* `numpy`: Numerical computations
* `scikit-learn`: Machine learning algorithms
* `xgboost`: Gradient boosting algorithm
* `matplotlib`/`seaborn`: Visualization
* `flask`: For serving the model in production
* `jupyter`: For running EDA and notebooks
* `docker`: For containerizing the app

## Exploratory Data Analysis (EDA)

The project contains an initial **Exploratory Data Analysis (EDA)**, which can be found in the Jupyter notebook: `notebooks/01_eda.ipynb`. This notebook covers:

* **Data loading and cleaning**: Understanding the raw dataset and handling any missing data.
* **Statistical analysis**: A statistical overview of the features and target variable.
* **Feature distribution**: Visualizing the distribution of features using histograms and boxplots.
* **Correlation analysis**: Examining relationships between numerical variables.
* **Categorical feature analysis**: Understanding the impact of categorical variables.
* **Outlier detection**: Identifying and handling outliers in the data.

To run the notebook:

```
jupyter notebook notebooks/01_eda.ipynb
```

## Model Training

This project compares several machine learning models to predict medical insurance charges. The models include:

1. **Linear Regression** (Baseline Model)
2. **Ridge Regression** (Regularized Linear Regression)
3. **Random Forest Regressor** (Ensemble Model)
4. **XGBoost Regressor** (Gradient Boosting Model)

### Training the Model

To train the models, use the following command:

```
python src/train.py
```

This script loads the dataset, processes it, trains multiple models, and evaluates their performance based on metrics such as RMSE, MAE, and R².

### Hyperparameter Tuning

* **GridSearchCV** is used to tune the hyperparameters for **Ridge Regression**, **Random Forest Regressor**, and **XGBoost Regressor**.

### Model Evaluation

The models are evaluated using the following metrics:

* **Root Mean Squared Error (RMSE)**: Measures the average magnitude of errors in predictions.
* **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions without considering their direction.
* **R² (Coefficient of Determination)**: Measures the proportion of variance in the target variable explained by the model.

## Making Predictions

Once the model is trained, you can use it to make predictions on new data. The `src/predict.py` script provides functionality for making predictions:

```
python src/predict.py --input data/test_data.csv
```

This will output predictions for the test data in the specified format.

## Dockerization

The project is Dockerized for easy deployment. The `Dockerfile` enables you to run the application in a containerized environment.

### Building and Running Docker

1. **Build the Docker image**:

   ```
   docker build -t insurance-prediction .
   ```

2. **Run the Docker container**:

   ```
   docker run -p 5000:5000 insurance-prediction
   ```

This will run the Flask application inside a container, accessible at `http://localhost:5000`.

## Deployment

The app can be deployed locally or in the cloud (e.g., Heroku, AWS, GCP). The `app.py` script contains the Flask web application that serves the model and provides an endpoint for making predictions.

To deploy the model, use any platform of your choice that supports Python applications.

## Contributing

Contributions are welcome! If you'd like to improve or add to the project, feel free to fork the repository and submit a pull request. Please make sure to follow the code of conduct and write tests for any new features.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

* Kaggle for providing the dataset.
* [ML Zoomcamp](https://github.com/alexeygrigorev/mlbook) for the inspiration and the curriculum.

---
