# Crop_Export_Forecasting_Machine_Learning_Model
A machine learning project using a multilayer perceptron (MLP) to forecast crop export values with FAOSTAT agricultural data. This project applies data science techniques like data preprocessing, feature engineering, and model evaluation with metrics (RMSE, R²), providing insights into the predictive power of MLPs in agricultural forecasting
# Crop Export Forecasting with Machine Learning Model

This project applies a machine learning approach using a **Multilayer Perceptron (MLP)** model to forecast the future export values of crop products. Leveraging historical agricultural and economic data from FAOSTAT, the model predicts export values three years ahead, aiding in data-driven decision-making for agricultural planning and trade analysis.

## Project Overview
The goal of this project is to accurately predict the export values of crop products for a future target year using historical data. The dataset includes multiple indicators relevant to agriculture and economics, such as crop production, emissions, consumer prices, and exchange rates. Using these features, the MLP model provides insight into the factors that may influence export values.

## Machine Learning Approach
This project utilizes a **Multilayer Perceptron (MLP)**, a type of artificial neural network widely used for regression and classification tasks. Here, the MLP model is configured to handle a regression task, predicting continuous values based on the input features. 

### Key Concepts and Techniques
1. **Data Preprocessing**: Preparing and transforming raw data to improve model performance. Steps include handling missing values, feature scaling, encoding categorical variables, and feature selection.
2. **Feature Engineering**: Selecting and transforming relevant indicators from the dataset, such as crop production levels, economic indicators, and price indices, to create meaningful input features for the model.
3. **Model Training**: Training the MLP model with optimized hyperparameters, ensuring that the model generalizes well to new data.
4. **Evaluation Metrics**: Using Root Mean Squared Error (RMSE) and R-squared (R²) to assess the model’s prediction accuracy and interpretability.

## Dataset
The dataset includes several CSV files from FAOSTAT, providing a range of agricultural and economic indicators. These indicators are selected based on their potential influence on crop export values. Key datasets include:
- **Consumer Prices**: Reflects food price inflation and consumer price index.
- **Crops Production Indicators**: Annual yield data for various crops.
- **Emissions**: Agricultural emissions data by type (e.g., CO₂, CH₄).
- **Employment**: Employment in agriculture, forestry, and fishing sectors.
- **Exchange Rate**: Local currency units per USD for various countries.

## Project Structure
- **`data/`**: Contains FAOSTAT data files in CSV format.
- **`docs/`**: Holds project report and code documents, including:
  - `Report_Muharrem_Altunbag_MLP_Coursework_2024-1.pdf`
  - `Code_Muharrem_Altunbag_MLP_Coursework_2024-1.pdf`
- **`notebook.ipynb`**: Main Jupyter Notebook containing code and analysis.
- **`predictions/`**: Contains the model’s prediction outputs as a CSV file.

## MLP Model Overview
The MLP model used in this project is a deep learning neural network structured for regression analysis. Key components of the model include:
- **Architecture**: Two hidden layers with 60 and 40 neurons, respectively, each using ReLU activation functions. A linear activation function is used for the output layer.
- **Loss Function**: Mean Squared Error (MSE) is used, a common choice for regression tasks, to minimize the difference between actual and predicted values.
- **Optimizer**: Adam optimizer with a learning rate of 0.001, chosen for efficient training.
- **Regularization**: Dropout layers and early stopping are used to prevent overfitting and enhance model generalization.

## Results and Performance
The model achieved the following performance metrics:
- **Root Mean Squared Error (RMSE)**: 3.3567 x 10^16
- **R-squared (R²)**: -0.00015, indicating high deviation from actual values.  

Detailed findings and interpretations are available in `docs/Report_Muharrem_Altunbag_MLP_Coursework_2024-1.pdf`.

## MLP_Notebook: Jupyter Notebook Walkthrough
The Jupyter Notebook (`notebook.ipynb`) is organized into the following sections, guiding users through the entire workflow:

### 1. **Data Loading and Exploration**
   - Imports and previews the dataset files.
   - Conducts initial exploratory data analysis (EDA) to understand variable distributions, correlations, and missing values.
   
### 2. **Data Preprocessing**
   - **Handling Missing Values**: Imputes or removes missing values to ensure data integrity.
   - **Feature Scaling**: Scales numerical features for improved MLP model performance.
   - **Encoding Categorical Data**: Applies one-hot encoding or label encoding to categorical features.

### 3. **Feature Engineering**
   - Selects relevant features based on domain knowledge and statistical insights.
   - Constructs new features or aggregates data where necessary to enhance model performance.

### 4. **MLP Model Building**
   - **Model Architecture**: Sets up the MLP model structure using TensorFlow or PyTorch.
   - **Training Configuration**: Configures the training process with early stopping and dropout layers to improve model generalization.

### 5. **Model Training**
   - Splits the data into training and testing sets.
   - Trains the model on the training data, adjusting weights to minimize the loss function.

### 6. **Model Evaluation**
   - Calculates RMSE and R² on the test set to assess prediction accuracy.
   - Compares predicted values with actual values, visualizing the model’s performance.

### 7. **Exporting Predictions**
   - Saves the model’s predictions for the test data to `predictions/MLP_Predictions_export_value.csv`, with columns for data instance ID, true values, and predicted values.

## Instructions for Running the Notebook
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Crop_Export_Forecasting_Machine_Learning_Model.git
   cd Crop_Export_Forecasting_Machine_Learning_Model

