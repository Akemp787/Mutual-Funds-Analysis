
# George Washington University Bootcampspot - Project 4
Created by Breakout Room 4 Members:
- Andrew Kemp
- Bryan P Johnson
- Reed Erickson
- Tom Regan

Our Project 4 presentation can be found [here](https://www.canva.com/design/DAGCzUtw8jc/kQuNKfmgiiwtoXon-PFA8A/edit?utm_content=DAGCzUtw8jc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

## An overview of the project and its purpose
Our project aims to analyze mutual funds and ETFs using machine learning techniques to uncover insights into fund performance, portfolio composition, and financial indicators. We plan to explore various targets to assess, such as identifying factors influencing fund performance, understanding the impact of portfolio composition on returns, and evaluating the significance of financial indicators in predicting fund behavior. Through our analysis, we aim to provide valuable insights for investors, fund managers, and financial analysts to make informed decisions regarding fund selection and portfolio management.
### Tools and Libraries
We will utilize Python for data cleaning, analysis, and backend development.For database management, we will use Spark/PySpark.

### The Dataset
Originally comprising over 5 million records, the dataset has been cleaned and refined to include 23,783 mutual funds and 2,310 ETFs, featuring attributes such as total net assets, fund family, inception date, portfolio composition (cash, stocks, bonds, and sectors), historical returns (yearly and quarterly), financial ratios (e.g., price/earnings, Treynor and Sharpe ratios, alpha, beta), and ESG scores.

## How we handled the initial data and cleaning
- Removed columns with only 1 unique value. There were 6 object columns with 1 unique value including "Currency", "Exchange Timezones" etc
- Removed columns with too many values. There were 49 columns with more than 14,000 NaNs
- Removed columns with more than 60% NaN
- Consistancy in Data Forms, like converting dates into dateTime
- Removed metrics we can caluculate ourselves Ex: fund return per quarter, annual return 3, 5, 10 years, alpha, beta, mean, r_squared, stdev, ratios, counts, min, max
- Replace NaN values with column means
- Remove columns with irrelevant data Ex: Management bios, Investment strategy, fund_long_name, etc.

### Binning
We had 5 columns of type "object"
- Fund Category: cutoff value > 150 to retain 82% of data
- Fund Family Type: cutoff value > 150 to retain 72% of data (and remove outliers)
- ESG peer group: cutoff value > 100 to remove outliers
- Investment Type: only had 3 unique values, Blend, Growth, Value
- Size Type: only had 3 values, Large, Medium, Small

### Utilizing SQLite for Data Management
* Database Creation: Established an SQLite database named "mutual_funds_data.db" to store Mutual Funds & ETFs data extracted from external sources.
* Data Import: Imported the cleaned and preprocessed dataset into the SQLite database for efficient storage and retrieval.
* Querying and Manipulation: Conducted SQL queries to extract, filter, and manipulate data directly within the database environment.
* Integration with Machine Learning: Integrated SQLite with machine learning pipelines, allowing seamless data retrieval and preprocessing for model training and testing.
- SQLite Database: https://drive.google.com/file/d/1Hom3giG43Gduxa-n9tFezwvsWyURMqFA/view?usp=sharing 

### Building the Model
- Convert Categorical Data
Using get_dummies, we converted data with dtype== “object”
- Split Preprocessed Data
We split our data using the target array of year_to_date_return (X,y) and used random_state of 42
- Split Preprocessed Data Using train_test_split
We chose to 
- Scaling Data
After creating a Standard Scaler instance, we fit the data using X_train. Then, we scaled the data using .transform(X_train) and (X_test)
- Model selection
We chose to use RandomForestRegressor as a model due to its outperforming other models via:
    - ability to mitigate overfitting
    - capture complex non-linear relationship/interactions
    - provide insights into feature importance
    - robustly handle outliers and missing values
    - efficiently scale for large datasets

### The Model: First Iteration
Test Variable: 'year_to_date_return'
Features: Selected key features: 'fund_yield', 'total_net_assets', 'morningstar_overall_rating', 'fund_annual_report_net_expense_ratio', 'fund_sector_technology'.

Model: Trained RandomForestRegressor with default parameters.

#### Evaluating the First Model
Test Variable: 'year_to_date_return'
Features: Selected key features: 'fund_yield', 'total_net_assets', 'morningstar_overall_rating', 'fund_annual_report_net_expense_ratio', 'fund_sector_technology'.

Model: Trained RandomForestRegressor with default parameters.
Results: Achieved MSE of 0.00240 and R2 of 0.634.

Because of the low R-squared achieved by the first testing with RandomForestRegressor, we decided to test using a GridSearchCV model as it is method for optimizing hyperparameters within the RandomForestRegressor model.

#### Building the Model - Grid Search with Cross Validation 
Parameter Grid Definition
* Defined a parameter grid param_grid containing different values for the hyperparameters 'n_estimators', 'max_depth', and 'min_samples_split'
* Define the parameter grid : param_grid = {'n_estimators': [50, 100, 150],  # Number of trees in the forest'max_depth': [None, 10, 20],
*  Maximum depth of the trees'min_samples_split': [2, 5, 10]
 
#### GridSearchCV Initialization: 
Initialized GridSearchCV with the chosen estimator (RandomForestRegressor), the defined parameter grid, 5-fold cross-validation

*Initialize the RandomForestRegressor model: model = RandomForestRegressor(random_state=42)

GridSearchCV systematically traverses the parameter grid, training and evaluating the RandomForestRegressor model with each hyperparameter combination using cross-validation.
* Model Training: Initialize GridSearchCVgrid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)# Fit the grid search to the datagrid_search.fit(X_train, y_train)
* Print the best parameters foundprint("Best parameters:", grid_search.best_params_)

Best Parameters Identification
* GridSearchCV identifies the best combination of hyperparameters and outputs them. 
* Best parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 150}
* Get the best modelbest_model = grid_search.best_estimator_
* Evaluate the best modelpredictions = best_model.predict(X_test)mse = mean_squared_error(y_test, predictions)r2 = r2_score(y_test, predictions)print("Mean Squared Error:", mse)print("R-squared:", r2)
  
Grid Search with Cross Validation
Hyperparameter Tuning: Conducted grid search to optimize RandomForestRegressor parameters.
Optimal Parameters: max_depth: None, min_samples_split: 2, n_estimators: 150.
Results: Improved performance with MSE of 0.00242 and R2 of 0.631.

### Random Forest Regressor
With only a slight improvement using the Grid search with Cross Validation method, we ultimately decided to revert to the original RandomForestRegressor model from Approach 1. However, this time for our model, we utilized all float columns in our dataset as the key features. Roughly 80% of the columns in our dataset were of type Float.  We expected this to improve the accuracy of our model.

### Building the Model - RandomForestRegressor
* Data Preparation
Define FLOAT columns as feature columns (X) and target variable (y) from the dataset.
Filter float columnsfloat_columns = mutualFunds.select_dtypes(include=['float']).columns.tolist()# Create X and yX = mutualFunds[float_columns]y = mutualFunds['year_to_date_return']

* Data Splitting
Preprocessed data to handle missing data
Split the data into training and testing sets.

* Preprocess data
X.fillna(X.mean(), inplace=True)
* Split Data into Training and Testing SetsX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Trained the model using RandomForestRegressor with Random State = 42

* Model Training
Training: 
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
* Model Testing
Evaluated the trained model's performance by making predictions on the test data (X_test) using the predict method and then calculating two metrics: mean squared error (MSE) and R-squared (R2).

* Evaluate the modelpredictions = model.predict(X_test)mse = mean_squared_error(y_test, predictions)r2 = r2_score(y_test, predictions)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

### Approach 3: Float Columns Implementation
Features: Utilized all float columns, imputed missing values with means.
Model: Trained RandomForestRegressor with default parameters.
Results: Exceptional performance with MSE 0.0002 and R2 ~0.967.


### Takeaways
Exceptional Performance: Demonstrated exceptional predictive accuracy with a remarkably low Mean Squared Error (MSE) of  0.0002 and a near-perfect R-squared (R2)  0.967, indicating the model's ability to explain 96.7% of the variance in the target variable, resulting in highly reliable and precise predictions.
Top indicators: week52_low_change_perc, fund_return_1year accounting for over 60% of the predictable capability. 

### Limitations
* Our dataset was so large that we spend the majority of our time cleaning the data to get it to a point where we could begin building a model
* We encountered several RAM problems when trying to run the original 75 million row dataset and joining the data
* If given more time, most impact on predicting fund performance and how these insights can be utilized by investors or fund managers.
* It would be beneficial to include visualizations or tables comparing the performance metrics (MSE, R2) of different models, such as the initial model versus the optimized model, to highlight the improvement achieved through hyperparameter tuning.


## Instructions on how to use and interact with the project
Clone the repository to your local machine by running the following command in terminal or git bash
```bash
git clone https://github.com/TRegan5/Project_4.git
```
The "fsily_etfs.ipynb" file in the repo is everything you will need to interact with our work.


## References for the data source(s)
- We found our initial data set using [Kaggle Link](https://www.kaggle.com/datasets/stefanoleone992/mutual-funds-and-etfs/data?select=MutualFunds.csv) can be found in the "Resources" folder.
- Our [project proposal](https://docs.google.com/document/d/11TZsD3AG_tvAou-sjt2WAz6kq3hwUo6OUEfyyc5nwRM/edit?usp=sharing).
- All split data can be found in the "Resources" folder.
