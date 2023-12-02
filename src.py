from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("ds_salaries.csv")
    target = 'salary_in_usd'
    # Drop rows where target value is missing
    df_clone = df.dropna(subset=[target])

    # Drop 'salary' column
    if ('salary' in df_clone.columns):
        df_clone = df_clone.drop('salary', axis=1)
    # Get features
    features = list(set(df_clone.columns) - set([target]))
    # Split data into train set and test set (note: by test set I mean to use for final evaluation, not validation)
    total_samples = df_clone.shape[0]

    df_clone_train = df_clone[0: int(total_samples * 0.8)]
    train_set = df_clone_train[features + [target]] # Rearrange columns so target column is the last one
    df_clone_test = df_clone[int(total_samples*0.8):]
    test_set = df_clone_test[features + [target]]

    X_train = train_set[features]
    y_train = train_set[target]

    X_test = test_set[features]
    y_test = test_set[target]

    # Get categorical features and numerical features
    cat_features = [col for col in features if X_train[col].dtype == object]
    numeric_features = list(set(features) - set(cat_features))

    numeric_transformer = Pipeline(steps=[
    ('numeric_imputer', SimpleImputer(strategy='median')),
    ('standard_scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numeric_transformer', numeric_transformer, numeric_features),
        ('cat_transformer', cat_transformer, cat_features)
    ])

    ridge_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('ridge', Ridge(alpha=1.81))
    ])

    ridge_model.fit(X_train, y_train)

    # Below ask user for input and pass that input to the model for prediction and display the prediction
    user_input = {}
    for feature in features:
        if feature in cat_features:
            # Display unique options for categorical features
            unique_values = df_clone_train[feature].unique()
            options_str = ", ".join([str(val) for val in unique_values])
            print(f"Possible options for {feature}: {options_str}")
            user_input[feature] = input(f"Enter your {feature} (select from above): ")
        else:
            # Input for numerical features
            user_input[feature] = int(input(f"Enter your {feature} (numerical): "))
    input_df = pd.DataFrame([user_input])

    prediction = ridge_model.predict(input_df)[0]
    print("Your estimated yearly salary is: {:.0f} dollars".format(prediction))

    return 0

if __name__ == "__main__":
    main()