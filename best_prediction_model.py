import pandas as pd
import numpy as np
import cloudpickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Load dataset
# =====================
df = pd.read_csv("Medical Insurance Cost.csv")
df.head()

# Drop duplicates
print('Duplicate: ',df.duplicated().sum())
df = df.drop_duplicates()

# Display Info
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.nunique())

# =====================
# Outlier
# =====================

# Outliers in BMI
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower bound", lower_bound)
print("Upper bound", upper_bound)

outliers = ((df['bmi'] < lower_bound) | (df['bmi'] > upper_bound))
outliers_df = df[outliers]

print(f"\nNumber of outliers detected: {len(outliers_df)}")
print(outliers_df)

# =====================
# Feature Engineering
# =====================
# Log-transform bmi
df['bmi_log'] = np.log(df['bmi'] + 1)

# BMI per Age
df['bmi_per_age'] = df['bmi'] / df['age']

# Binning Age
df['age_group'] = pd.cut(
    df['age'],
    bins=[18, 25, 45, 65],
    labels=['Young', 'Adult', 'Senior'], right=False
)

# Target and features
X = df.drop('charges', axis=1)
y = df['charges']

# =====================
# Column split
# =====================
numerical_cols = ['age', 'children', 'bmi_per_age', 'bmi_log']
categorical_cols_for_label_encoding = ['sex', 'smoker']
categorical_cols_for_one_hot_encoding = ['region', 'age_group']

# =====================
# Preprocessing
# =====================
# Numerical features: impute missing values and scale
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features for label encoding
cat_label_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', FunctionTransformer(lambda X: np.array([LabelEncoder().fit_transform(col) for col in X.T]).T))
])

# Categorical features for one-hot encoding
cat_one_hot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine all transformers into a preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat_label', cat_label_transformer, categorical_cols_for_label_encoding),
    ('cat_one_hot', cat_one_hot_transformer, categorical_cols_for_one_hot_encoding)
])

# =====================
# Best Model: XGBoost
# =====================
best_model_xgb = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1
)

# Create full pipeline with preprocessing and model
best_model_xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model_xgb)
])

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================
# Train the model
# =====================
best_model_xgb_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = best_model_xgb_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Best Model Scores: ")
print("R2 Score: ", r2)
print("RMSE: ", rmse)
print("MAE: ", mae)

# =====================
# Save Best Model
# =====================
with open('best_model_xgb.pkl', 'wb') as f:
    cloudpickle.dump(best_model_xgb_pipeline, f)

print("✅ Best model saved as best_model_xgb.pkl")

# =====================
# Plot Actual vs Predicted
# =====================
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Insurance Charges")
plt.ylabel("Predicted Insurance Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.grid(True)
plt.show()