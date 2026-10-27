!pip install ucimlrepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# loading student dataset
data = fetch_ucirepo(id=320)
x = data.data.features
y = data.data.targets

# combining both into one dataframe
df = pd.concat([x, y], axis=1)
df.head()

# checking shape and data types
print(df.shape)
print(df.columns)
print(df.dtypes)

# checking null values and duplicates
print(df.isnull().sum())
print("duplicates:", df.duplicated().sum())

df = df.drop_duplicates()

# plotting final grade distribution
plt.hist(df['G3'], bins=15)
plt.title('Final Grade Distribution')
plt.xlabel('G3')
plt.ylabel('count')
plt.show()

# studytime vs final grade
sns.barplot(x='studytime', y='G3', data=df)
plt.title('studytime vs G3')
plt.show()

# scatter plot between G2 and G3
plt.scatter(df['G2'], df['G3'])
plt.xlabel('G2 grade')
plt.ylabel('G3 grade')
plt.title('G2 vs G3')
plt.show()

# checking outliers in absences
sns.boxplot(x=df['absences'])
plt.show()

# correlation heatmap for numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 7))
sns.heatmap(num_cols.corr(), cmap='coolwarm')
plt.show()

# average grade by studytime
print(df.groupby('studytime')['G3'].mean())

# removing outliers from absences using IQR method
q1 = df['absences'].quantile(0.25)
q3 = df['absences'].quantile(0.75)
iqr = q3 - q1

low = q1 - 1.5 * iqr
high = q3 + 1.5 * iqr

df = df[(df['absences'] >= low) & (df['absences'] <= high)]
print("shape after outlier removal:", df.shape)

# converting categorical columns to numeric
df_new = pd.get_dummies(df, drop_first=True)

# splitting x and y
X = df_new.drop('G3', axis=1)
Y = df_new['G3']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# min max scaling on age and absences just to check
mm = MinMaxScaler()
sample_scaled = mm.fit_transform(X_train[['age', 'absences']])
print(sample_scaled[:5])

# standard scaling for model
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

# checking accuracy metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 score:", r2_score(y_test, y_pred))

# comparing actual and predicted values
res = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(res.head(10))

# plotting actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs Predicted')
plt.show()
