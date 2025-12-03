import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# -----------------------------
# 2. Clean missing values + duplicates
# -----------------------------
df.drop_duplicates(inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# -----------------------------
# 3. Feature selection
# -----------------------------
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# -----------------------------
# 4. Preprocessing pipeline
# -----------------------------
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Embarked']

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# -----------------------------
# 5. Full ML Pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', LogisticRegression(max_iter=300))
])

# -----------------------------
# 6. Train & evaluate the model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))

# -----------------------------
# 7. Save model using joblib
# -----------------------------
joblib.dump(pipeline, "titanic_model.joblib")

print("Model saved as titanic_model.joblib")

