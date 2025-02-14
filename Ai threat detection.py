import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 🔹 Load the dataset (Make sure the file path is correct)
file_path = "D:/task2/UNSW-NB15_1.csv"
df = pd.read_csv(file_path, low_memory=False)

# 🔹 Drop unwanted columns (fixes mixed dtype issue)
df = df.select_dtypes(include=[np.number])  # Keep only numerical columns

# 🔹 Separate features & labels
X = df.drop(columns=['0.18'], errors='ignore')  # Features
y = df['0.18']  # Target variable (attack or normal traffic)

# 🔹 Check class distribution before split
print("🔍 Class Distribution Before Split:\n", y.value_counts())

# 🔹 Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🔹 Check if both classes exist in training data
print("\n✅ Class Distribution in Training Data Before SMOTE:\n", y_train.value_counts())

# 🔹 Apply SMOTE only if both classes exist
if len(y_train.unique()) > 1:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("\n✅ Class Distribution After SMOTE:\n", pd.Series(y_train_resampled).value_counts())
else:
    print("\n⚠️ SMOTE not applied because only one class is present in y_train.")
    X_train_resampled, y_train_resampled = X_train, y_train  # Keep original if SMOTE fails

# 🔹 Train a balanced RandomForest model
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 🔹 Make predictions
y_pred = model.predict(X_test)

# 🔹 Evaluate the model
print("\n🎯 Model Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))









