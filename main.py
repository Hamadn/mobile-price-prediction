import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

dataset = pd.read_csv("./Mobile Prices Prediction Dataset.csv")

dataset.info()
dataset.head()
dataset.tail()
dataset.describe()

X = dataset[
    [
        "battery_power",
        "clock_speed",
        "dual_sim",
        "four_g",
        "n_cores",
        "px_height",
        "px_width",
        "ram",
        "touch_screen",
        "wifi",
    ]
]
y = dataset["price_range"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

print("Classification Report:\n", classification_report(y_test, y_pred))

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind="bar", figsize=(10, 6))
plt.title("Feature Importance")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("Cross-Validation Accuracy: {:.2f}".format(scores.mean()))
