import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("customer_churn.csv")

print("Loaded dataset shape:", data.shape)
data.info()

print("\nMissing values per column:")
print(data.isna().sum())

print("\nDuplicate rows:", data.duplicated().sum())
print("Repeated customer IDs:", data["Customer_ID"].duplicated().sum())

invalid_rows = data[
    ~data["Age"].between(18, 75)
    | ~data["Tenure"].between(1, 72)
    | ~data["Monthly_Charges"].between(20, 120)
    | ~data["Internet_Usage"].between(5, 500)
    | (data["Support_Calls"] < 0)
    | (data["Previous_Complaints"] < 0)
    | ~data["Contract_Type"].isin(["Monthly", "One year", "Two years"])
    | ~data["Payment_Method"].isin(["Card", "Bank transfer", "Cash"])
    | ~data["Churn"].isin([0, 1])
    | ~np.isclose(data["Total_Charges"], data["Tenure"] * data["Monthly_Charges"])
]

print("Invalid records:", len(invalid_rows))
if not invalid_rows.empty:
    print(invalid_rows.head().to_string(index=False))

counts = data["Churn"].value_counts().reindex([0, 1], fill_value=0)

fig, ax = plt.subplots()
bars = ax.bar(["Stayed", "Left"], counts, color=["steelblue", "darkorange"])

for bar, count in zip(bars, counts):
    percentage = count / len(data) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count,
        f"{count:,} ({percentage:.2f}%)",
        ha="center",
        va="bottom"
    )

ax.set_title("Customer Churn Distribution")
ax.set_ylabel("Number of customers")
ax.set_ylim(0, counts.max() * 1.15)
fig.tight_layout()
plt.show()


contract_summary = data.groupby("Contract_Type")["Churn"].agg(
    Customers="count",
    Left="sum"
)

contract_summary["Churn Rate (%)"] = (
    100 * contract_summary["Left"] / contract_summary["Customers"]
)
contract_summary = contract_summary.sort_values("Churn Rate (%)", ascending=False)

print("\nChurn by contract type:")
print(contract_summary.round(2))

fig, ax = plt.subplots()
bars = ax.bar(contract_summary.index, contract_summary["Churn Rate (%)"])
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_title("Churn Rate by Contract Type")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Customers who left (%)")
ax.margins(y=0.15)
fig.tight_layout()
plt.show()


churn_group = data["Churn"].map({0: "Stayed", 1: "Left"})
charge_summary = data.groupby(churn_group)["Monthly_Charges"].agg(
    Customers="count",
    Median="median",
    Mean="mean"
)

print("\nMonthly charges by churn:")
print(charge_summary.loc[["Stayed", "Left"]].round(2))

fig, ax = plt.subplots()
sns.boxplot(
    x=churn_group,
    y=data["Monthly_Charges"],
    order=["Stayed", "Left"],
    ax=ax
)
ax.set_title("Monthly Charges by Churn")
ax.set_xlabel("Customer outcome")
ax.set_ylabel("Monthly charges")
fig.tight_layout()
plt.show()


tenure_summary = data.groupby(churn_group)["Tenure"].agg(
    Customers="count",
    Median_Months="median",
    Mean_Months="mean"
)

print("\nTenure by churn:")
print(tenure_summary.loc[["Stayed", "Left"]].round(2))

fig, ax = plt.subplots()
sns.boxplot(
    x=churn_group,
    y=data["Tenure"],
    order=["Stayed", "Left"],
    ax=ax
)
ax.set_title("Customer Tenure by Churn")
ax.set_xlabel("Customer outcome")
ax.set_ylabel("Tenure (months)")
fig.tight_layout()
plt.show()


call_group = pd.cut(
    data["Support_Calls"],
    bins=[-1, 0, 1, 2, 3, 4, np.inf],
    labels=["0", "1", "2", "3", "4", "5+"]
)

call_summary = data.groupby(call_group, observed=True)["Churn"].agg(
    Customers="count",
    Left="sum"
)
call_summary["Churn Rate (%)"] = 100 * call_summary["Left"] / call_summary["Customers"]

print("\nChurn by support calls:")
print(call_summary.round(2))

fig, ax = plt.subplots()
bars = ax.bar(call_summary.index.astype(str), call_summary["Churn Rate (%)"])
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_title("Churn Rate by Number of Support Calls")
ax.set_xlabel("Number of support calls")
ax.set_ylabel("Customers who left (%)")
ax.margins(y=0.15)
fig.tight_layout()
plt.show()


X = data.drop(columns=["Customer_ID", "Churn"])
y = data["Churn"]

print("Features:", X.columns.tolist())
print("Target:", y.name)
print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

class_counts = pd.DataFrame({
    "Train": y_train.value_counts(),
    "Test": y_test.value_counts()
}).sort_index()

print(class_counts)


category_columns = ["Contract_Type", "Payment_Method"]
numeric_columns = X_train.select_dtypes(include="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_columns),
    ("category", OneHotEncoder(handle_unknown="ignore"), category_columns)
])

model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

print("Model training completed")
print("Classes:", model.classes_)

predictions = model.predict(X_test)
churn_probability = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "Customer_ID": data.loc[X_test.index, "Customer_ID"],
    "Actual": y_test,
    "Predicted": predictions,
    "Churn Probability (%)": churn_probability * 100
})

print(results.head(10).round(1).to_string(index=False))
print("Total test predictions:", len(results))


new_customers = pd.DataFrame({
    "Age":                 [24, 46, 32, 58, 29, 41, 35, 62, 27, 50],
    "Tenure":              [3, 48, 12, 65, 6, 28, 18, 70, 9, 36],
    "Monthly_Charges":     [105, 55, 85, 45, 110, 70, 90, 50, 95, 65],
    "Support_Calls":       [5, 1, 3, 0, 6, 2, 4, 1, 2, 3],
    "Contract_Type":       ["Monthly", "One year", "Monthly", "Two years", "Monthly",
                            "One year", "Monthly", "Two years", "Monthly", "One year"],
    "Payment_Method":      ["Card", "Cash", "Bank transfer", "Card", "Cash",
                            "Bank transfer", "Card", "Cash", "Card", "Bank transfer"],
    "Internet_Usage":      [120, 240, 180, 95, 310, 205, 260, 130, 155, 220],
    "Previous_Complaints": [2, 0, 1, 0, 3, 1, 2, 0, 0, 1]
})

new_customers["Total_Charges"] = (
    new_customers["Tenure"] * new_customers["Monthly_Charges"]
)
new_inputs = new_customers[X_train.columns]

new_results = new_customers.copy()
new_results["Customer_ID"] = [f"NEW{i:02d}" for i in range(1, 11)]
new_results["Predicted"] = model.predict(new_inputs)
new_results["Churn Probability (%)"] = model.predict_proba(new_inputs)[:, 1] * 100

print("\nNew customer predictions:")
print(new_results[["Customer_ID", "Predicted", "Churn Probability (%)"]]
      .round(1).to_string(index=False))



matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()

print("\nTest evaluation")
print("Total test records:", len(y_test))
print("Correct predictions:", tn + tp)
print("Incorrect predictions:", fp + fn)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

print("\nConfusion matrix")
print(pd.DataFrame(
    matrix,
    index=["Actual Stayed", "Actual Left"],
    columns=["Predicted Stayed", "Predicted Left"]
))

print("\nTrue negatives  (stayed, predicted stayed):", tn)
print("False positives (stayed, predicted left):", fp)
print("False negatives (left, predicted stayed):", fn)
print("True positives  (left, predicted left):", tp)


review = results.join(X_test)
review["Incorrect"] = review["Actual"] != review["Predicted"]

errors = review.loc[review["Incorrect"]].copy()
errors["Error_Type"] = errors["Actual"].map({
    1: "False negative",
    0: "False positive"
})

columns = [
    "Customer_ID", "Error_Type", "Churn Probability (%)",
    "Contract_Type", "Tenure", "Support_Calls"
]
print("\nFirst 10 incorrect predictions:")
print(errors[columns].head(10).round(1).to_string(index=False))

group_errors = review.groupby("Contract_Type")["Incorrect"].agg(
    Test_Customers="size",
    Errors="sum"
)
group_errors["Error Rate (%)"] = (
    100 * group_errors["Errors"] / group_errors["Test_Customers"]
)

print("\nError rate by contract type:")
print(group_errors.sort_values("Error Rate (%)", ascending=False).round(2))