import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("../data/European_Bank.csv")

print("=" * 60)
print("CUSTOMER ENGAGEMENT & PRODUCT UTILIZATION ANALYTICS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. BASIC DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 3. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe().T)


# ============================================================
# 4. CHURN ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CHURN ANALYSIS")
print("=" * 60)

churn_count = df["Exited"].value_counts()

print("\nChurn Count:")
print(churn_count)

churn_percentage = df["Exited"].value_counts(normalize=True) * 100

print("\nChurn Percentage:")
print(churn_percentage)

total_customers = len(df)
churned_customers = df["Exited"].sum()
retained_customers = total_customers - churned_customers

print("\nTotal Customers:", total_customers)
print("Churned Customers:", churned_customers)
print("Retained Customers:", retained_customers)


# ============================================================
# 5. ENGAGEMENT ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ENGAGEMENT ANALYSIS")
print("=" * 60)

engagement_churn = df.groupby("IsActiveMember")["Exited"].mean() * 100

print("\nChurn Rate by Activity:")
print(engagement_churn)

activity_table = pd.crosstab(
    df["IsActiveMember"],
    df["Exited"],
    normalize="index"
) * 100

print("\nActivity vs Churn Percentage:")
print(activity_table)


# ============================================================
# 6. PRODUCT UTILIZATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("PRODUCT UTILIZATION ANALYSIS")
print("=" * 60)

product_churn = df.groupby("NumOfProducts")["Exited"].mean() * 100

print("\nChurn Rate by Number of Products:")
print(product_churn)

product_count = df["NumOfProducts"].value_counts().sort_index()

print("\nCustomer Count by Number of Products:")
print(product_count)


# Single vs Multi Product

df["Product_Type"] = np.where(
    df["NumOfProducts"] == 1,
    "Single Product",
    "Multi Product"
)

single_multi_churn = df.groupby("Product_Type")["Exited"].mean() * 100

print("\nSingle vs Multi Product Churn:")
print(single_multi_churn)


# ============================================================
# 7. BALANCE VS ENGAGEMENT
# ============================================================

print("\n" + "=" * 60)
print("BALANCE VS ENGAGEMENT")
print("=" * 60)

balance_activity = df.groupby("IsActiveMember").agg(
    Average_Balance=("Balance", "mean"),
    Customer_Count=("CustomerId", "count"),
    Churn_Rate=("Exited", "mean")
)

balance_activity["Churn_Rate"] = balance_activity["Churn_Rate"] * 100

print(balance_activity)


# ============================================================
# 8. HIGH-BALANCE DISENGAGED CUSTOMERS
# ============================================================

balance_threshold = df["Balance"].quantile(0.75)

df["High_Balance"] = df["Balance"] >= balance_threshold

high_balance_disengaged = df[
    (df["High_Balance"] == True) &
    (df["IsActiveMember"] == 0)
]

high_balance_disengaged_churn = (
    high_balance_disengaged["Exited"].mean() * 100
)

print("\n" + "=" * 60)
print("HIGH-BALANCE DISENGAGED CUSTOMERS")
print("=" * 60)

print("\n75th Percentile Balance:", round(balance_threshold, 2))

print(
    "High-Balance Disengaged Customers:",
    len(high_balance_disengaged)
)

print(
    "High-Balance Disengaged Churn Rate:",
    round(high_balance_disengaged_churn, 2),
    "%"
)


# ============================================================
# 9. AT-RISK PREMIUM CUSTOMERS
# ============================================================

salary_threshold = df["EstimatedSalary"].quantile(0.75)

at_risk_premium = df[
    (df["Balance"] >= balance_threshold) &
    (df["EstimatedSalary"] >= salary_threshold) &
    (df["IsActiveMember"] == 0)
]

print("\n" + "=" * 60)
print("AT-RISK PREMIUM CUSTOMERS")
print("=" * 60)

print("\nSalary Threshold:", round(salary_threshold, 2))

print(
    "At-Risk Premium Customers:",
    len(at_risk_premium)
)

print(
    "At-Risk Premium Churn Rate:",
    round(at_risk_premium["Exited"].mean() * 100, 2),
    "%"
)


# ============================================================
# 10. CREDIT CARD STICKINESS
# ============================================================

print("\n" + "=" * 60)
print("CREDIT CARD STICKINESS")
print("=" * 60)

credit_card_churn = df.groupby("HasCrCard")["Exited"].mean() * 100

print("\nChurn Rate by Credit Card Ownership:")
print(credit_card_churn)

card_retention = df.groupby("HasCrCard")["Exited"].apply(
    lambda x: (1 - x.mean()) * 100
)

print("\nRetention Rate by Credit Card Ownership:")
print(card_retention)


# ============================================================
# 11. RELATIONSHIP STRENGTH INDEX
# ============================================================

# Product score: more products = higher score
product_score = df["NumOfProducts"] / df["NumOfProducts"].max()

# Activity score: active = 1, inactive = 0
activity_score = df["IsActiveMember"]

# Credit card score
card_score = df["HasCrCard"]

# Combined relationship score
df["Relationship_Strength_Index"] = (
    product_score * 0.5 +
    activity_score * 0.3 +
    card_score * 0.2
) * 100

print("\n" + "=" * 60)
print("RELATIONSHIP STRENGTH")
print("=" * 60)

print(
    "\nAverage Relationship Strength:",
    round(df["Relationship_Strength_Index"].mean(), 2)
)


# Relationship Strength Categories

df["Relationship_Tier"] = pd.cut(
    df["Relationship_Strength_Index"],
    bins=[-1, 30, 60, 100],
    labels=["Weak", "Medium", "Strong"]
)

relationship_churn = df.groupby(
    "Relationship_Tier",
    observed=False
)["Exited"].mean() * 100

print("\nChurn Rate by Relationship Tier:")
print(relationship_churn)


# ============================================================
# 12. ENGAGEMENT PROFILES
# ============================================================

def classify_customer(row):

    if row["IsActiveMember"] == 1 and row["NumOfProducts"] >= 2:
        return "Active Engaged"

    elif row["IsActiveMember"] == 0 and row["NumOfProducts"] == 1:
        return "Inactive Disengaged"

    elif row["IsActiveMember"] == 1 and row["NumOfProducts"] == 1:
        return "Active Low-Product"

    elif row["IsActiveMember"] == 0 and row["Balance"] >= balance_threshold:
        return "Inactive High-Balance"

    else:
        return "Other"


df["Engagement_Profile"] = df.apply(
    classify_customer,
    axis=1
)

profile_analysis = df.groupby(
    "Engagement_Profile",
    observed=False
).agg(
    Customers=("CustomerId", "count"),
    Churn_Rate=("Exited", "mean"),
    Average_Balance=("Balance", "mean"),
    Average_Products=("NumOfProducts", "mean")
)

profile_analysis["Churn_Rate"] = (
    profile_analysis["Churn_Rate"] * 100
)

print("\n" + "=" * 60)
print("ENGAGEMENT PROFILES")
print("=" * 60)

print(profile_analysis)


# ============================================================
# 13. SAVE ANALYSIS DATA
# ============================================================

os.makedirs("../reports", exist_ok=True)

df.to_csv(
    "../reports/customer_engagement_analysis.csv",
    index=False
)

profile_analysis.to_csv(
    "../reports/engagement_profile_analysis.csv"
)

print("\nAnalysis files saved successfully.")


# ============================================================
# 14. VISUALIZATIONS
# ============================================================

os.makedirs("../images", exist_ok=True)


# Churn Distribution

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Exited"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = Retained, 1 = Churned)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "../images/churn_distribution.png"
)

plt.close()


# Activity vs Churn

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="IsActiveMember",
    y="Exited"
)

plt.title("Activity vs Churn")
plt.xlabel("Active Member (0 = No, 1 = Yes)")
plt.ylabel("Churn Rate")

plt.tight_layout()

plt.savefig(
    "../images/activity_vs_churn.png"
)

plt.close()


# Product vs Churn

plt.figure(figsize=(7, 5))

product_churn.plot(
    kind="bar"
)

plt.title("Product Count vs Churn Rate")
plt.xlabel("Number of Products")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "../images/product_vs_churn.png"
)

plt.close()


# Balance Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Balance",
    bins=30
)

plt.title("Customer Balance Distribution")
plt.xlabel("Balance")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "../images/balance_distribution.png"
)

plt.close()


# Relationship Tier vs Churn

plt.figure(figsize=(7, 5))

relationship_churn.plot(
    kind="bar"
)

plt.title("Relationship Strength vs Churn")
plt.xlabel("Relationship Tier")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "../images/relationship_strength_vs_churn.png"
)

plt.close()


print("\n" + "=" * 60)
print("EDA AND KPI ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)