import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Engagement Analytics",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "European_Bank.csv")

df = pd.read_csv(DATA_PATH)


# ============================================================
# CREATE FEATURES
# ============================================================

balance_threshold = df["Balance"].quantile(0.75)
salary_threshold = df["EstimatedSalary"].quantile(0.75)

df["Product_Type"] = df["NumOfProducts"].apply(
    lambda x: "Single Product"
    if x == 1
    else "Multi Product"
)

df["Relationship_Strength_Index"] = (
    (
        (df["NumOfProducts"] / df["NumOfProducts"].max()) * 0.5
        + df["IsActiveMember"] * 0.3
        + df["HasCrCard"] * 0.2
    )
    * 100
)

df["Relationship_Tier"] = pd.cut(
    df["Relationship_Strength_Index"],
    bins=[-1, 30, 60, 100],
    labels=["Weak", "Medium", "Strong"]
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Customer Engagement & Product Utilization Analytics")

st.write(
    "Bank customer retention analysis based on engagement, "
    "product utilization and financial commitment."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

activity_filter = st.sidebar.selectbox(
    "Customer Activity",
    ["All", "Active", "Inactive"]
)

product_filter = st.sidebar.slider(
    "Number of Products",
    min_value=int(df["NumOfProducts"].min()),
    max_value=int(df["NumOfProducts"].max()),
    value=(
        int(df["NumOfProducts"].min()),
        int(df["NumOfProducts"].max())
    )
)

balance_filter = st.sidebar.slider(
    "Minimum Balance",
    min_value=0.0,
    max_value=float(df["Balance"].max()),
    value=0.0
)

salary_filter = st.sidebar.slider(
    "Minimum Salary",
    min_value=0.0,
    max_value=float(df["EstimatedSalary"].max()),
    value=0.0
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if activity_filter == "Active":
    filtered_df = filtered_df[
        filtered_df["IsActiveMember"] == 1
    ]

elif activity_filter == "Inactive":
    filtered_df = filtered_df[
        filtered_df["IsActiveMember"] == 0
    ]


filtered_df = filtered_df[
    (filtered_df["NumOfProducts"] >= product_filter[0]) &
    (filtered_df["NumOfProducts"] <= product_filter[1])
]

filtered_df = filtered_df[
    filtered_df["Balance"] >= balance_filter
]

filtered_df = filtered_df[
    filtered_df["EstimatedSalary"] >= salary_filter
]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(filtered_df)

churned_customers = filtered_df["Exited"].sum()

churn_rate = (
    filtered_df["Exited"].mean() * 100
    if total_customers > 0
    else 0
)

active_rate = (
    filtered_df["IsActiveMember"].mean() * 100
    if total_customers > 0
    else 0
)

avg_products = (
    filtered_df["NumOfProducts"].mean()
    if total_customers > 0
    else 0
)


# ============================================================
# KPI DISPLAY
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Churned Customers",
    f"{int(churned_customers):,}"
)

col3.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

col4.metric(
    "Avg Products",
    f"{avg_products:.2f}"
)


st.divider()


# ============================================================
# ENGAGEMENT VS CHURN
# ============================================================

st.subheader("📊 Engagement vs Churn")

activity_analysis = (
    filtered_df
    .groupby("IsActiveMember")
    .agg(
        Customers=("CustomerId", "count"),
        Churn_Rate=("Exited", "mean")
    )
    .reset_index()
)

activity_analysis["Activity"] = activity_analysis[
    "IsActiveMember"
].map(
    {
        0: "Inactive",
        1: "Active"
    }
)

activity_analysis["Churn_Rate"] *= 100

fig_activity = px.bar(
    activity_analysis,
    x="Activity",
    y="Churn_Rate",
    text="Churn_Rate",
    title="Churn Rate: Active vs Inactive Customers"
)

fig_activity.update_traces(
    texttemplate="%{text:.2f}%"
)

st.plotly_chart(
    fig_activity,
    use_container_width=True
)


# ============================================================
# PRODUCT UTILIZATION
# ============================================================

st.subheader("📦 Product Utilization Impact")

product_analysis = (
    filtered_df
    .groupby("NumOfProducts")
    .agg(
        Customers=("CustomerId", "count"),
        Churn_Rate=("Exited", "mean")
    )
    .reset_index()
)

product_analysis["Churn_Rate"] *= 100

fig_product = px.bar(
    product_analysis,
    x="NumOfProducts",
    y="Churn_Rate",
    text="Churn_Rate",
    title="Churn Rate by Number of Products"
)

fig_product.update_traces(
    texttemplate="%{text:.2f}%"
)

st.plotly_chart(
    fig_product,
    use_container_width=True
)


# ============================================================
# BALANCE VS ENGAGEMENT
# ============================================================

st.subheader("💰 Balance vs Customer Engagement")

fig_balance = px.scatter(
    filtered_df,
    x="Balance",
    y="EstimatedSalary",
    color="IsActiveMember",
    size="NumOfProducts",
    hover_data=[
        "CustomerId",
        "Age",
        "Exited"
    ],
    title="Balance vs Salary by Activity"
)

st.plotly_chart(
    fig_balance,
    use_container_width=True
)


# ============================================================
# HIGH-VALUE DISENGAGED CUSTOMERS
# ============================================================

st.subheader("⚠️ High-Value Disengaged Customers")

high_value = filtered_df[
    (filtered_df["Balance"] >= balance_threshold) &
    (filtered_df["IsActiveMember"] == 0)
]

st.write(
    f"High-balance threshold: **{balance_threshold:,.2f}**"
)

st.write(
    f"Number of high-value disengaged customers: "
    f"**{len(high_value):,}**"
)

if len(high_value) > 0:

    high_value_display = high_value[
        [
            "CustomerId",
            "Surname",
            "Geography",
            "Age",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Exited"
        ]
    ]

    st.dataframe(
        high_value_display,
        use_container_width=True
    )

else:

    st.info(
        "No high-value disengaged customers found "
        "for the selected filters."
    )


# ============================================================
# RELATIONSHIP STRENGTH
# ============================================================

st.subheader("💪 Relationship Strength")

relationship_analysis = (
    filtered_df
    .groupby("Relationship_Tier", observed=False)
    .agg(
        Customers=("CustomerId", "count"),
        Churn_Rate=("Exited", "mean")
    )
    .reset_index()
)

relationship_analysis["Churn_Rate"] *= 100

fig_relationship = px.bar(
    relationship_analysis,
    x="Relationship_Tier",
    y="Churn_Rate",
    text="Churn_Rate",
    title="Relationship Strength vs Churn"
)

fig_relationship.update_traces(
    texttemplate="%{text:.2f}%"
)

st.plotly_chart(
    fig_relationship,
    use_container_width=True
)


# ============================================================
# CREDIT CARD ANALYSIS
# ============================================================

st.subheader("💳 Credit Card Stickiness")

card_analysis = (
    filtered_df
    .groupby("HasCrCard")
    .agg(
        Customers=("CustomerId", "count"),
        Churn_Rate=("Exited", "mean")
    )
    .reset_index()
)

card_analysis["Card Ownership"] = card_analysis[
    "HasCrCard"
].map(
    {
        0: "No Credit Card",
        1: "Has Credit Card"
    }
)

card_analysis["Churn_Rate"] *= 100

fig_card = px.bar(
    card_analysis,
    x="Card Ownership",
    y="Churn_Rate",
    text="Churn_Rate",
    title="Credit Card Ownership vs Churn"
)

fig_card.update_traces(
    texttemplate="%{text:.2f}%"
)

st.plotly_chart(
    fig_card,
    use_container_width=True
)


# ============================================================
# AT-RISK PREMIUM CUSTOMERS
# ============================================================

st.subheader("🚨 At-Risk Premium Customers")

premium_customers = filtered_df[
    (filtered_df["Balance"] >= balance_threshold) &
    (filtered_df["EstimatedSalary"] >= salary_threshold) &
    (filtered_df["IsActiveMember"] == 0)
]

st.write(
    f"Premium customer balance threshold: "
    f"**{balance_threshold:,.2f}**"
)

st.write(
    f"Premium customer salary threshold: "
    f"**{salary_threshold:,.2f}**"
)

st.write(
    f"At-risk premium customers: "
    f"**{len(premium_customers):,}**"
)

if len(premium_customers) > 0:

    st.dataframe(
        premium_customers[
            [
                "CustomerId",
                "Surname",
                "Geography",
                "Age",
                "Balance",
                "EstimatedSalary",
                "NumOfProducts",
                "Exited"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Engagement & Product Utilization Analytics | "
    "Unified Mentor Internship Project"
)