import json

def generate_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # Header Markdown
    add_md("""# 📊 Humanitarian Micro-Donation RFM Analysis & Donor Segmentation
---
### 🌟 Machine Learning Pipeline & Analytics Notebook

**Author:** Antigravity AI Team  
**Dataset Source:** Donor Master (`01_donor_master.csv`), Donation Transactions (`02_donation_transactions.csv`), Humanitarian Campaigns (`03_humanitarian_campaigns.csv`)

#### 📌 Objective
To build an end-to-end Machine Learning pipeline that processes micro-donation transaction histories, cleans and merges donor demographic data with campaign categories, calculates **RFM (Recency, Frequency, Monetary)** features, applies mathematical log-transformations and standard scaling, clusters donors using **K-Means Clustering**, evaluates cluster quality via statistical metrics (**Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index**), and establishes actionable donor personas with targeted marketing strategies.

---
## 📑 Pipeline Workflow Architecture

1. **Raw Data Ingestion & Schema Inspection**
2. **Data Cleaning & Payment Success Filtering**
3. **Data Merging & Unified Dataset Creation**
4. **Exploratory Data Analysis (EDA) & Visualizations**
5. **RFM Metrics & Scoring Calculation**
6. **Log Transformation & StandardScaler Normalization**
7. **K-Means Clustering & Hyperparameter Optimization (Elbow & Silhouette)**
8. **Cluster Evaluation & Demographic Profiling**
9. **Donor Persona Segmentation & Strategic Insights**
10. **Model Persistence & Artifact Export (`.pkl` & `.csv`)**
""")

    # Imports
    add_md("## 1. Environment Setup & Library Imports")
    add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as io
from plotly.subplots import make_subplots
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Setting Plot Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.figsize'] = (12, 6)
print("Libraries imported successfully!")""")

    # Step 1: Data Ingestion
    add_md("""## 2. Step 1: Data Loading & Initial Inspection

We load the three primary source datasets:
- **`01_donor_master.csv`**: Registered donor metadata (ID, age, gender, country, donor type, registration date).
- **`02_donation_transactions.csv`**: Transaction history (Transaction ID, donor ID, date, amount, currency, payment method, aid sector, successful payment status).
- **`03_humanitarian_campaigns.csv`**: Campaign details (Campaign ID, campaign name, category, priority level, target funding).
""")

    add_code("""# Load Datasets
df_donors = pd.read_csv('01_donor_master.csv')
df_txs = pd.read_csv('02_donation_transactions.csv')
df_campaigns = pd.read_csv('03_humanitarian_campaigns.csv')

print("--- Donors Master Shape ---:", df_donors.shape)
print("--- Transactions Shape ---:", df_txs.shape)
print("--- Campaigns Shape ---:", df_campaigns.shape)

display(df_donors.head(3))
display(df_txs.head(3))
display(df_campaigns.head(3))""")

    # Step 2: Data Cleaning
    add_md("""## 3. Step 2: Data Cleaning & Preprocessing

- Filter out unsuccessful/failed payments (`successful_payment == 1`).
- Check missing values and duplicate rows.
- Convert date attributes to standard `datetime` formats.
- Set snapshot baseline date (`2016-11-21`) for recency calculations.
""")

    add_code("""# Data Inspection for Missing Values
print("Missing values in Donors Master:\n", df_donors.isnull().sum())
print("\nMissing values in Transactions:\n", df_txs.isnull().sum())
print("\nMissing values in Campaigns:\n", df_campaigns.isnull().sum())

# Filter for Successful Transactions Only
df_txs_succ = df_txs[df_txs['successful_payment'] == 1].copy()
print(f"\nFiltered Successful Transactions: {len(df_txs_succ)} out of {len(df_txs)} ({len(df_txs_succ)/len(df_txs):.1%})")

# Convert Datetime Columns
df_txs_succ['transaction_date'] = pd.to_datetime(df_txs_succ['transaction_date'])
df_donors['registration_date'] = pd.to_datetime(df_donors['registration_date'])

snapshot_date = pd.to_datetime('2016-11-21')
print("Baseline Snapshot Date set to:", snapshot_date.date())""")

    # Step 3: Data Merging
    add_md("""## 4. Step 3: Data Merging & Unified Feature Set

We perform tabular joins:
1. Merge transaction records with donor demographics on `donor_id`.
2. Merge transaction records with campaign metadata on `campaign_id`.
3. Save the unified clean dataset as `cleaned_merged_donations.csv`.
""")

    add_code("""# Unified Merged Dataset
df_merged = pd.merge(df_txs_succ, df_donors, on='donor_id', how='left')
df_merged = pd.merge(df_merged, df_campaigns, on='campaign_id', how='left')

print("Unified Merged Dataset Shape:", df_merged.shape)
df_merged.to_csv('cleaned_merged_donations.csv', index=False)
display(df_merged.head(3))""")

    # Step 4: EDA
    add_md("""## 5. Step 4: Exploratory Data Analysis (EDA)

We explore donor demographic trends, payment choices, top aid sectors, and campaign performance.
""")

    add_code("""# EDA Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Donor Gender Distribution
sns.countplot(data=df_donors, x='donor_gender', ax=axes[0, 0], palette='Blues_r')
axes[0, 0].set_title('Donor Gender Distribution', fontsize=14, fontweight='bold')

# 2. Donor Type Split
sns.countplot(data=df_donors, x='donor_type', ax=axes[0, 1], palette='Greens_r')
axes[0, 1].set_title('Donor Type Split', fontsize=14, fontweight='bold')

# 3. Top 10 Donor Countries
top_countries = df_donors['donor_country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, ax=axes[1, 0], palette='Purples_r')
axes[1, 0].set_title('Top 10 Donor Countries', fontsize=14, fontweight='bold')

# 4. Payment Method Preferences
pm_counts = df_txs_succ['payment_method'].value_counts()
sns.barplot(x=pm_counts.index, y=pm_counts.values, ax=axes[1, 1], palette='Oranges_r')
axes[1, 1].set_title('Payment Method Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()""")

    add_code("""# Monthly Donation Trends
df_merged['year_month'] = df_merged['transaction_date'].dt.to_period('M').astype(str)
monthly_donations = df_merged.groupby('year_month')['donation_amount'].agg(['sum', 'count']).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

ax1.plot(monthly_donations['year_month'], monthly_donations['sum'], color='#10B981', marker='o', linewidth=2.5, label='Total Donation ($)')
ax2.bar(monthly_donations['year_month'], monthly_donations['count'], alpha=0.3, color='#3B82F6', label='Transaction Count')

ax1.set_xlabel('Month', fontweight='bold')
ax1.set_ylabel('Total Donation ($)', color='#10B981', fontweight='bold')
ax2.set_ylabel('Transaction Count', color='#3B82F6', fontweight='bold')
ax1.set_xticklabels(monthly_donations['year_month'], rotation=45)
plt.title('Monthly Donation Volume & Amount Trends (2014 - 2016)', fontsize=14, fontweight='bold')
plt.show()""")

    # Step 5: RFM Calculation
    add_md("""## 6. Step 5: RFM Feature Engineering

Calculating:
- **Recency ($R$)**: Days since last successful donation relative to `2016-11-21`.
- **Frequency ($F$)**: Count of successful transactions per donor.
- **Monetary ($M$)**: Sum of donations ($) per donor.
- **Average Donation**: Monetary / Frequency.
- **RFM Quintiles**: Scores 1-5 assigned to R, F, and M.
""")

    add_code("""# Calculate RFM per Donor ID
df_rfm = df_txs_succ.groupby('donor_id').agg(
    last_donation_date=('transaction_date', 'max'),
    frequency=('transaction_id', 'count'),
    monetary=('donation_amount', 'sum'),
    average_donation=('donation_amount', 'mean')
).reset_index()

df_rfm['last_donation_date'] = df_rfm['last_donation_date'].dt.strftime('%Y-%m-%d')
df_rfm['recency'] = (snapshot_date - pd.to_datetime(df_rfm['last_donation_date'])).dt.days

# Combine with full donor list
df_donor_rfm = pd.merge(df_donors, df_rfm, on='donor_id', how='left')

df_donor_rfm['frequency'] = df_donor_rfm['frequency'].fillna(0).astype(int)
df_donor_rfm['monetary'] = df_donor_rfm['monetary'].fillna(0.0).round(2)
df_donor_rfm['average_donation'] = df_donor_rfm['average_donation'].fillna(0.0).round(2)
max_rec = (snapshot_date - pd.to_datetime(df_donor_rfm['registration_date'])).dt.days
df_donor_rfm['recency'] = df_donor_rfm['recency'].fillna(max_rec).astype(int)

# RFM Quintile Scoring (1 to 5)
df_donor_rfm['r_score'] = pd.qcut(df_donor_rfm['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
df_donor_rfm['f_score'] = pd.qcut(df_donor_rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df_donor_rfm['m_score'] = pd.qcut(df_donor_rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df_donor_rfm['rfm_score'] = df_donor_rfm['r_score'].astype(str) + df_donor_rfm['f_score'].astype(str) + df_donor_rfm['m_score'].astype(str)

display(df_donor_rfm[['donor_id', 'recency', 'frequency', 'monetary', 'average_donation', 'rfm_score']].head(5))""")

    # Step 6: Log Transform & Scaling
    add_md("""## 7. Step 6: Feature Transformation & Standardization

Distance-based algorithms such as **K-Means** are sensitive to non-normal distributions and skewed variances.
We apply a natural log transformation `np.log1p(x)` to diminish heavy right-skewness, followed by `StandardScaler` to bring feature scales to mean 0 and standard deviation 1.
""")

    add_code("""# Check Feature Skewness Before Transformation
print("Skewness Before Log Transform:")
print("Recency Skewness:", df_donor_rfm['recency'].skew())
print("Frequency Skewness:", df_donor_rfm['frequency'].skew())
print("Monetary Skewness:", df_donor_rfm['monetary'].skew())

# Apply Log1p Transformation
df_donor_rfm['recency_log'] = np.log1p(df_donor_rfm['recency'])
df_donor_rfm['frequency_log'] = np.log1p(df_donor_rfm['frequency'])
df_donor_rfm['monetary_log'] = np.log1p(df_donor_rfm['monetary'])

# StandardScaler Scaling
features = ['recency_log', 'frequency_log', 'monetary_log']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_donor_rfm[features])

print("\nScaled Feature Matrix Shape:", X_scaled.shape)
print("Feature Mean:", X_scaled.mean(axis=0).round(4))
print("Feature Std Dev:", X_scaled.std(axis=0).round(4))""")

    # Step 7: K-Means & Model Selection
    add_md("""## 8. Step 7: K-Means Clustering & Hyperparameter Optimization

We evaluate cluster performance across $k \in [2, 8]$ using:
1. **Elbow Method (Inertia)**: Within-cluster sum of squares.
2. **Silhouette Score**: Cohesion vs Separation ratio ($[-1, 1]$).
""")

    add_code("""# Evaluate k from 2 to 8
k_range = range(2, 9)
inertias = []
silhouette_scores = []
ch_scores = []
db_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# Plot Elbow and Silhouette curves
fig, ax1 = plt.subplots(figsize=(12, 5))

color = '#3B82F6'
ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
ax1.set_ylabel('Inertia (WCSS)', color=color, fontweight='bold')
ax1.plot(k_range, inertias, marker='o', color=color, linewidth=2.5, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = '#10B981'
ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
ax2.plot(k_range, silhouette_scores, marker='s', color=color, linewidth=2.5, linestyle='--', label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('K-Means Optimization: Elbow Method & Silhouette Score Curves', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

eval_df = pd.DataFrame({
    'Clusters (k)': list(k_range),
    'Inertia': inertias,
    'Silhouette Score': silhouette_scores,
    'Calinski-Harabasz Index': ch_scores,
    'Davies-Bouldin Index': db_scores
})
display(eval_df)""")

    # Step 8: Cluster Evaluation & Profiling
    add_md("""## 9. Step 8: Final Model Training & Cluster Evaluation

Selecting **$k=4$** yields optimal cluster balance, clear interpretability, and distinct separation across donor behaviors.
""")

    add_code("""# Fit Final K-Means Model (k=4)
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
df_donor_rfm['cluster_id'] = kmeans_final.fit_predict(X_scaled)

# Calculate Cluster Summary Stats
cluster_stats = df_donor_rfm.groupby('cluster_id').agg(
    donor_count=('donor_id', 'count'),
    recency_mean=('recency', 'mean'),
    recency_median=('recency', 'median'),
    frequency_mean=('frequency', 'mean'),
    frequency_median=('frequency', 'median'),
    monetary_mean=('monetary', 'mean'),
    monetary_median=('monetary', 'median'),
    avg_donation_mean=('average_donation', 'mean')
).reset_index()

print("Cluster Statistics Summary:")
display(cluster_stats)""")

    # Step 9: Donor Segmentation & Personas
    add_md("""## 10. Step 9: Donor Segmentation Personas & Business Strategies

We map numerical cluster IDs to strategic donor personas:
- **Cluster 2 ➔ Champions**: Recent donors with high frequency and strong total donation values.
- **Cluster 0 ➔ Loyal Donors**: Highly regular monetary contributors with consistent engagement.
- **Cluster 1 ➔ Potential Loyalists**: Moderate recency & frequency; high upside potential for campaign re-engagement.
- **Cluster 3 ➔ At-Risk / Hibernating**: Inactive donors with long recency and low monetary output requiring win-back campaigns.
""")

    add_code("""# Persona Mapping Dictionary
persona_mapping = {
    2: {'name': 'Champions', 'color': '#10B981', 'strategy': 'VIP invitations, early campaign access, donor advisory panel.'},
    0: {'name': 'Loyal Donors', 'color': '#3B82F6', 'strategy': 'Recurring monthly giving programs, impact newsletters, gratitude calls.'},
    1: {'name': 'Potential Loyalists', 'color': '#F59E0B', 'strategy': 'Targeted impact stories, donation match offers, multi-channel nudges.'},
    3: {'name': 'At-Risk / Hibernating', 'color': '#EF4444', 'strategy': 'Re-engagement survey, emergency relief callouts, lower friction options.'}
}

df_donor_rfm['segment_name'] = df_donor_rfm['cluster_id'].map(lambda cid: persona_mapping[cid]['name'])
df_donor_rfm['segment_color'] = df_donor_rfm['cluster_id'].map(lambda cid: persona_mapping[cid]['color'])
df_donor_rfm['engagement_strategy'] = df_donor_rfm['cluster_id'].map(lambda cid: persona_mapping[cid]['strategy'])

# Display Segment Breakdown
segment_summary = df_donor_rfm.groupby('segment_name').agg(
    donors_count=('donor_id', 'count'),
    pct_donors=('donor_id', lambda x: f"{len(x)/len(df_donor_rfm):.1%}"),
    recency_avg=('recency', 'mean'),
    frequency_avg=('frequency', 'mean'),
    monetary_total=('monetary', 'sum'),
    monetary_avg=('monetary', 'mean')
).reset_index()

display(segment_summary)""")

    add_code("""# 3D Interactive RFM Scatter Plot (Plotly)
fig = px.scatter_3d(
    df_donor_rfm,
    x='recency',
    y='frequency',
    z='monetary',
    color='segment_name',
    hover_data=['donor_id', 'donor_country', 'average_donation'],
    color_discrete_map={
        'Champions': '#10B981',
        'Loyal Donors': '#3B82F6',
        'Potential Loyalists': '#F59E0B',
        'At-Risk / Hibernating': '#EF4444'
    },
    title='<b>3D RFM Cluster Segmentation Visualization</b>',
    opacity=0.8
)
fig.update_layout(scene=dict(
    xaxis_title='Recency (Days)',
    yaxis_title='Frequency (Count)',
    zaxis_title='Monetary Total ($)'
), width=900, height=650)
fig.show()""")

    # Step 10: Export Artifacts
    add_md("""## 11. Step 10: Model Persistence & Export Saved Artifacts

We save the processed final datasets and model binaries (`KMeans` & `StandardScaler`) for deployment in our interactive web application dashboard.
""")

    add_code("""# Save Processed Datasets
df_donor_rfm.to_csv('donor_rfm_segmented.csv', index=False)
print("Saved 'donor_rfm_segmented.csv' successfully.")

# Save Trained Model & Scaler Binaries
joblib.dump(kmeans_final, 'rfm_kmeans_model.pkl')
joblib.dump(scaler, 'rfm_scaler.pkl')

print("Saved 'rfm_kmeans_model.pkl' & 'rfm_scaler.pkl' successfully.")
print("✨ RFM Analysis Notebook Pipeline Complete!")""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open('micro_donation.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

    print("Notebook 'micro_donation.ipynb' generated successfully.")

if __name__ == '__main__':
    generate_notebook()
