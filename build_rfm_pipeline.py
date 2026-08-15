import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def run_pipeline():
    print("Step 1: Loading raw datasets...")
    donors = pd.read_csv('01_donor_master.csv')
    txs = pd.read_csv('02_donation_transactions.csv')
    campaigns = pd.read_csv('03_humanitarian_campaigns.csv')

    print(f"Donors: {donors.shape}, Transactions: {txs.shape}, Campaigns: {campaigns.shape}")

    print("Step 2: Data Cleaning & Preprocessing...")
    txs_succ = txs[txs['successful_payment'] == 1].copy()
    txs_succ['transaction_date'] = pd.to_datetime(txs_succ['transaction_date'])
    
    snapshot_date = pd.to_datetime('2016-11-21')

    print("Step 3: Calculating RFM Metrics...")
    rfm = txs_succ.groupby('donor_id').agg(
        last_donation_date=('transaction_date', 'max'),
        frequency=('transaction_id', 'count'),
        monetary=('donation_amount', 'sum'),
        average_donation=('donation_amount', 'mean')
    ).reset_index()

    rfm['last_donation_date_str'] = rfm['last_donation_date'].dt.strftime('%Y-%m-%d')
    rfm['recency'] = (snapshot_date - rfm['last_donation_date']).dt.days

    donor_rfm = pd.merge(donors, rfm[['donor_id', 'last_donation_date_str', 'recency', 'frequency', 'monetary', 'average_donation']], on='donor_id', how='left')
    donor_rfm.rename(columns={'last_donation_date_str': 'last_donation_date'}, inplace=True)

    # Fill missing values for donors without transactions (if any)
    donor_rfm['frequency'] = donor_rfm['frequency'].fillna(0).astype(int)
    donor_rfm['monetary'] = donor_rfm['monetary'].fillna(0.0).round(2)
    donor_rfm['average_donation'] = donor_rfm['average_donation'].fillna(0.0).round(2)
    max_rec = (snapshot_date - pd.to_datetime(donor_rfm['registration_date'])).dt.days
    donor_rfm['recency'] = donor_rfm['recency'].fillna(max_rec).astype(int)

    # Quintile RFM Scoring (1 to 5)
    donor_rfm['r_score'] = pd.qcut(donor_rfm['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    donor_rfm['f_score'] = pd.qcut(donor_rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    donor_rfm['m_score'] = pd.qcut(donor_rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    donor_rfm['rfm_score'] = donor_rfm['r_score'].astype(str) + donor_rfm['f_score'].astype(str) + donor_rfm['m_score'].astype(str)

    print("Step 4: Log Transformation & StandardScaler Scaling...")
    donor_rfm['recency_log'] = np.log1p(donor_rfm['recency'])
    donor_rfm['frequency_log'] = np.log1p(donor_rfm['frequency'])
    donor_rfm['monetary_log'] = np.log1p(donor_rfm['monetary'])

    features = ['recency_log', 'frequency_log', 'monetary_log']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(donor_rfm[features])

    print("Step 5: Training K-Means Clustering Model (k=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    donor_rfm['cluster_id'] = kmeans.fit_predict(X_scaled)

    # Model Evaluation Metrics
    sil_score = silhouette_score(X_scaled, donor_rfm['cluster_id'])
    ch_score = calinski_harabasz_score(X_scaled, donor_rfm['cluster_id'])
    db_score = davies_bouldin_score(X_scaled, donor_rfm['cluster_id'])

    print(f"Model Evaluation: Silhouette={sil_score:.4f}, Calinski-Harabasz={ch_score:.2f}, Davies-Bouldin={db_score:.4f}")

    # Map Clusters to Personas
    cluster_means = donor_rfm.groupby('cluster_id')[['recency', 'frequency', 'monetary']].mean()
    print("\nCluster Center Profiles:")
    print(cluster_means)

    segment_map = {
        2: {'name': 'Champions', 'color': '#10B981', 'desc': 'Recent, high-frequency donors with high total contributions.'},
        0: {'name': 'Loyal Donors', 'color': '#3B82F6', 'desc': 'High monetary value and regular frequency, moderate recency.'},
        1: {'name': 'Potential Loyalists', 'color': '#F59E0B', 'desc': 'Moderate recency & frequency, good growth potential.'},
        3: {'name': 'At-Risk / Hibernating', 'color': '#EF4444', 'desc': 'High recency, low frequency, low monetary output.'}
    }

    donor_rfm['segment_name'] = donor_rfm['cluster_id'].map(lambda cid: segment_map[cid]['name'])
    donor_rfm['segment_color'] = donor_rfm['cluster_id'].map(lambda cid: segment_map[cid]['color'])
    donor_rfm['segment_desc'] = donor_rfm['cluster_id'].map(lambda cid: segment_map[cid]['desc'])

    print("\nStep 6: Exporting Processed Datasets & Saved Model Artifacts...")
    donor_rfm.to_csv('donor_rfm_segmented.csv', index=False)

    merged_txs = pd.merge(txs, donors, on='donor_id', how='left')
    merged_txs = pd.merge(merged_txs, campaigns, on='campaign_id', how='left')
    merged_txs = pd.merge(merged_txs, donor_rfm[['donor_id', 'cluster_id', 'segment_name', 'rfm_score']], on='donor_id', how='left')

    merged_txs.to_csv('cleaned_merged_donations.csv', index=False)

    joblib.dump(kmeans, 'rfm_kmeans_model.pkl')
    joblib.dump(scaler, 'rfm_scaler.pkl')

    print("Pipeline execution successful! All artifacts saved.")

if __name__ == '__main__':
    run_pipeline()
