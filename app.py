import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, template_folder='templates', static_folder='static')

# Base Directory & Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEGMENTED_CSV = os.path.join(BASE_DIR, 'donor_rfm_segmented.csv')
MERGED_CSV = os.path.join(BASE_DIR, 'cleaned_merged_donations.csv')
MODEL_PKL = os.path.join(BASE_DIR, 'rfm_kmeans_model.pkl')
SCALER_PKL = os.path.join(BASE_DIR, 'rfm_scaler.pkl')

# Load Data & Models
df_segmented = pd.read_csv(SEGMENTED_CSV)
df_merged = pd.read_csv(MERGED_CSV)
kmeans_model = joblib.load(MODEL_PKL)
scaler_model = joblib.load(SCALER_PKL)

# Persona metadata
PERSONA_INFO = {
    'Champions': {
        'color': '#10B981',
        'badge': 'VIP Champion',
        'desc': 'Recent donors with high frequency and top monetary contributions.',
        'strategy': 'Provide VIP donor appreciation, early access to new humanitarian projects, and invite to donor advisory council.'
    },
    'Loyal Donors': {
        'color': '#3B82F6',
        'badge': 'Loyal Contributor',
        'desc': 'High monetary contributions and steady donation frequency.',
        'strategy': 'Enroll in monthly recurring giving programs, send quarterly impact newsletters, and personalized thank-you calls.'
    },
    'Potential Loyalists': {
        'color': '#F59E0B',
        'badge': 'Rising Donor',
        'desc': 'Moderate recency & frequency; high potential for upgrading to loyal tier.',
        'strategy': 'Send story-driven emergency campaigns, donation matching opportunities, and multi-channel engagement nudges.'
    },
    'At-Risk / Hibernating': {
        'color': '#EF4444',
        'badge': 'At-Risk / Lapsed',
        'desc': 'Long recency since last donation, low transaction frequency and total monetary output.',
        'strategy': 'Deploy win-back re-engagement surveys, present low-friction micro-donation options ($5-$10), and spotlight high-impact emergency relief.'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summary', methods=['GET'])
def get_summary():
    total_donors = int(len(df_segmented))
    total_monetary = float(df_segmented['monetary'].sum())
    avg_donation = float(df_segmented['average_donation'].mean())
    total_txs = int(df_segmented['frequency'].sum())
    
    # Top Country
    top_country = df_segmented['donor_country'].mode()[0] if not df_segmented.empty else 'N/A'
    
    # Top Campaign Category
    top_campaign_cat = df_merged['campaign_category'].mode()[0] if 'campaign_category' in df_merged.columns else 'N/A'

    # Segment Breakdown
    seg_breakdown = []
    for seg_name, group in df_segmented.groupby('segment_name'):
        info = PERSONA_INFO.get(seg_name, {'color': '#6B7280', 'badge': seg_name, 'desc': '', 'strategy': ''})
        seg_breakdown.append({
            'name': seg_name,
            'color': info['color'],
            'count': int(len(group)),
            'pct_donors': round(len(group) / total_donors * 100, 1),
            'total_monetary': float(group['monetary'].sum()),
            'pct_monetary': round(group['monetary'].sum() / total_monetary * 100, 1),
            'avg_recency': float(group['recency'].mean()),
            'avg_frequency': float(group['frequency'].mean()),
            'avg_monetary': float(group['monetary'].mean()),
            'strategy': info['strategy']
        })

    return jsonify({
        'total_donors': total_donors,
        'total_monetary': round(total_monetary, 2),
        'avg_donation': round(avg_donation, 2),
        'total_txs': total_txs,
        'top_country': top_country,
        'top_campaign_category': top_campaign_cat,
        'segments': seg_breakdown
    })

@app.route('/api/rfm_scatter', methods=['GET'])
def get_rfm_scatter():
    # Sample or send all points for 3D scatter plot
    data = []
    for _, row in df_segmented.iterrows():
        data.append({
            'donor_id': int(row['donor_id']),
            'recency': int(row['recency']),
            'frequency': int(row['frequency']),
            'monetary': float(row['monetary']),
            'avg_donation': float(row['average_donation']),
            'segment_name': row['segment_name'],
            'segment_color': row.get('segment_color', '#3B82F6'),
            'country': str(row['donor_country']),
            'donor_type': str(row['donor_type'])
        })
    return jsonify(data)

@app.route('/api/demographics', methods=['GET'])
def get_demographics():
    # Country breakdown
    country_counts = df_segmented['donor_country'].value_counts().head(10).to_dict()
    
    # Donor Type breakdown
    donor_type_counts = df_segmented['donor_type'].value_counts().to_dict()

    # Gender breakdown
    gender_counts = df_segmented['donor_gender'].value_counts().to_dict()

    # Age groups
    bins = [0, 25, 40, 55, 100]
    labels = ['<25', '25-40', '41-55', '55+']
    df_segmented['age_group'] = pd.cut(df_segmented['donor_age'], bins=bins, labels=labels)
    age_counts = df_segmented['age_group'].value_counts().to_dict()

    return jsonify({
        'countries': country_counts,
        'donor_types': donor_type_counts,
        'genders': gender_counts,
        'age_groups': age_counts
    })

@app.route('/api/campaigns', methods=['GET'])
def get_campaigns():
    # Campaign category stats
    cat_stats = df_merged.groupby('campaign_category').agg(
        tx_count=('transaction_id', 'count'),
        total_raised=('donation_amount', 'sum'),
        avg_donation=('donation_amount', 'mean')
    ).reset_index()

    cat_list = []
    for _, row in cat_stats.iterrows():
        cat_list.append({
            'category': row['campaign_category'],
            'tx_count': int(row['tx_count']),
            'total_raised': float(round(row['total_raised'], 2)),
            'avg_donation': float(round(row['avg_donation'], 2))
        })

    # Priority stats
    priority_stats = df_merged.groupby('priority_level')['donation_amount'].sum().to_dict()

    # Monthly Trends
    df_merged['year_month'] = pd.to_datetime(df_merged['transaction_date']).dt.to_period('M').astype(str)
    monthly = df_merged.groupby('year_month').agg(
        total_raised=('donation_amount', 'sum'),
        tx_count=('transaction_id', 'count')
    ).reset_index()

    monthly_list = []
    for _, row in monthly.iterrows():
        monthly_list.append({
            'month': row['year_month'],
            'total_raised': float(round(row['total_raised'], 2)),
            'tx_count': int(row['tx_count'])
        })

    return jsonify({
        'categories': cat_list,
        'priorities': priority_stats,
        'monthly_trends': monthly_list
    })

@app.route('/api/donors', methods=['GET'])
def get_donors():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 15))
    search = request.args.get('search', '').lower().strip()
    segment_filter = request.args.get('segment', '').strip()
    country_filter = request.args.get('country', '').strip()
    donor_type_filter = request.args.get('donor_type', '').strip()

    filtered = df_segmented.copy()

    if search:
        filtered = filtered[
            filtered['donor_id'].astype(str).str.contains(search) |
            filtered['donor_country'].str.lower().str.contains(search) |
            filtered['donor_type'].str.lower().str.contains(search)
        ]

    if segment_filter and segment_filter != 'All':
        filtered = filtered[filtered['segment_name'] == segment_filter]

    if country_filter and country_filter != 'All':
        filtered = filtered[filtered['donor_country'] == country_filter]

    if donor_type_filter and donor_type_filter != 'All':
        filtered = filtered[filtered['donor_type'] == donor_type_filter]

    total_count = len(filtered)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_data = filtered.iloc[start_idx:end_idx]

    records = []
    for _, row in page_data.iterrows():
        records.append({
            'donor_id': int(row['donor_id']),
            'age': int(row['donor_age']),
            'gender': str(row['donor_gender']),
            'country': str(row['donor_country']),
            'donor_type': str(row['donor_type']),
            'recency': int(row['recency']),
            'frequency': int(row['frequency']),
            'monetary': float(row['monetary']),
            'avg_donation': float(row['average_donation']),
            'rfm_score': str(row['rfm_score']),
            'segment_name': str(row['segment_name']),
            'segment_color': str(row.get('segment_color', '#3B82F6'))
        })

    return jsonify({
        'total_count': total_count,
        'page': page,
        'per_page': per_page,
        'total_pages': int(np.ceil(total_count / per_page)),
        'records': records
    })

@app.route('/api/donor/<int:donor_id>', methods=['GET'])
def get_donor_detail(donor_id):
    donor_rows = df_segmented[df_segmented['donor_id'] == donor_id]
    if donor_rows.empty:
        return jsonify({'error': 'Donor not found'}), 404

    row = donor_rows.iloc[0]
    tx_history = df_merged[df_merged['donor_id'] == donor_id].sort_values('transaction_date', ascending=False)

    tx_list = []
    for _, tx in tx_history.head(10).iterrows():
        tx_list.append({
            'transaction_id': int(tx['transaction_id']),
            'date': str(tx['transaction_date']),
            'amount': float(tx['donation_amount']),
            'payment_method': str(tx['payment_method']),
            'aid_country': str(tx['aid_country']),
            'aid_sector': str(tx['aid_sector']),
            'campaign_name': str(tx.get('campaign_name', 'General'))
        })

    info = PERSONA_INFO.get(row['segment_name'], {'color': '#3B82F6', 'badge': row['segment_name'], 'desc': '', 'strategy': ''})

    return jsonify({
        'donor_id': int(row['donor_id']),
        'age': int(row['donor_age']),
        'gender': str(row['donor_gender']),
        'country': str(row['donor_country']),
        'donor_type': str(row['donor_type']),
        'registration_date': str(row['registration_date']),
        'last_donation_date': str(row.get('last_donation_date', 'N/A')),
        'recency': int(row['recency']),
        'frequency': int(row['frequency']),
        'monetary': float(row['monetary']),
        'avg_donation': float(row['average_donation']),
        'rfm_score': str(row['rfm_score']),
        'segment_name': str(row['segment_name']),
        'segment_color': info['color'],
        'segment_badge': info['badge'],
        'segment_desc': info['desc'],
        'segment_strategy': info['strategy'],
        'transactions': tx_list
    })

@app.route('/api/predict', methods=['POST'])
def predict_rfm():
    req = request.get_json(force=True)
    
    # Check if donor_id supplied
    if 'donor_id' in req and req['donor_id']:
        try:
            did = int(req['donor_id'])
            found = df_segmented[df_segmented['donor_id'] == did]
            if not found.empty:
                r_val = float(found.iloc[0]['recency'])
                f_val = float(found.iloc[0]['frequency'])
                m_val = float(found.iloc[0]['monetary'])
            else:
                r_val = float(req.get('recency', 30))
                f_val = float(req.get('frequency', 5))
                m_val = float(req.get('monetary', 200))
        except ValueError:
            r_val = float(req.get('recency', 30))
            f_val = float(req.get('frequency', 5))
            m_val = float(req.get('monetary', 200))
    else:
        r_val = float(req.get('recency', 30))
        f_val = float(req.get('frequency', 5))
        m_val = float(req.get('monetary', 200))

    # Log transform
    r_log = np.log1p(r_val)
    f_log = np.log1p(f_val)
    m_log = np.log1p(m_val)

    # Scale using DataFrame with fitted feature names
    X_df = pd.DataFrame([[r_log, f_log, m_log]], columns=['recency_log', 'frequency_log', 'monetary_log'])
    X_input = scaler_model.transform(X_df)
    
    # Predict Cluster
    cluster_id = int(kmeans_model.predict(X_input)[0])

    # Calculate distance to cluster centers
    distances = np.linalg.norm(kmeans_model.cluster_centers_ - X_input, axis=1)
    confidence = float(round(1 / (1 + distances[cluster_id]) * 100, 1))

    # Persona mapping
    seg_names = {
        2: 'Champions',
        0: 'Loyal Donors',
        1: 'Potential Loyalists',
        3: 'At-Risk / Hibernating'
    }

    predicted_segment = seg_names.get(cluster_id, 'Potential Loyalists')
    info = PERSONA_INFO.get(predicted_segment, {'color': '#3B82F6', 'badge': predicted_segment, 'desc': '', 'strategy': ''})

    return jsonify({
        'recency_input': r_val,
        'frequency_input': f_val,
        'monetary_input': m_val,
        'cluster_id': cluster_id,
        'predicted_segment': predicted_segment,
        'segment_color': info['color'],
        'segment_badge': info['badge'],
        'segment_desc': info['desc'],
        'segment_strategy': info['strategy'],
        'confidence_score': confidence
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Launching Micro-Donation RFM Dashboard Flask Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
