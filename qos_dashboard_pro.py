# qos_dashboard_pro.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import joblib
import os
import hashlib
from io import BytesIO
import sqlite3
from contextlib import contextmanager

# -------------------------------
# Configuration & Constants
# -------------------------------
st.set_page_config(
    page_title="Call QoS Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DB_PATH = "qos_database.db"
MODEL_PATH = "qos_model.pkl"

# -------------------------------
# Database Setup
# -------------------------------
@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize database tables."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS call_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                latency REAL,
                jitter REAL,
                packet_loss REAL,
                qos_score REAL,
                call_id TEXT,
                is_anomaly BOOLEAN DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password_hash TEXT,
                role TEXT,
                last_login DATETIME
            )
        """)
        conn.commit()

def init_sample_data():
    """Initialize sample data if database is empty."""
    with get_db_connection() as conn:
        count = conn.execute("SELECT COUNT(*) as count FROM call_metrics").fetchone()['count']
        
        if count == 0:
            st.info("📊 Generating sample data...")
            
            # Generate 100 sample records for the past 24 hours
            base_time = datetime.now() - timedelta(hours=24)
            sample_data = []
            
            for i in range(100):
                timestamp = base_time + timedelta(minutes=i*14.4)  # Spread over 24 hours
                
                # Generate realistic network metrics
                latency = max(10, np.random.normal(45, 20))
                jitter = max(1, np.random.normal(8, 4))
                packet_loss = max(0, np.random.uniform(0, 3))
                
                # Simple QoS calculation (will be replaced by ML model)
                qos_score = max(0, min(100, 100 - (latency/5) - (jitter*2) - (packet_loss*10) + np.random.normal(0, 5)))
                
                sample_data.append((
                    timestamp,
                    latency,
                    jitter,
                    packet_loss,
                    qos_score,
                    f"call_{1000 + i}",
                    qos_score < 60  # Anomaly if QoS < 60
                ))
            
            # Insert sample data
            conn.executemany(
                """INSERT INTO call_metrics 
                (timestamp, latency, jitter, packet_loss, qos_score, call_id, is_anomaly) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                sample_data
            )
            conn.commit()
            st.success("✅ Sample data generated!")

# -------------------------------
# Security & Authentication
# -------------------------------
def hash_password(password):
    """Secure password hashing."""
    return hashlib.sha256(password.encode()).hexdigest()

def init_default_users():
    """Initialize default users if they don't exist."""
    default_users = [
        ("admin", "admin123", "Admin"),
        ("engineer", "engineer123", "Engineer"),
        ("analyst", "analyst123", "Analyst")
    ]
    
    with get_db_connection() as conn:
        for username, password, role in default_users:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, hash_password(password), role)
                )
        conn.commit()

# -------------------------------
# ML Model & Data Processing
# -------------------------------
@st.cache_resource
def load_ml_model():
    """Load ML model with efficient caching."""
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ ML model not found. Using fallback scoring.")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except:
        st.warning("⚠️ Error loading ML model. Using fallback scoring.")
        return None

def predict_qos_score(model, latency, jitter, packet_loss):
    """Predict QoS score using ML model or fallback calculation."""
    if model:
        features = np.array([[latency, jitter, packet_loss]])
        return float(np.clip(model.predict(features)[0], 0, 100))
    else:
        # Fallback calculation if model not available
        return max(0, min(100, 100 - (latency/4) - (jitter*1.5) - (packet_loss*8) + np.random.normal(0, 3)))

def detect_anomaly(latency, jitter, packet_loss, qos_score):
    """Simple anomaly detection based on thresholds."""
    return (latency > 100 or jitter > 20 or packet_loss > 5 or qos_score < 60)

# -------------------------------
# Data Management
# -------------------------------
def save_call_metrics(metrics_dict):
    """Save call metrics to database efficiently."""
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO call_metrics 
               (timestamp, latency, jitter, packet_loss, qos_score, call_id, is_anomaly) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (metrics_dict['timestamp'], metrics_dict['latency'], metrics_dict['jitter'],
             metrics_dict['packet_loss'], metrics_dict['qos_score'], 
             metrics_dict['call_id'], metrics_dict.get('is_anomaly', False))
        )
        conn.commit()

def get_recent_metrics(limit=100):
    """Efficiently retrieve recent metrics from database."""
    with get_db_connection() as conn:
        df = pd.read_sql(
            f"SELECT * FROM call_metrics ORDER BY timestamp DESC LIMIT {limit}",
            conn
        )
    return df

def get_metrics_count():
    """Get total number of metrics in database."""
    with get_db_connection() as conn:
        count = conn.execute("SELECT COUNT(*) as count FROM call_metrics").fetchone()['count']
    return count

# -------------------------------
# UI Components
# -------------------------------
def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a standardized metric card."""
    return st.metric(title, value, delta)

def create_quality_gauge(score):
    """Create a beautiful gauge chart for QoS score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "QoS Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightcoral"},
                {'range': [60, 80], 'color': "lightyellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# -------------------------------
# Main Application
# -------------------------------
def main():
    # Initialize database and data
    init_database()
    init_default_users()
    init_sample_data()
    
    # Session state initialization
    if 'authenticated' not in st.session_state:
        st.session_state.update({
            'authenticated': False,
            'username': None,
            'role': None,
            'model': load_ml_model(),
            'auto_refresh': False,
            'last_update': datetime.now()
        })

    # Login Page
    if not st.session_state.authenticated:
        show_login_page()
        return

    # Main Dashboard
    show_dashboard()

def show_login_page():
    """Modern login page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔐 Call QoS Analytics Pro")
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            if st.form_submit_button("🚀 Login", use_container_width=True):
                authenticate_user(username, password)
        
        # Demo credentials
        with st.expander("ℹ️ Demo Credentials"):
            st.code("""
            Admin:     admin / admin123
            Engineer:  engineer / engineer123
            Analyst:   analyst / analyst123
            """)

def authenticate_user(username, password):
    """Authenticate user against database."""
    with get_db_connection() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        
        if user and user['password_hash'] == hash_password(password):
            st.session_state.update({
                'authenticated': True,
                'username': username,
                'role': user['role'],
                'last_login': datetime.now()
            })
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

def show_dashboard():
    """Main dashboard display."""
    # Sidebar
    with st.sidebar:
        display_sidebar()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title(f"📊 Call Quality Dashboard")
        st.caption(f"Welcome back, {st.session_state.username} ({st.session_state.role})")
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Display metrics and charts
    display_metrics()
    display_charts()
    display_alerts()

def display_sidebar():
    """Sidebar components."""
    st.title("⚙️ Controls")
    
    # Database info
    total_calls = get_metrics_count()
    st.info(f"📊 Total calls: {total_calls}")
    
    # Real-time controls
    if st.button("📞 Simulate New Call", use_container_width=True, type="primary"):
        simulate_new_call()
        st.rerun()
    
    if st.button("📋 Generate 10 Calls", use_container_width=True):
        for _ in range(10):
            simulate_new_call()
        st.rerun()
    
    st.session_state.auto_refresh = st.checkbox(
        "🔄 Live Auto-Refresh (3s)", 
        value=st.session_state.auto_refresh
    )
    
    if st.session_state.auto_refresh:
        time.sleep(3)
        simulate_new_call()
        st.rerun()
    
    st.markdown("---")
    
    # Data export
    if st.session_state.role in ["Admin", "Engineer"]:
        st.subheader("📤 Export Data")
        if st.button("💾 Export to CSV", use_container_width=True):
            export_data()
    
    st.markdown("---")
    
    # System info
    st.subheader("ℹ️ System Info")
    st.info(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    if st.button("🚪 Logout", use_container_width=True):
        logout_user()

def simulate_new_call():
    """Simulate and save a new call."""
    # Generate realistic metrics
    metrics = {
        'timestamp': datetime.now(),
        'latency': max(10, np.random.normal(50, 25)),
        'jitter': max(1, np.random.normal(10, 6)),
        'packet_loss': max(0, np.random.uniform(0, 4)),
        'call_id': f"call_{int(time.time() * 1000)}"  # Unique ID
    }
    
    # Predict QoS score
    metrics['qos_score'] = predict_qos_score(
        st.session_state.model,
        metrics['latency'],
        metrics['jitter'],
        metrics['packet_loss']
    )
    
    # Detect anomalies
    metrics['is_anomaly'] = detect_anomaly(
        metrics['latency'],
        metrics['jitter'],
        metrics['packet_loss'],
        metrics['qos_score']
    )
    
    # Save to database
    save_call_metrics(metrics)
    st.session_state.last_update = datetime.now()

def display_metrics():
    """Display key metrics."""
    df = get_recent_metrics(50)
    if df.empty:
        st.info("No data available. Click 'Simulate New Call' to generate data!")
        return
    
    latest = df.iloc[0]
    prev = df.iloc[1] if len(df) > 1 else latest
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_lat = latest['latency'] - prev['latency']
        create_metric_card("Latency (ms)", f"{latest['latency']:.1f}", f"{delta_lat:+.1f}")
    
    with col2:
        delta_jit = latest['jitter'] - prev['jitter']
        create_metric_card("Jitter (ms)", f"{latest['jitter']:.1f}", f"{delta_jit:+.1f}")
    
    with col3:
        delta_pl = latest['packet_loss'] - prev['packet_loss']
        create_metric_card("Packet Loss (%)", f"{latest['packet_loss']:.1f}%", f"{delta_pl:+.1f}%")
    
    with col4:
        st.plotly_chart(create_quality_gauge(latest['qos_score']), use_container_width=True)

def display_charts():
    """Display analytical charts."""
    df = get_recent_metrics(100)
    if len(df) < 2:
        st.info("Need more data to display charts. Simulate more calls!")
        return
    
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "📊 Quality Distribution", "📱 Real-time Metrics"])
    
    with tab1:
        fig = px.line(df, x='timestamp', y='qos_score', title='QoS Score Trend Over Time')
        fig.add_hrect(y0=0, y1=60, fillcolor="red", opacity=0.2, line_width=0)
        fig.add_hrect(y0=60, y1=80, fillcolor="yellow", opacity=0.2, line_width=0)
        fig.add_hrect(y0=80, y1=100, fillcolor="green", opacity=0.2, line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.histogram(df, x='qos_score', title='Quality Score Distribution', 
                          color_discrete_sequence=['blue'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        metric = st.selectbox("Select Metric", ['latency', 'jitter', 'packet_loss'])
        fig = px.line(df, x='timestamp', y=metric, title=f'{metric.title()} Over Time')
        st.plotly_chart(fig, use_container_width=True)

def display_alerts():
    """Display system alerts."""
    df = get_recent_metrics(20)
    critical = df[df['qos_score'] < 60]
    
    if not critical.empty:
        st.error("🚨 Critical Alerts (QoS < 60)")
        for _, row in critical.iterrows():
            st.error(f"**{row['call_id']}** - QoS: {row['qos_score']:.1f} | Latency: {row['latency']:.1f}ms | Packet Loss: {row['packet_loss']:.1f}%")
    else:
        st.success("✅ No critical alerts - System normal")

def export_data():
    """Export data functionality."""
    df = get_recent_metrics(1000)
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"qos_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )


def logout_user():
    """Clean logout functionality."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
print("Hi")
if __name__ == "__main__":
    main()