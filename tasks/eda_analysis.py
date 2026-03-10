import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import shapiro, normaltest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f3c88;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .insight-box {
        background-color: #e8f4fd;
        border-left: 4px solid #17a2b8;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .analysis-step {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #4A90E2;
    }
    .insight-highlight {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #FF6B6B;
    }
    .stat-box {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def perform_eda(cleaned_data):
    """
    🎯 Perform professional-grade exploratory data analysis with comprehensive insights
    
    Parameters:
    -----------
    cleaned_data : dict
        Dictionary containing cleaned dataframes:
        - 'customers': Customer demographic data
        - 'loans': Loan portfolio data
        - 'payments': Payment transaction data
    
    Returns:
    --------
    None: Displays interactive EDA dashboard in Streamlit
    """
    
    try:
        # Header with gradient
        st.markdown('<div class="main-header">🔬 Advanced Exploratory Data Analysis Dashboard</div>', unsafe_allow_html=True)
        
        # 📊 ACTION: Show analysis progress
        with st.status("🚀 Initializing Analysis...", expanded=True) as status:
            st.write("🔍 Loading datasets...")
            customers = cleaned_data['customers'].copy()
            loans = cleaned_data['loans'].copy()
            payments = cleaned_data['payments'].copy()
            
            st.write("✅ Validating data quality...")
            # Basic validation
            if customers.empty or loans.empty:
                st.error("❌ Critical data missing. Please check your input data.")
                return
            
            st.write("📈 Setting up analysis environment...")
            status.update(label="✅ Analysis Initialized Successfully!", state="complete", expanded=False)
        
        # Sidebar for controls
        with st.sidebar:
            st.markdown("### ⚙️ Analysis Controls")
            
            # Analysis settings
            st.markdown("#### Analysis Parameters")
            confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
            outlier_threshold = st.slider("Outlier Threshold (σ)", 2.0, 4.0, 3.0, 0.1)
            
            # Visualization settings
            st.markdown("#### Visualization Settings")
            color_scheme = st.selectbox("Color Scheme", 
                                      ["Viridis", "Plasma", "Inferno", "Blues", "RdBu", "Set2", "Dark24"])
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                enable_sampling = st.checkbox("Enable Data Sampling", value=False)
                if enable_sampling:
                    sample_size = st.slider("Sample Size (%)", 10, 100, 50, 5)
                
                show_debug = st.checkbox("Show Debug Information", value=False)
                enable_ml = st.checkbox("Enable ML Analysis", value=True)
        
        # Apply sampling if enabled
        if 'enable_sampling' in locals() and enable_sampling:
            sample_size_frac = sample_size / 100
            customers = customers.sample(frac=sample_size_frac, random_state=42)
            loans = loans.sample(frac=sample_size_frac, random_state=42)
            payments = payments.sample(frac=sample_size_frac, random_state=42)
            st.info(f"📊 Using {sample_size}% sample of data for analysis")
        
        # Professional color palette based on selection
        color_palettes = {
            "Viridis": px.colors.sequential.Viridis,
            "Plasma": px.colors.sequential.Plasma,
            "Inferno": px.colors.sequential.Inferno,
            "Blues": px.colors.sequential.Blues,
            "RdBu": px.colors.diverging.RdBu,
            "Set2": px.colors.qualitative.Set2,
            "Dark24": px.colors.qualitative.Dark24
        }
        PROFESSIONAL_COLORS = color_palettes.get(color_scheme, px.colors.sequential.Viridis)
        
        # 📊 Create analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Executive Summary", 
            "👤 Customer Analytics", 
            "💰 Loan Portfolio", 
            "💳 Payment Intelligence", 
            "📊 Statistical Analysis",
            "🔍 Advanced Analytics",
            "📋 Data Quality"
        ])
        
        with tab1:
            st.markdown('<div class="sub-header">📈 Executive Dashboard & KPIs</div>', unsafe_allow_html=True)
            
            # Show analysis steps
            st.markdown("""
            <div class="analysis-step">
                <strong>📋 Analysis Steps:</strong><br>
                1. Calculating Key Performance Indicators (KPIs)<br>
                2. Portfolio Health Assessment<br>
                3. Risk Level Evaluation<br>
                4. Performance Trend Analysis
            </div>
            """, unsafe_allow_html=True)
            
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_customers = customers.shape[0]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>👥 Total Customers</h3>
                    <h2>{total_customers:,}</h2>
                    <p>Unique customer records</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                default_rate = loans['default_flag'].mean() if 'default_flag' in loans.columns else 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {'#ff6b6b' if default_rate > 0.05 else '#4CAF50'} 0%, {'#c44569' if default_rate > 0.05 else '#2ecc71'} 100%);">
                    <h3>⚠️ Default Rate</h3>
                    <h2>{default_rate:.2%}</h2>
                    <p>{"⚠️ Above Threshold" if default_rate > 0.05 else "✅ Within Limits"}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_loan_amount = loans['loan_amount_usd'].mean() if 'loan_amount_usd' in loans.columns else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 Avg Loan Size</h3>
                    <h2>${avg_loan_amount:,.0f}</h2>
                    <p>Average loan amount</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                total_portfolio = loans['loan_amount_usd'].sum() if 'loan_amount_usd' in loans.columns else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏦 Total Portfolio</h3>
                    <h2>${total_portfolio:,.0f}</h2>
                    <p>Total loan portfolio value</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Portfolio Health Dashboard
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Portfolio composition
                st.markdown("#### 🏦 Portfolio Composition")
                if 'loan_status' in loans.columns:
                    status_composition = loans['loan_status'].value_counts()
                    fig = px.sunburst(
                        names=status_composition.index,
                        parents=[''] * len(status_composition),
                        values=status_composition.values,
                        title="Loan Status Distribution",
                        color=status_composition.values,
                        color_continuous_scale=PROFESSIONAL_COLORS
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="insight-highlight">
                        <strong>🔍 Insight:</strong> This visualization shows the distribution of loans across different status categories. 
                        A healthy portfolio typically has a high percentage of 'Active' loans and low 'Default' rates.
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Quick insights panel
                st.markdown("#### 📊 Performance Snapshot")
                
                insights_container = st.container()
                with insights_container:
                    # Customer growth trend
                    if 'customer_since' in customers.columns:
                        try:
                            customers['customer_since'] = pd.to_datetime(customers['customer_since'], errors='coerce')
                            monthly_growth = customers.set_index('customer_since').resample('M').size().cumsum()
                            if len(monthly_growth) > 1:
                                growth_rate = ((monthly_growth.iloc[-1] / monthly_growth.iloc[-2]) - 1) * 100
                                trend = "📈 Growing" if growth_rate > 0 else "📉 Declining"
                                st.markdown(f"**Customer Growth**: {trend} ({growth_rate:.1f}%)")
                        except:
                            pass
                    
                    # Default trend
                    if 'application_date' in loans.columns and 'default_flag' in loans.columns:
                        try:
                            loans['application_date'] = pd.to_datetime(loans['application_date'], errors='coerce')
                            monthly_defaults = loans.groupby(loans['application_date'].dt.to_period('M'))['default_flag'].mean()
                            if len(monthly_defaults) > 1:
                                default_trend = monthly_defaults.iloc[-1] - monthly_defaults.iloc[-2]
                                trend_icon = "⚠️" if default_trend > 0 else "✅"
                                st.markdown(f"**Default Trend**: {trend_icon} {default_trend:+.2%}")
                        except:
                            pass
                    
                    # Payment performance
                    if 'payment_status' in payments.columns:
                        on_time_rate = (payments['payment_status'] == 'on_time').mean() if 'on_time' in payments['payment_status'].values else 0
                        performance = "Excellent" if on_time_rate > 0.9 else "Good" if on_time_rate > 0.8 else "Needs Attention"
                        st.markdown(f"**Payment Performance**: {performance} ({on_time_rate:.1%})")
                    
                    # Portfolio concentration
                    if 'loan_amount_usd' in loans.columns:
                        try:
                            sorted_loans = loans['loan_amount_usd'].sort_values(ascending=False)
                            top_10_count = max(1, int(len(sorted_loans) * 0.1))
                            top_10_pct = sorted_loans.head(top_10_count).sum()
                            total_portfolio_val = sorted_loans.sum()
                            concentration = top_10_pct / total_portfolio_val if total_portfolio_val > 0 else 0
                            concentration_status = "High" if concentration > 0.3 else "Moderate" if concentration > 0.2 else "Low"
                            st.markdown(f"**Concentration Risk**: {concentration_status} ({concentration:.1%})")
                        except:
                            pass
        
        with tab2:
            st.markdown('<div class="sub-header">👤 Customer Analytics & Segmentation</div>', unsafe_allow_html=True)
            
            # Show analysis steps
            st.markdown("""
            <div class="analysis-step">
                <strong>📋 Analysis Steps:</strong><br>
                1. Demographic distribution analysis<br>
                2. Income pattern analysis<br>
                3. Customer segmentation<br>
                4. Behavioral pattern identification
            </div>
            """, unsafe_allow_html=True)
            
            # Customer segmentation analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution with statistical summary
                if 'age' in customers.columns:
                    st.markdown("##### 👥 Age Distribution Analysis")
                    
                    # Calculate statistics
                    age_data = customers['age'].dropna()
                    if len(age_data) > 0:
                        age_stats = age_data.describe()
                        
                        # Create distribution plot with statistical annotations
                        fig = go.Figure()
                        
                        # Histogram
                        fig.add_trace(go.Histogram(
                            x=age_data,
                            nbinsx=30,
                            name='Age Distribution',
                            marker_color=PROFESSIONAL_COLORS[0],
                            opacity=0.7
                        ))
                        
                        # Add mean and median lines
                        fig.add_vline(x=age_stats['mean'], line_dash="dash", 
                                    line_color="red", annotation_text=f"Mean: {age_stats['mean']:.1f}")
                        fig.add_vline(x=age_stats['50%'], line_dash="dash", 
                                    line_color="green", annotation_text=f"Median: {age_stats['50%']:.1f}")
                        
                        # Add normal distribution curve
                        x_norm = np.linspace(age_data.min(), age_data.max(), 100)
                        y_norm = stats.norm.pdf(x_norm, age_stats['mean'], age_stats['std'])
                        fig.add_trace(go.Scatter(x=x_norm, y=y_norm * len(age_data) * (age_data.max() - age_data.min()) / 30,
                                               mode='lines', name='Normal Distribution', line=dict(color='black', width=2)))
                        
                        fig.update_layout(
                            title="Age Distribution with Statistical Analysis",
                            xaxis_title="Age",
                            yaxis_title="Frequency",
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical summary box
                        with st.expander("📊 Statistical Summary"):
                            cols = st.columns(4)
                            cols[0].metric("Mean", f"{age_stats['mean']:.1f}")
                            cols[1].metric("Std Dev", f"{age_stats['std']:.1f}")
                            cols[2].metric("Skewness", f"{age_data.skew():.2f}")
                            cols[3].metric("Kurtosis", f"{age_data.kurtosis():.2f}")
                    else:
                        st.warning("No age data available for analysis")
            
            with col2:
                # Customer segmentation by income and age
                if 'annual_income_usd' in customers.columns and 'age' in customers.columns:
                    st.markdown("##### 🎯 Customer Segmentation")
                    
                    # Filter out missing values
                    seg_data = customers[['age', 'annual_income_usd']].dropna()
                    
                    if len(seg_data) > 10:
                        # Create scatter plot
                        fig = px.scatter(
                            seg_data,
                            x='age',
                            y='annual_income_usd',
                            title="Customer Segmentation by Age and Income",
                            color_discrete_sequence=[PROFESSIONAL_COLORS[1]],
                            trendline="ols"
                        )
                        
                        fig.update_traces(marker=dict(size=8, opacity=0.6, line=dict(width=1, color='DarkSlateGrey')))
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate correlation
                        correlation = seg_data['age'].corr(seg_data['annual_income_usd'])
                        
                        st.markdown(f"""
                        <div class="stat-box">
                            <strong>Correlation Analysis:</strong><br>
                            <strong>Correlation (Age vs Income):</strong> {correlation:.3f}<br>
                            <strong>Interpretation:</strong> {
                                "Strong positive relationship" if correlation > 0.7 else 
                                "Moderate positive relationship" if correlation > 0.3 else 
                                "Weak positive relationship" if correlation > 0 else 
                                "Weak negative relationship" if correlation > -0.3 else 
                                "Moderate negative relationship" if correlation > -0.7 else 
                                "Strong negative relationship"
                            }
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Insufficient data for segmentation analysis")
            
            # Customer Education Analysis
            if 'education_level' in customers.columns:
                st.markdown("##### 🎓 Education Level Analysis")
                
                edu_dist = customers['education_level'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig = px.bar(
                        x=edu_dist.index,
                        y=edu_dist.values,
                        title="Education Level Distribution",
                        labels={'x': 'Education Level', 'y': 'Count'},
                        color=edu_dist.values,
                        color_continuous_scale=PROFESSIONAL_COLORS
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig = px.pie(
                        values=edu_dist.values,
                        names=edu_dist.index,
                        title="Education Level Proportion",
                        hole=0.3,
                        color_discrete_sequence=PROFESSIONAL_COLORS
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown('<div class="sub-header">💰 Loan Portfolio Analysis</div>', unsafe_allow_html=True)
            
            # Show analysis steps
            st.markdown("""
            <div class="analysis-step">
                <strong>📋 Analysis Steps:</strong><br>
                1. Loan amount distribution analysis<br>
                2. Interest rate pattern analysis<br>
                3. Default risk assessment<br>
                4. Portfolio concentration analysis
            </div>
            """, unsafe_allow_html=True)
            
            # Loan distribution analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Loan amount distribution with box plot
                if 'loan_amount_usd' in loans.columns:
                    st.markdown("##### 📊 Loan Amount Analysis")
                    
                    loan_amounts = loans['loan_amount_usd'].dropna()
                    
                    if len(loan_amounts) > 0:
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Distribution', 'Box Plot with Outliers'),
                            vertical_spacing=0.15
                        )
                        
                        # Histogram
                        fig.add_trace(
                            go.Histogram(
                                x=loan_amounts,
                                nbinsx=50,
                                name='Distribution',
                                marker_color=PROFESSIONAL_COLORS[0]
                            ),
                            row=1, col=1
                        )
                        
                        # Box plot
                        fig.add_trace(
                            go.Box(
                                y=loan_amounts,
                                name='Loan Amount',
                                boxpoints='outliers',
                                marker_color=PROFESSIONAL_COLORS[1]
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical analysis
                        loan_stats = loan_amounts.describe()
                        with st.expander("📈 Loan Amount Statistics"):
                            st.write(f"**Mean**: ${loan_stats['mean']:,.0f}")
                            st.write(f"**Median**: ${loan_stats['50%']:,.0f}")
                            st.write(f"**Std Dev**: ${loan_stats['std']:,.0f}")
                            st.write(f"**Range**: ${loan_amounts.min():,.0f} - ${loan_amounts.max():,.0f}")
                            st.write(f"**IQR**: ${loan_stats['75%'] - loan_stats['25%']:,.0f}")
                    else:
                        st.warning("No loan amount data available")
            
            with col2:
                # Default risk analysis
                if 'default_flag' in loans.columns:
                    st.markdown("##### ⚠️ Default Risk Analysis")
                    
                    # Calculate default statistics
                    default_stats = loans['default_flag'].describe()
                    total_loans = len(loans)
                    default_count = loans['default_flag'].sum()
                    default_rate = default_count / total_loans if total_loans > 0 else 0
                    
                    col1_stats, col2_stats = st.columns(2)
                    with col1_stats:
                        st.metric("Total Loans", f"{total_loans:,}")
                    with col2_stats:
                        st.metric("Default Rate", f"{default_rate:.2%}")
                    
                    # Default by loan purpose
                    if 'loan_purpose' in loans.columns:
                        default_by_purpose = loans.groupby('loan_purpose')['default_flag'].mean().sort_values(ascending=False)
                        
                        if len(default_by_purpose) > 0:
                            fig = px.bar(
                                x=default_by_purpose.index,
                                y=default_by_purpose.values,
                                title="Default Rate by Loan Purpose",
                                labels={'x': 'Loan Purpose', 'y': 'Default Rate'},
                                color=default_by_purpose.values,
                                color_continuous_scale='RdYlGn_r'
                            )
                            fig.update_layout(yaxis_tickformat=".1%")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Interest rate vs default analysis
                    if 'interest_rate_pct' in loans.columns:
                        st.markdown("##### 📈 Interest Rate Analysis")
                        
                        # Create scatter plot
                        fig = px.scatter(
                            loans,
                            x='interest_rate_pct',
                            y='loan_amount_usd' if 'loan_amount_usd' in loans.columns else None,
                            color='default_flag',
                            title="Interest Rate vs Loan Amount by Default Status",
                            color_discrete_sequence=['green', 'red'],
                            opacity=0.6
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown('<div class="sub-header">💳 Payment Behavior Intelligence</div>', unsafe_allow_html=True)
            
            # Show analysis steps
            st.markdown("""
            <div class="analysis-step">
                <strong>📋 Analysis Steps:</strong><br>
                1. Payment timeline analysis<br>
                2. Payment status distribution<br>
                3. Payment pattern identification<br>
                4. Late payment analysis
            </div>
            """, unsafe_allow_html=True)
            
            if not payments.empty:
                # Payment Status Analysis
                st.markdown("##### 📊 Payment Status Overview")
                
                if 'payment_status' in payments.columns:
                    status_dist = payments['payment_status'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Donut chart
                        fig = px.pie(
                            values=status_dist.values,
                            names=status_dist.index,
                            title="Payment Status Distribution",
                            hole=0.4,
                            color_discrete_sequence=PROFESSIONAL_COLORS
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Bar chart
                        fig = px.bar(
                            x=status_dist.index,
                            y=status_dist.values,
                            title="Payment Status Count",
                            labels={'x': 'Status', 'y': 'Count'},
                            color=status_dist.values,
                            color_continuous_scale=PROFESSIONAL_COLORS
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Payment Amount Analysis
                if 'payment_amount_usd' in payments.columns:
                    st.markdown("##### 💰 Payment Amount Analysis")
                    
                    payment_amounts = payments['payment_amount_usd'].dropna()
                    
                    if len(payment_amounts) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram
                            fig = px.histogram(
                                payment_amounts,
                                nbins=50,
                                title="Payment Amount Distribution",
                                labels={'value': 'Payment Amount ($)', 'count': 'Frequency'},
                                color_discrete_sequence=[PROFESSIONAL_COLORS[0]]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Box plot
                            fig = px.box(
                                payment_amounts,
                                title="Payment Amount Statistics",
                                points="outliers"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Payment amount statistics
                        with st.expander("📊 Payment Amount Statistics"):
                            stats_df = payment_amounts.describe().to_frame().T
                            st.dataframe(stats_df, use_container_width=True)
                
                # Time-based payment analysis
                if 'payment_date' in payments.columns:
                    st.markdown("##### ⏰ Payment Timeline Analysis")
                    
                    try:
                        payments['payment_date'] = pd.to_datetime(payments['payment_date'], errors='coerce')
                        payments_clean = payments.dropna(subset=['payment_date'])
                        
                        if not payments_clean.empty:
                            # Monthly trend
                            payments_clean['month'] = payments_clean['payment_date'].dt.to_period('M')
                            monthly_stats = payments_clean.groupby('month').agg({
                                'payment_amount_usd': 'sum',
                                'payment_id': 'count'
                            }).reset_index()
                            monthly_stats['month'] = monthly_stats['month'].astype(str)
                            
                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=('Total Payment Amount by Month', 'Number of Payments by Month'),
                                vertical_spacing=0.15
                            )
                            
                            fig.add_trace(
                                go.Bar(
                                    x=monthly_stats['month'],
                                    y=monthly_stats['payment_amount_usd'],
                                    name='Total Amount',
                                    marker_color=PROFESSIONAL_COLORS[0]
                                ),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Bar(
                                    x=monthly_stats['month'],
                                    y=monthly_stats['payment_id'],
                                    name='Payment Count',
                                    marker_color=PROFESSIONAL_COLORS[1]
                                ),
                                row=2, col=1
                            )
                            
                            fig.update_layout(height=600, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        if show_debug:
                            st.warning(f"Time analysis error: {str(e)}")
        
        with tab5:
            st.markdown('<div class="sub-header">📊 Statistical Analysis & Hypothesis Testing</div>', unsafe_allow_html=True)
            
            # Show analysis steps
            st.markdown("""
            <div class="analysis-step">
                <strong>📋 Analysis Steps:</strong><br>
                1. Correlation matrix calculation<br>
                2. Statistical hypothesis testing<br>
                3. Distribution analysis<br>
                4. Outlier detection
            </div>
            """, unsafe_allow_html=True)
            
            # Prepare data for statistical analysis
            st.markdown("##### 🔗 Correlation Analysis")
            
            # Create combined dataset for correlation
            analysis_data = customers.copy()
            
            # Merge with loan data if possible
            if 'customer_id' in analysis_data.columns and 'customer_id' in loans.columns:
                loan_features = ['loan_amount_usd', 'interest_rate_pct', 'loan_term_months', 'default_flag']
                available_loan_features = [f for f in loan_features if f in loans.columns]
                
                if available_loan_features:
                    loan_subset = loans[['customer_id'] + available_loan_features].dropna(subset=['customer_id'])
                    analysis_data = analysis_data.merge(loan_subset, on='customer_id', how='inner')
            
            # Select numeric columns
            numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = analysis_data[numeric_cols].corr()
                
                # Visualize correlation matrix
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1,
                    aspect="auto"
                )
                fig.update_xaxes(side="top", tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations
                st.markdown("##### 📈 Top Correlations")
                
                # Flatten correlation matrix
                corr_pairs = corr_matrix.unstack()
                corr_pairs = corr_pairs[corr_pairs != 1]  # Remove self-correlations
                top_correlations = corr_pairs.abs().sort_values(ascending=False).head(10)
                
                # Display top correlations
                top_corr_df = pd.DataFrame({
                    'Variable 1': [idx[0] for idx in top_correlations.index],
                    'Variable 2': [idx[1] for idx in top_correlations.index],
                    'Correlation': [corr_matrix.loc[idx[0], idx[1]] for idx in top_correlations.index],
                    'Absolute Correlation': top_correlations.values
                })
                
                st.dataframe(top_corr_df, use_container_width=True)
                
                # Distribution Analysis
                st.markdown("##### 📊 Distribution Analysis")
                
                selected_variable = st.selectbox(
                    "Select variable for distribution analysis",
                    numeric_cols,
                    key="dist_var_select"
                )
                
                if selected_variable:
                    data_series = analysis_data[selected_variable].dropna()
                    
                    if len(data_series) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram with KDE
                            fig = px.histogram(
                                data_series,
                                nbins=50,
                                marginal="box",
                                title=f"Distribution of {selected_variable}",
                                color_discrete_sequence=[PROFESSIONAL_COLORS[0]]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Statistical summary
                            stats_summary = data_series.describe()
                            skew_val = data_series.skew()
                            kurt_val = data_series.kurtosis()
                            
                            st.markdown(f"""
                            <div class="stat-box">
                                <strong>Statistical Summary for {selected_variable}:</strong><br><br>
                                <strong>Count:</strong> {int(stats_summary['count']):,}<br>
                                <strong>Mean:</strong> {stats_summary['mean']:.2f}<br>
                                <strong>Std Dev:</strong> {stats_summary['std']:.2f}<br>
                                <strong>Min:</strong> {stats_summary['min']:.2f}<br>
                                <strong>25%:</strong> {stats_summary['25%']:.2f}<br>
                                <strong>Median:</strong> {stats_summary['50%']:.2f}<br>
                                <strong>75%:</strong> {stats_summary['75%']:.2f}<br>
                                <strong>Max:</strong> {stats_summary['max']:.2f}<br>
                                <strong>Skewness:</strong> {skew_val:.2f}<br>
                                <strong>Kurtosis:</strong> {kurt_val:.2f}
                            </div>
                            """, unsafe_allow_html=True)
        
        with tab6:
            st.markdown('<div class="sub-header">🔍 Advanced Analytics & Insights</div>', unsafe_allow_html=True)
            
            if enable_ml:
                # Feature Importance Analysis
                st.markdown("##### 🎯 Feature Importance for Default Prediction")
                
                try:
                    # Prepare data for ML analysis
                    ml_data = customers.copy()
                    
                    # Merge with loan data
                    if 'customer_id' in ml_data.columns and 'customer_id' in loans.columns:
                        ml_data = ml_data.merge(
                            loans[['customer_id', 'default_flag']],
                            on='customer_id',
                            how='inner'
                        )
                    
                    # Select numeric features
                    numeric_features = ml_data.select_dtypes(include=[np.number]).columns.tolist()
                    if 'default_flag' in numeric_features:
                        numeric_features.remove('default_flag')
                    if 'customer_id' in numeric_features:
                        numeric_features.remove('customer_id')
                    
                    if len(numeric_features) > 0 and 'default_flag' in ml_data.columns:
                        from sklearn.ensemble import RandomForestClassifier
                        
                        # Prepare data
                        X = ml_data[numeric_features].fillna(ml_data[numeric_features].median())
                        y = ml_data['default_flag']
                        
                        # Train random forest
                        rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                        rf.fit(X, y)
                        
                        # Get feature importances
                        importances = pd.DataFrame({
                            'Feature': numeric_features,
                            'Importance': rf.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Visualize feature importances
                        fig = px.bar(
                            importances.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Features for Default Prediction',
                            color='Importance',
                            color_continuous_scale=PROFESSIONAL_COLORS
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        <div class="insight-highlight">
                            <strong>🔍 Interpretation:</strong><br>
                            Features with higher importance scores have greater predictive power for identifying default risk.
                            Focus risk management efforts on customers with high values in these important features.
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    if show_debug:
                        st.warning(f"ML analysis error: {str(e)}")
                    st.info("ML analysis requires sufficient data and features. Check your dataset completeness.")
            
            # Outlier Detection
            st.markdown("##### 🎯 Outlier Detection Analysis")
            
            if 'loan_amount_usd' in loans.columns:
                loan_amounts = loans['loan_amount_usd'].dropna()
                
                if len(loan_amounts) > 0:
                    # Calculate outliers using IQR method
                    Q1 = loan_amounts.quantile(0.25)
                    Q3 = loan_amounts.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = loan_amounts[(loan_amounts < lower_bound) | (loan_amounts > upper_bound)]
                    outlier_percentage = (len(outliers) / len(loan_amounts)) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Loans", f"{len(loan_amounts):,}")
                    col2.metric("Outliers", f"{len(outliers):,}")
                    col3.metric("Outlier %", f"{outlier_percentage:.2f}%")
                    
                    # Visualize outliers
                    fig = px.box(
                        loan_amounts,
                        title="Loan Amount Outlier Detection",
                        points="outliers"
                    )
                    fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", 
                                annotation_text=f"Upper Bound: ${upper_bound:,.0f}")
                    fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", 
                                annotation_text=f"Lower Bound: ${lower_bound:,.0f}")
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab7:
            st.markdown('<div class="sub-header">📋 Data Quality & Integrity Report</div>', unsafe_allow_html=True)
            
            # Show analysis steps
            st.markdown("""
            <div class="analysis-step">
                <strong>📋 Analysis Steps:</strong><br>
                1. Dataset completeness assessment<br>
                2. Missing value analysis<br>
                3. Data type validation<br>
                4. Cross-dataset consistency check
            </div>
            """, unsafe_allow_html=True)
            
            # Comprehensive data quality analysis
            datasets = {
                'Customers': customers,
                'Loans': loans,
                'Payments': payments
            }
            
            quality_report = []
            
            for name, df in datasets.items():
                st.markdown(f"#### 📄 {name} Dataset Quality")
                
                # Calculate metrics
                total_rows = df.shape[0]
                total_cols = df.shape[1]
                total_cells = total_rows * total_cols
                missing_cells = df.isnull().sum().sum()
                missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", f"{total_rows:,}")
                
                with col2:
                    st.metric("Total Columns", f"{total_cols}")
                
                with col3:
                    st.metric("Missing Values", f"{missing_percentage:.1f}%")
                
                # Store for summary
                quality_report.append({
                    'Dataset': name,
                    'Records': total_rows,
                    'Columns': total_cols,
                    'Missing %': missing_percentage,
                    'Quality Score': max(0, 100 - missing_percentage)
                })
                
                # Detailed analysis in expander
                with st.expander(f"📊 Detailed Analysis for {name}"):
                    # Missing values by column
                    missing_by_col = df.isnull().sum()
                    missing_by_col = missing_by_col[missing_by_col > 0]
                    
                    if len(missing_by_col) > 0:
                        st.markdown("**Missing Values by Column:**")
                        missing_df = pd.DataFrame({
                            'Column': missing_by_col.index,
                            'Missing Count': missing_by_col.values,
                            'Missing %': (missing_by_col.values / total_rows * 100).round(2)
                        }).sort_values('Missing %', ascending=False)
                        
                        st.dataframe(missing_df, use_container_width=True)
                        
                        # Visualize missing values
                        fig = px.bar(
                            missing_df,
                            x='Column',
                            y='Missing %',
                            title=f'Missing Values in {name}',
                            color='Missing %',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("✅ No missing values found in any column!")
                    
                    # Data types summary
                    st.markdown("**Data Types Summary:**")
                    dtype_summary = pd.DataFrame(df.dtypes.value_counts()).reset_index()
                    dtype_summary.columns = ['Data Type', 'Count']
                    st.dataframe(dtype_summary, use_container_width=True)
                
                st.divider()
            
            # Overall Quality Summary
            st.markdown("#### 🏆 Overall Data Quality Summary")
            
            if quality_report:
                quality_df = pd.DataFrame(quality_report)
                
                # Calculate overall score
                overall_score = quality_df['Quality Score'].mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Datasets", f"{len(quality_df)}")
                with col2:
                    st.metric("Total Records", f"{quality_df['Records'].sum():,}")
                with col3:
                    st.metric("Overall Quality Score", f"{overall_score:.1f}/100")
                
                # Quality score visualization
                fig = px.bar(
                    quality_df,
                    x='Dataset',
                    y='Quality Score',
                    title='Data Quality Score by Dataset',
                    color='Quality Score',
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 100]
                )
                fig.add_hline(y=90, line_dash="dash", line_color="green", 
                            annotation_text="Excellent Threshold: 90")
                fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                            annotation_text="Acceptable Threshold: 70")
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations based on quality
                st.markdown("#### 📝 Quality Recommendations")
                
                recommendations = []
                for _, row in quality_df.iterrows():
                    if row['Quality Score'] < 70:
                        recommendations.append(f"⚠️ **{row['Dataset']}**: Needs immediate attention ({row['Quality Score']:.1f}/100)")
                    elif row['Quality Score'] < 90:
                        recommendations.append(f"🔶 **{row['Dataset']}**: Could be improved ({row['Quality Score']:.1f}/100)")
                    else:
                        recommendations.append(f"✅ **{row['Dataset']}**: Excellent quality ({row['Quality Score']:.1f}/100)")
                
                for rec in recommendations:
                    st.write(rec)
        
        # 📊 Final Summary and Export
        st.markdown("---")
        st.markdown('<div class="sub-header">📋 Analysis Summary</div>', unsafe_allow_html=True)
        
        # Generate summary statistics
        summary_stats = {
            'Total Customers': customers.shape[0],
            'Total Loans': loans.shape[0],
            'Total Payments': payments.shape[0],
            'Default Rate': f"{default_rate:.2%}" if 'default_rate' in locals() else "N/A",
            'Average Loan Amount': f"${avg_loan_amount:,.0f}" if 'avg_loan_amount' in locals() else "N/A",
            'Data Quality Score': f"{overall_score:.1f}/100" if 'overall_score' in locals() else "N/A",
            'Analysis Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Display summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Key Metrics")
            for key, value in list(summary_stats.items())[:4]:
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.markdown("#### 📈 Performance Indicators")
            for key, value in list(summary_stats.items())[4:]:
                st.write(f"**{key}:** {value}")
        
        # Export functionality
        st.markdown("---")
        st.markdown("#### 📥 Export Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate Summary Report", type="primary"):
                # Create summary dataframe
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Report CSV",
                    data=csv,
                    file_name=f"eda_summary_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export Visualizations"):
                st.success("Visualization export functionality would be implemented here.")
                st.info("In a production environment, this would generate PNG/PDF reports of all visualizations.")
        
        with col3:
            if st.button("🔄 Reset Analysis"):
                st.rerun()
        
        # Success message
        st.success("✅ Advanced EDA Analysis Completed Successfully!")
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error in EDA analysis: {str(e)}")
        st.markdown("""
        <div class="warning-box">
            <strong>💡 Troubleshooting Steps:</strong><br>
            1. Check if all required datasets are provided<br>
            2. Verify data formats and column names<br>
            3. Ensure data has been properly cleaned<br>
            4. Check for missing or invalid values
        </div>
        """, unsafe_allow_html=True)
        
        # Debug information if enabled
        if 'show_debug' in locals() and show_debug:
            with st.expander("🔧 Debug Information"):
                st.write("**Error Details:**", str(e))
                st.write("**Available Data Keys:**", list(cleaned_data.keys()) if 'cleaned_data' in locals() else "N/A")
                
                for key, df in cleaned_data.items():
                    st.write(f"**{key} dataset shape:**", df.shape)
                    st.write(f"**{key} columns:**", list(df.columns))