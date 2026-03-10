import streamlit as st
import sys
import os
import pandas as pd
import base64
from datetime import datetime

# Add tasks directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
tasks_dir = os.path.join(current_dir, 'tasks')
sys.path.insert(0, tasks_dir)
sys.path.insert(0, current_dir)

# Page configuration
st.set_page_config(
    page_title="AI Loan Analyst",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 2rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #10B981;
    }
    .stButton button {
        width: 100%;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .report-download {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #3B82F6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🔍 AI Loan Analyst Navigation")
st.sidebar.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_cleaned' not in st.session_state:
    st.session_state.data_cleaned = False
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_report' not in st.session_state:
    st.session_state.model_report = None
if 'generated_report' not in st.session_state:
    st.session_state.generated_report = None
if 'visualizations' not in st.session_state:
    st.session_state.visualizations = None

# Function to import modules safely
def import_task_modules():
    """Safely import task modules with error handling"""
    modules = {}
    errors = []
    
    module_names = [
        ('data_loader', 'load_and_display_data'),
        ('data_cleaner', 'clean_datasets'),
        ('eda_analysis', 'perform_eda'),
        ('loan_default_predictor', 'predict_loan_defaults'),
        ('shap_explainer', 'explain_with_shap'),
        ('rag_financial', 'financial_rag_system'),
        ('financial_models', 'run_financial_models'),
        ('report_generator', 'generate_comprehensive_report'),
        ('report_generator', 'create_visualizations')
    ]
    
    for module_name, function_name in module_names:
        try:
            module = __import__(f'tasks.{module_name}', fromlist=[function_name])
            func = getattr(module, function_name)
            modules[function_name] = func
        except ImportError as e:
            errors.append(f"{module_name}: {str(e)}")
            modules[function_name] = None
        except AttributeError as e:
            errors.append(f"{module_name}.{function_name}: {str(e)}")
            modules[function_name] = None
    
    return modules, errors

# Import modules
modules, import_errors = import_task_modules()

# Show import errors if any
if import_errors and st.session_state.get('show_import_errors', True):
    with st.expander("⚠️ Import Issues Detected", expanded=True):
        st.error("Some modules failed to import:")
        for error in import_errors:
            st.write(f"- {error}")
        st.info("Make sure all task files exist in the 'tasks' folder.")
        if st.button("Hide Import Warnings"):
            st.session_state.show_import_errors = False
            st.rerun()

# Sidebar navigation options
app_mode = st.sidebar.selectbox(
    "Choose Analysis Module",
    [
        "📊 1. Load & View Raw Data",
        "🧹 2. Automated Data Cleaning",
        "📈 3. Exploratory Data Analysis",
        "🤖 4. Loan Default Prediction",
        "💡 5. SHAP Model Explanations",
        "📚 6. RAG Financial Assistant",
        "💰 7. Financial Models",
        "📋 8. Comprehensive Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("**Features:**\n- Automated Data Cleaning\n- ML Prediction Models\n- SHAP Explainable AI\n- RAG Financial Q&A\n- Statistical Analysis")

# Module 1: Load and View Raw Data
if app_mode == "📊 1. Load & View Raw Data":
    st.markdown('<h2 class="sub-header">📊 Raw Data Exploration</h2>', unsafe_allow_html=True)
    
    load_func = modules.get('load_and_display_data')
    if not load_func:
        st.error("❌ Data loader module not available. Please check tasks/data_loader.py")
    else:
        if st.button("Load All CSV Files", type="primary", use_container_width=True):
            with st.spinner("Loading datasets..."):
                raw_data = load_func()
                if raw_data:
                    st.session_state.data_loaded = True
                    st.session_state.raw_data = raw_data
                    st.success("✅ All datasets loaded successfully!")
                    
                    # Show data preview
                    for name, df in raw_data.items():
                        st.subheader(f"{name.replace('_', ' ').title()}")
                        st.dataframe(df.head(100), use_container_width=True)
                        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                        st.markdown("---")
        elif st.session_state.data_loaded:
            st.info("✅ Data already loaded. You can proceed to Module 2.")
            
            # Show data preview
            for name, df in st.session_state.raw_data.items():
                st.subheader(f"{name.replace('_', ' ').title()}")
                st.dataframe(df.head(100), use_container_width=True)
                st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                st.markdown("---")

# Module 2: Automated Data Cleaning
elif app_mode == "🧹 2. Automated Data Cleaning":
    st.markdown('<h2 class="sub-header">🧹 Automated Data Cleaning</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please load data first in Module 1")
    else:
        clean_func = modules.get('clean_datasets')
        if not clean_func:
            st.error("❌ Data cleaner module not available. Please check tasks/data_cleaner.py")
        else:
            if st.button("Run Automated Cleaning", type="primary", use_container_width=True):
                with st.spinner("Cleaning data with AI-powered algorithms..."):
                    cleaned_data, cleaning_report = clean_func(st.session_state.raw_data)
                    
                    if cleaned_data:
                        st.session_state.data_cleaned = True
                        st.session_state.cleaned_data = cleaned_data
                        
                        # Display cleaning report
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Cleaning Report")
                        for table, report in cleaning_report.items():
                            st.markdown(f"**{table}:**")
                            st.json(report)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show cleaned data preview
                        st.subheader("Cleaned Data Preview")
                        tab1, tab2, tab3, tab4 = st.tabs(["Customers", "Loans", "Payments", "Documents"])
                        
                        with tab1:
                            st.dataframe(cleaned_data['customers'].head(), use_container_width=True)
                        with tab2:
                            st.dataframe(cleaned_data['loans'].head(), use_container_width=True)
                        with tab3:
                            st.dataframe(cleaned_data['payments'].head(), use_container_width=True)
                        with tab4:
                            st.dataframe(cleaned_data['documents'].head(), use_container_width=True)
            elif st.session_state.data_cleaned:
                st.info("✅ Data already cleaned. You can proceed to other modules.")
                
                # Show cleaned data preview
                st.subheader("Cleaned Data Preview")
                tab1, tab2, tab3, tab4 = st.tabs(["Customers", "Loans", "Payments", "Documents"])
                
                with tab1:
                    st.dataframe(st.session_state.cleaned_data['customers'].head(), use_container_width=True)
                with tab2:
                    st.dataframe(st.session_state.cleaned_data['loans'].head(), use_container_width=True)
                with tab3:
                    st.dataframe(st.session_state.cleaned_data['payments'].head(), use_container_width=True)
                with tab4:
                    st.dataframe(st.session_state.cleaned_data['documents'].head(), use_container_width=True)

# Module 3: Exploratory Data Analysis
elif app_mode == "📈 3. Exploratory Data Analysis":
    st.markdown('<h2 class="sub-header">📈 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_cleaned', False):
        st.warning("⚠️ Please clean data first in Module 2")
    else:
        eda_func = modules.get('perform_eda')
        if not eda_func:
            st.error("❌ EDA module not available. Please check tasks/eda_analysis.py")
        else:
            eda_func(st.session_state.cleaned_data)

# Module 4: Loan Default Prediction
elif app_mode == "🤖 4. Loan Default Prediction":
    st.markdown('<h2 class="sub-header">🤖 Loan Default Prediction Model</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_cleaned', False):
        st.warning("⚠️ Please clean data first in Module 2")
    else:
        predict_func = modules.get('predict_loan_defaults')
        if not predict_func:
            st.error("❌ Prediction module not available. Please check tasks/loan_default_predictor.py")
        else:
            results, model_report = predict_func(st.session_state.cleaned_data)
            if results is not None:
                st.session_state.predictions = results
                st.session_state.model_report = model_report

# Module 5: SHAP Explanations
elif app_mode == "💡 5. SHAP Model Explanations":
    st.markdown('<h2 class="sub-header">💡 SHAP Explainable AI</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('predictions', None):
        st.warning("⚠️ Please run default prediction first in Module 4")
    else:
        shap_func = modules.get('explain_with_shap')
        if not shap_func:
            st.error("❌ SHAP module not available. Please check tasks/shap_explainer.py")
        else:
            shap_func(st.session_state.cleaned_data, st.session_state.predictions)

# Module 6: RAG Financial Assistant
elif app_mode == "📚 6. RAG Financial Assistant":
    st.markdown('<h2 class="sub-header">📚 RAG Financial Q&A System</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_cleaned', False):
        st.warning("⚠️ Please clean data first in Module 2")
    else:
        rag_func = modules.get('financial_rag_system')
        if not rag_func:
            st.error("❌ RAG module not available. Please check tasks/rag_financial.py")
            st.info("Make sure the file exists and contains a function named 'financial_rag_system'")
        else:
            rag_func(st.session_state.cleaned_data)

# Module 7: Financial Models
elif app_mode == "💰 7. Financial Models":
    st.markdown('<h2 class="sub-header">💰 Advanced Financial Models</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('data_cleaned', False):
        st.warning("⚠️ Please clean data first in Module 2")
    else:
        finance_func = modules.get('run_financial_models')
        if not finance_func:
            st.error("❌ Financial models module not available. Please check tasks/financial_models.py")
        else:
            finance_func(st.session_state.cleaned_data)

# Module 8: Comprehensive Report
elif app_mode == "📋 8. Comprehensive Report":
    st.markdown('<h2 class="sub-header">📋 Comprehensive Analysis Report</h2>', unsafe_allow_html=True)
    
    # Generate and display comprehensive report
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Cleaned" if st.session_state.get('data_cleaned') else "❌ Raw"
        st.metric("Data Status", status)
    
    with col2:
        if st.session_state.get('model_report'):
            accuracy = st.session_state.model_report.get('accuracy', 0)
            st.metric("Model Accuracy", f"{accuracy:.2%}")
        else:
            st.metric("Model Accuracy", "Not Run")
    
    with col3:
        if st.session_state.get('data_cleaned'):
            if 'customers' in st.session_state.cleaned_data:
                total_customers = len(st.session_state.cleaned_data['customers'])
                st.metric("Total Customers", total_customers)
            else:
                st.metric("Total Customers", "N/A")
        else:
            st.metric("Total Customers", "N/A")
    
    # Display summary
    st.markdown("### 📊 System Status Summary")
    
    status_data = {
        "Module": ["Data Loading", "Data Cleaning", "EDA", "ML Prediction", "SHAP", "RAG", "Financial Models"],
        "Status": [
            "✅ Complete" if st.session_state.data_loaded else "❌ Pending",
            "✅ Complete" if st.session_state.data_cleaned else "❌ Pending",
            "✅ Available" if st.session_state.data_cleaned else "❌ Requires Cleaning",
            "✅ Available" if st.session_state.predictions else "❌ Requires Prediction",
            "✅ Available" if st.session_state.predictions else "❌ Requires Prediction",
            "✅ Available" if st.session_state.data_cleaned else "❌ Requires Cleaning",
            "✅ Available" if st.session_state.data_cleaned else "❌ Requires Cleaning"
        ]
    }
    
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    # Add import for plotly if needed for feature importance visualization
    try:
        import plotly.express as px
        
        if st.session_state.get('model_report') and 'feature_importance' in st.session_state.model_report:
            st.markdown("### 🎯 Feature Importance")
            
            # Create feature importance visualization
            importance_data = st.session_state.model_report.get('feature_importance', [])
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                fig = px.bar(
                    importance_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Most Important Features",
                    color='importance',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass
    
    # Create visualizations if not already created
    if st.session_state.get('data_cleaned') and not st.session_state.get('visualizations'):
        vis_func = modules.get('create_visualizations')
        if vis_func:
            with st.spinner("Creating report visualizations..."):
                st.session_state.visualizations = vis_func(
                    st.session_state.cleaned_data,
                    st.session_state.predictions
                )
    
    # Display visualizations if available
    if st.session_state.get('visualizations'):
        st.markdown("### 📈 Report Visualizations")
        
        # Create tabs for different visualizations
        vis_tabs = st.tabs(list(st.session_state.visualizations.keys()))
        
        for i, (vis_name, fig) in enumerate(st.session_state.visualizations.items()):
            with vis_tabs[i]:
                st.plotly_chart(fig, use_container_width=True)
    
    # Report Generation Section
    st.markdown("---")
    st.markdown("### 📥 Report Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
            if not st.session_state.get('data_cleaned'):
                st.error("⚠️ Please clean data first in Module 2")
            else:
                report_func = modules.get('generate_comprehensive_report')
                if not report_func:
                    st.error("❌ Report generator module not available. Please check tasks/report_generator.py")
                else:
                    with st.spinner("Generating comprehensive report..."):
                        report_result = report_func(
                            st.session_state.raw_data,
                            st.session_state.cleaned_data,
                            st.session_state.predictions,
                            st.session_state.model_report
                        )
                        
                        if report_result['status'] == 'success':
                            st.session_state.generated_report = report_result
                            st.success("✅ Report generated successfully!")
                            
                            # Display download buttons
                            if report_result['pdf']:
                                with open(report_result['pdf'], 'rb') as f:
                                    pdf_data = f.read()
                                    b64_pdf = base64.b64encode(pdf_data).decode()
                                
                                st.markdown('<div class="report-download">', unsafe_allow_html=True)
                                st.markdown("**📄 PDF Report Download**")
                                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="AI_Loan_Analyst_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📥 Download PDF Report</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if report_result['html']:
                                with open(report_result['html'], 'r', encoding='utf-8') as f:
                                    html_data = f.read()
                                    b64_html = base64.b64encode(html_data.encode()).decode()
                                
                                st.markdown('<div class="report-download">', unsafe_allow_html=True)
                                st.markdown("**🌐 Interactive HTML Report**")
                                href = f'<a href="data:text/html;base64,{b64_html}" download="AI_Loan_Analyst_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">📥 Download HTML Report</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"❌ Error generating report: {report_result['message']}")
    
    with col2:
        if st.button("🔄 Regenerate Visualizations", use_container_width=True):
            if st.session_state.get('data_cleaned'):
                vis_func = modules.get('create_visualizations')
                if vis_func:
                    with st.spinner("Regenerating visualizations..."):
                        st.session_state.visualizations = vis_func(
                            st.session_state.cleaned_data,
                            st.session_state.predictions
                        )
                    st.success("✅ Visualizations regenerated!")
                    st.rerun()
    
    # Report Preview Section
    if st.session_state.get('generated_report'):
        st.markdown("### 👁️ Report Preview")
        
        if st.session_state.generated_report['html']:
            with open(st.session_state.generated_report['html'], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Display HTML preview in an iframe
            st.components.v1.html(html_content, height=600, scrolling=True)
    
    # Report Statistics
    st.markdown("### 📊 Report Statistics")
    
    if st.session_state.get('data_cleaned'):
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            total_datasets = len(st.session_state.cleaned_data)
            st.metric("Datasets Analyzed", total_datasets)
        
        with stats_col2:
            total_rows = sum([df.shape[0] for df in st.session_state.cleaned_data.values()])
            st.metric("Total Rows", f"{total_rows:,}")
        
        with stats_col3:
            total_columns = sum([df.shape[1] for df in st.session_state.cleaned_data.values()])
            st.metric("Total Columns", total_columns)
        
        with stats_col4:
            report_status = "✅ Generated" if st.session_state.get('generated_report') else "❌ Pending"
            st.metric("Report Status", report_status)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 AI Loan Analyst Platform | Powered by Streamlit | Free & Open Source</p>
</div>
""", unsafe_allow_html=True)