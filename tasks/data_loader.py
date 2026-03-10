import pandas as pd
import streamlit as st
import os

def load_and_display_data():
    """Load all CSV files from data directory"""
    data_files = {
        'customers': 'data/customers.csv',
        'loans': 'data/loans.csv',
        'payments': 'data/payments.csv',
        'documents': 'data/financial_documents_rag.csv'
    }
    
    loaded_data = {}
    
    for name, file_path in data_files.items():
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                loaded_data[name] = df
                st.success(f"✅ {name.capitalize()} loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            else:
                st.error(f"❌ File not found: {file_path}")
                return None
        except Exception as e:
            st.error(f"❌ Error loading {name}: {str(e)}")
            return None
    
    return loaded_data