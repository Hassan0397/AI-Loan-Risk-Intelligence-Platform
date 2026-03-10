import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

def clean_datasets(raw_data):
    """Automated data cleaning pipeline"""
    cleaned_data = {}
    cleaning_report = {}
    
    # Clean customers data
    customers = raw_data['customers'].copy()
    cust_report = {}
    
    # Handle missing values
    initial_missing = customers.isnull().sum().sum()
    customers['age'].fillna(customers['age'].median(), inplace=True)
    customers['annual_income_usd'].fillna(customers['annual_income_usd'].median(), inplace=True)
    customers['credit_score'].fillna(customers['credit_score'].median(), inplace=True)
    customers['education_level'].fillna('Unknown', inplace=True)
    customers['employment_type'].fillna('Unknown', inplace=True)
    
    # Remove duplicates
    duplicates = customers.duplicated().sum()
    customers.drop_duplicates(inplace=True)
    
    # Handle outliers
    Q1 = customers['annual_income_usd'].quantile(0.25)
    Q3 = customers['annual_income_usd'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    customers = customers[(customers['annual_income_usd'] >= lower_bound) & 
                         (customers['annual_income_usd'] <= upper_bound)]
    
    cust_report['initial_missing'] = initial_missing
    cust_report['duplicates_removed'] = duplicates
    cust_report['final_shape'] = customers.shape
    cleaned_data['customers'] = customers
    
    # Clean loans data
    loans = raw_data['loans'].copy()
    loan_report = {}
    
    loans['loan_start_date'] = pd.to_datetime(loans['loan_start_date'], errors='coerce')
    loans['interest_rate_pct'].fillna(loans['interest_rate_pct'].median(), inplace=True)
    loans['loan_amount_usd'].fillna(loans['loan_amount_usd'].median(), inplace=True)
    
    # Calculate loan-to-income ratio
    merged = pd.merge(loans, customers[['customer_id', 'annual_income_usd']], 
                     on='customer_id', how='left')
    loans['loan_to_income_ratio'] = loans['loan_amount_usd'] / merged['annual_income_usd']
    
    loan_report['date_converted'] = True
    loan_report['missing_filled'] = loans.isnull().sum().sum()
    cleaned_data['loans'] = loans
    
    # Clean payments data
    payments = raw_data['payments'].copy()
    pay_report = {}
    
    payments['payment_date'] = pd.to_datetime(payments['payment_date'], errors='coerce')
    payments['amount_due_usd'].fillna(payments['amount_due_usd'].median(), inplace=True)
    payments['amount_paid_usd'].fillna(payments['amount_due_usd'], inplace=True)
    
    # Calculate payment metrics
    payments['payment_ratio'] = payments['amount_paid_usd'] / payments['amount_due_usd']
    payments['payment_delay'] = (payments['payment_date'] - loans['loan_start_date'].min()).dt.days
    
    pay_report['records_cleaned'] = payments.shape[0]
    cleaned_data['payments'] = payments
    
    # Clean documents data
    documents = raw_data['documents'].copy()
    doc_report = {}
    
    documents['doc_info'].fillna('No information', inplace=True)
    documents.drop_duplicates(subset=['doc_id'], inplace=True)
    
    doc_report['unique_docs'] = documents['doc_id'].nunique()
    cleaned_data['documents'] = documents
    
    # Compile all reports
    cleaning_report = {
        'customers': cust_report,
        'loans': loan_report,
        'payments': pay_report,
        'documents': doc_report
    }
    
    return cleaned_data, cleaning_report