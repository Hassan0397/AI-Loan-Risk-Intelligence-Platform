import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

def format_currency(value):
    """Format number as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format number as percentage"""
    return f"{value:.1%}"

def calculate_age(birth_date, reference_date=None):
    """Calculate age from birth date"""
    if reference_date is None:
        reference_date = datetime.now()
    return reference_date.year - birth_date.year - \
           ((reference_date.month, reference_date.day) < (birth_date.month, birth_date.day))

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

def create_summary_statistics(df):
    """Create comprehensive summary statistics"""
    summary = {
        'count': df.shape[0],
        'columns': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).shape[1],
        'categorical_columns': df.select_dtypes(include=['object']).shape[1]
    }
    return summary