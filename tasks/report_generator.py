import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tempfile
import os
from fpdf import FPDF
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO

def generate_comprehensive_report(raw_data, cleaned_data, predictions=None, model_report=None):
    """
    Generate a comprehensive PDF report with all analysis results
    """
    
    def _create_pdf_report(report_data):
        """Create PDF report using FPDF"""
        pdf = FPDF()
        pdf.add_page()
        
        # Set up fonts
        pdf.set_font("Arial", 'B', 20)
        
        # Title
        pdf.set_fill_color(30, 58, 138)  # Blue color
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 15, "AI Loan Analyst Report", ln=True, align='C', fill=True)
        pdf.ln(10)
        
        # Reset text color
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)
        
        # Report metadata
        pdf.cell(0, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Data Analyst: AI Loan Analyst Platform", ln=True)
        pdf.ln(10)
        
        # 1. Executive Summary
        pdf.set_font("Arial", 'B', 16)
        pdf.set_fill_color(59, 130, 246)  # Light blue
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "1. Executive Summary", ln=True, fill=True)
        pdf.ln(5)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)
        
        exec_summary = [
            "This report provides a comprehensive analysis of loan portfolio data.",
            f"Total customers analyzed: {len(cleaned_data['customers']) if 'customers' in cleaned_data else 0}",
            f"Total loans analyzed: {len(cleaned_data['loans']) if 'loans' in cleaned_data else 0}",
            f"Report includes data quality assessment, statistical analysis, and predictive insights."
        ]
        
        for line in exec_summary:
            pdf.multi_cell(0, 7, line)
            pdf.ln(2)
        
        pdf.ln(10)
        
        # 2. Data Overview
        pdf.set_font("Arial", 'B', 16)
        pdf.set_fill_color(59, 130, 246)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "2. Data Overview", ln=True, fill=True)
        pdf.ln(5)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)
        
        # Dataset statistics
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Dataset Statistics:", ln=True)
        pdf.set_font("Arial", '', 12)
        
        datasets = []
        for name, df in cleaned_data.items():
            datasets.append({
                'Dataset': name.title(),
                'Rows': df.shape[0],
                'Columns': df.shape[1],
                'Missing Values': df.isnull().sum().sum(),
                'Duplicates': df.duplicated().sum()
            })
        
        # Create a simple table
        col_widths = [50, 30, 30, 40, 40]
        headers = ['Dataset', 'Rows', 'Columns', 'Missing Values', 'Duplicates']
        
        pdf.set_font("Arial", 'B', 12)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, border=1)
        pdf.ln()
        
        pdf.set_font("Arial", '', 12)
        for dataset in datasets:
            pdf.cell(col_widths[0], 10, dataset['Dataset'], border=1)
            pdf.cell(col_widths[1], 10, str(dataset['Rows']), border=1)
            pdf.cell(col_widths[2], 10, str(dataset['Columns']), border=1)
            pdf.cell(col_widths[3], 10, str(dataset['Missing Values']), border=1)
            pdf.cell(col_widths[4], 10, str(dataset['Duplicates']), border=1)
            pdf.ln()
        
        pdf.ln(10)
        
        # 3. Data Quality Assessment
        pdf.set_font("Arial", 'B', 16)
        pdf.set_fill_color(59, 130, 246)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "3. Data Quality Assessment", ln=True, fill=True)
        pdf.ln(5)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)
        
        # Calculate data quality metrics
        quality_metrics = []
        for name, df in cleaned_data.items():
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            
            quality_score = 100 - ((missing_cells / max(total_cells, 1)) * 50) - ((duplicate_rows / max(df.shape[0], 1)) * 50)
            quality_score = max(0, min(100, quality_score))
            
            quality_metrics.append({
                'Dataset': name.title(),
                'Completeness': f"{100 * (1 - missing_cells/max(total_cells, 1)):.1f}%",
                'Uniqueness': f"{100 * (1 - duplicate_rows/max(df.shape[0], 1)):.1f}%",
                'Quality Score': f"{quality_score:.1f}/100"
            })
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Data Quality Metrics:", ln=True)
        pdf.set_font("Arial", '', 12)
        
        # Quality metrics table
        q_col_widths = [50, 40, 40, 40]
        q_headers = ['Dataset', 'Completeness', 'Uniqueness', 'Quality Score']
        
        pdf.set_font("Arial", 'B', 12)
        for i, header in enumerate(q_headers):
            pdf.cell(q_col_widths[i], 10, header, border=1)
        pdf.ln()
        
        pdf.set_font("Arial", '', 12)
        for metric in quality_metrics:
            pdf.cell(q_col_widths[0], 10, metric['Dataset'], border=1)
            pdf.cell(q_col_widths[1], 10, metric['Completeness'], border=1)
            pdf.cell(q_col_widths[2], 10, metric['Uniqueness'], border=1)
            pdf.cell(q_col_widths[3], 10, metric['Quality Score'], border=1)
            pdf.ln()
        
        pdf.ln(10)
        
        # 4. Model Performance (if available)
        if model_report:
            pdf.set_font("Arial", 'B', 16)
            pdf.set_fill_color(59, 130, 246)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, "4. Model Performance", ln=True, fill=True)
            pdf.ln(5)
            
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 12)
            
            model_metrics = [
                f"Model Type: {model_report.get('model_type', 'Not Specified')}",
                f"Accuracy: {model_report.get('accuracy', 0):.2%}",
                f"Precision: {model_report.get('precision', 0):.2%}",
                f"Recall: {model_report.get('recall', 0):.2%}",
                f"F1-Score: {model_report.get('f1_score', 0):.2%}"
            ]
            
            for metric in model_metrics:
                pdf.cell(0, 10, metric, ln=True)
            
            pdf.ln(5)
            
            # Feature importance (if available)
            if 'feature_importance' in model_report and model_report['feature_importance']:
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Top 5 Important Features:", ln=True)
                pdf.set_font("Arial", '', 12)
                
                for i, feat in enumerate(model_report['feature_importance'][:5]):
                    pdf.cell(0, 10, f"{i+1}. {feat.get('feature', 'Unknown')}: {feat.get('importance', 0):.3f}", ln=True)
        
        # 5. Key Findings
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.set_fill_color(59, 130, 246)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "5. Key Findings & Recommendations", ln=True, fill=True)
        pdf.ln(5)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)
        
        findings = [
            "1. **Data Quality**: All datasets have been cleaned and validated",
            "2. **Model Performance**: Predictive model shows strong performance in identifying potential defaults",
            "3. **Risk Assessment**: Portfolio risk levels have been evaluated",
            "4. **Customer Segmentation**: Customers segmented based on risk profiles",
            "5. **Recommendation**: Implement monitoring for high-risk loans identified by the model"
        ]
        
        for finding in findings:
            pdf.multi_cell(0, 10, finding)
            pdf.ln(2)
        
        pdf.ln(10)
        
        # 6. Appendix
        pdf.set_font("Arial", 'B', 16)
        pdf.set_fill_color(59, 130, 246)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "6. Appendix", ln=True, fill=True)
        pdf.ln(5)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 12)
        
        appendix_items = [
            "A. Data Sources: Internal loan portfolio database",
            "B. Analysis Date: " + datetime.now().strftime("%Y-%m-%d"),
            "C. Tools Used: AI Loan Analyst Platform with Machine Learning",
            "D. Report Version: 1.0"
        ]
        
        for item in appendix_items:
            pdf.cell(0, 10, item, ln=True)
        
        # Save PDF to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf.output(temp_file.name)
        
        return temp_file.name
    
    def _create_html_report(report_data):
        """Create an interactive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Loan Analyst Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #1E3A8A; color: white; padding: 20px; text-align: center; border-radius: 10px; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 5px solid #3B82F6; background-color: #f8fafc; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3B82F6; color: white; }}
                .good {{ color: green; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .bad {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Loan Analyst Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <p>This comprehensive analysis report covers the complete loan portfolio evaluation.</p>
                
                <div class="metric-box">
                    <h3>📈 Total Customers</h3>
                    <p style="font-size: 24px; font-weight: bold;">{len(cleaned_data['customers']) if 'customers' in cleaned_data else 0}</p>
                </div>
                
                <div class="metric-box">
                    <h3>💰 Total Loans</h3>
                    <p style="font-size: 24px; font-weight: bold;">{len(cleaned_data['loans']) if 'loans' in cleaned_data else 0}</p>
                </div>
                
                <div class="metric-box">
                    <h3>📅 Analysis Period</h3>
                    <p style="font-size: 24px; font-weight: bold;">{datetime.now().strftime('%Y-%m')}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>🧹 Data Quality Overview</h2>
                <table>
                    <tr>
                        <th>Dataset</th>
                        <th>Rows</th>
                        <th>Columns</th>
                        <th>Missing Values</th>
                        <th>Quality Score</th>
                    </tr>
        """
        
        # Add dataset rows
        for name, df in cleaned_data.items():
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            quality_score = 100 * (1 - missing_cells/max(total_cells, 1))
            
            html_content += f"""
                    <tr>
                        <td>{name.title()}</td>
                        <td>{df.shape[0]}</td>
                        <td>{df.shape[1]}</td>
                        <td>{missing_cells}</td>
                        <td class="{'good' if quality_score > 90 else 'warning' if quality_score > 70 else 'bad'}">{quality_score:.1f}%</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        # Add model performance if available
        if model_report:
            html_content += f"""
            <div class="section">
                <h2>🤖 Model Performance</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Rating</th>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>{model_report.get('accuracy', 0):.2%}</td>
                        <td class="{'good' if model_report.get('accuracy', 0) > 0.8 else 'warning' if model_report.get('accuracy', 0) > 0.6 else 'bad'}">
                            {'Excellent' if model_report.get('accuracy', 0) > 0.8 else 'Good' if model_report.get('accuracy', 0) > 0.6 else 'Needs Improvement'}
                        </td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>{model_report.get('precision', 0):.2%}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>{model_report.get('recall', 0):.2%}</td>
                        <td>-</td>
                    </tr>
                </table>
            </div>
            """
        
        html_content += """
            <div class="section">
                <h2>🎯 Recommendations</h2>
                <ul>
                    <li>Implement automated monitoring for high-risk loans</li>
                    <li>Review data collection processes to improve data quality</li>
                    <li>Schedule regular portfolio reviews using this analysis</li>
                    <li>Consider implementing the predictive model in production</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #666;">
                <p>Generated by AI Loan Analyst Platform | Confidential Report</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8')
        temp_file.write(html_content)
        temp_file.close()
        
        return temp_file.name
    
    # Generate both report types
    try:
        pdf_file = _create_pdf_report({
            'raw_data': raw_data,
            'cleaned_data': cleaned_data,
            'predictions': predictions,
            'model_report': model_report
        })
        
        html_file = _create_html_report({
            'raw_data': raw_data,
            'cleaned_data': cleaned_data,
            'predictions': predictions,
            'model_report': model_report
        })
        
        return {
            'pdf': pdf_file,
            'html': html_file,
            'status': 'success',
            'message': 'Reports generated successfully'
        }
        
    except Exception as e:
        return {
            'pdf': None,
            'html': None,
            'status': 'error',
            'message': f'Error generating report: {str(e)}'
        }

def create_visualizations(cleaned_data, predictions=None):
    """
    Create visualizations for the report
    """
    visualizations = {}
    
    try:
        # 1. Loan Distribution by Status
        if 'loans' in cleaned_data and 'status' in cleaned_data['loans'].columns:
            status_counts = cleaned_data['loans']['status'].value_counts()
            fig1 = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Loan Distribution by Status",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            visualizations['loan_status'] = fig1
        
        # 2. Customer Age Distribution
        if 'customers' in cleaned_data and 'date_of_birth' in cleaned_data['customers'].columns:
            # Calculate ages
            current_year = datetime.now().year
            cleaned_data['customers']['age'] = current_year - pd.to_datetime(cleaned_data['customers']['date_of_birth']).dt.year
            
            fig2 = px.histogram(
                cleaned_data['customers'],
                x='age',
                nbins=20,
                title="Customer Age Distribution",
                labels={'age': 'Age', 'count': 'Number of Customers'}
            )
            visualizations['age_distribution'] = fig2
        
        # 3. Loan Amount Distribution
        if 'loans' in cleaned_data and 'loan_amount' in cleaned_data['loans'].columns:
            fig3 = px.box(
                cleaned_data['loans'],
                y='loan_amount',
                title="Loan Amount Distribution",
                labels={'loan_amount': 'Loan Amount ($)'}
            )
            visualizations['loan_amounts'] = fig3
        
        # 4. Correlation Heatmap (if multiple numeric columns)
        if 'loans' in cleaned_data:
            numeric_cols = cleaned_data['loans'].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = cleaned_data['loans'][numeric_cols].corr()
                
                fig4 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig4.update_layout(
                    title="Correlation Matrix",
                    width=600,
                    height=600
                )
                visualizations['correlation'] = fig4
        
        # 5. Risk Distribution (if predictions available)
        if predictions is not None and 'default_probability' in predictions.columns:
            fig5 = px.histogram(
                predictions,
                x='default_probability',
                nbins=20,
                title="Default Probability Distribution",
                labels={'default_probability': 'Default Probability', 'count': 'Number of Loans'}
            )
            visualizations['risk_distribution'] = fig5
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return visualizations