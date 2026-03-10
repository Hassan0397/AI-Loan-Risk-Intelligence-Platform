import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import xgboost as xgb
import lightgbm as lgb
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def predict_loan_defaults(cleaned_data):
    """
    🤖 Advanced Machine Learning Model for Loan Default Prediction
    -------------------------------------------------------------
    This function:
    1. Merges customer, loan, and payment data
    2. Performs feature engineering and preprocessing
    3. Trains multiple ML models
    4. Evaluates and compares model performance
    5. Provides interpretable results and visualizations
    """
    
    try:
        # ============================================
        # 📊 STEP 1: DATA AVAILABILITY CHECK
        # ============================================
        st.subheader("📦 Data Verification")
        st.markdown("""
        **What we're doing:** Checking if all required datasets are available.
        **Why it's important:** We need customer, loan, and payment data to build accurate predictions.
        """)
        
        required_datasets = ['customers', 'loans', 'payments']
        missing_datasets = [ds for ds in required_datasets if ds not in cleaned_data]
        
        if missing_datasets:
            st.error(f"❌ **Missing Required Datasets:** {', '.join(missing_datasets)}")
            st.info("Please ensure all three datasets are uploaded and cleaned.")
            return None, None
        
        # Load datasets
        customers = cleaned_data['customers'].copy()
        loans = cleaned_data['loans'].copy()
        payments = cleaned_data['payments'].copy()
        
        # Display dataset statistics
        st.subheader("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Customers", f"{len(customers):,}", 
                     help="Total number of unique customers")
        with col2:
            st.metric("💰 Loans", f"{len(loans):,}", 
                     help="Total number of loan records")
        with col3:
            st.metric("💳 Payments", f"{len(payments):,}", 
                     help="Total payment transactions")
        
        # Display data preview
        with st.expander("🔍 View Data Samples"):
            tab1, tab2, tab3 = st.tabs(["Customers", "Loans", "Payments"])
            with tab1:
                st.dataframe(customers.head(), use_container_width=True)
            with tab2:
                st.dataframe(loans.head(), use_container_width=True)
            with tab3:
                st.dataframe(payments.head(), use_container_width=True)
        
        # ============================================
        # 🔍 STEP 2: COLUMN IDENTIFICATION
        # ============================================
        st.subheader("🔍 Column Identification")
        st.markdown("""
        **What we're doing:** Automatically detecting column names in your datasets.
        **Why it's important:** Different datasets may use different naming conventions for IDs and features.
        """)
        
        def find_column(df, possible_names, dataset_name=""):
            """Find a column from list of possible names with informative output"""
            for name in possible_names:
                if name in df.columns:
                    st.success(f"✅ **{dataset_name}:** Found '{name}' as {possible_names[0]} column")
                    return name
            
            # If not found, show available columns
            with st.expander(f"⚠️ Available columns in {dataset_name}"):
                st.write(df.columns.tolist())
            return None
        
        # Find ID columns for each dataset
        customer_id_col = find_column(customers, 
                                     ['customer_id', 'Customer_ID', 'cust_id', 'Cust_ID', 'customerID', 'id', 'CustomerID'],
                                     "Customers Dataset")
        
        loan_customer_id_col = find_column(loans,
                                          ['customer_id', 'Customer_ID', 'cust_id', 'Cust_ID', 'customerID', 'client_id'],
                                          "Loans Dataset")
        
        loan_id_col = find_column(loans,
                                 ['loan_id', 'Loan_ID', 'loanID', 'LoanID', 'id', 'loan_number'],
                                 "Loans Dataset")
        
        payment_loan_id_col = find_column(payments,
                                         ['loan_id', 'Loan_ID', 'loanID', 'LoanID', 'loan_id', 'loan_number'],
                                         "Payments Dataset")
        
        # Validate required columns
        if not all([customer_id_col, loan_customer_id_col, loan_id_col]):
            st.error("❌ **Critical Error:** Could not find all required ID columns.")
            return None, None
        
        # ============================================
        # 🔧 STEP 3: FEATURE ENGINEERING
        # ============================================
        st.subheader("🔧 Feature Engineering")
        st.markdown("""
        **What we're doing:** Creating new features from raw data to improve model performance.
        **Why it's important:** Better features lead to more accurate predictions and insights.
        """)
        
        progress_bar = st.progress(0, text="Starting feature engineering...")
        
        # 1. Customer Features
        progress_bar.progress(10, text="Processing customer features...")
        cust_features = customers.copy()
        
        # 2. Loan Features
        progress_bar.progress(30, text="Processing loan features...")
        loan_features = loans.copy()
        
        # 3. Payment Behavior Features
        progress_bar.progress(50, text="Analyzing payment patterns...")
        
        payment_stats = None
        if payment_loan_id_col:
            # Find amount column for payments
            amount_candidates = ['amount_paid_usd', 'amount_paid', 'amount', 'payment_amount', 'paid_amount']
            amount_col = None
            for candidate in amount_candidates:
                if candidate in payments.columns:
                    amount_col = candidate
                    break
            
            if amount_col:
                # Calculate comprehensive payment statistics
                payment_stats = payments.groupby(payment_loan_id_col).agg({
                    amount_col: ['sum', 'mean', 'std', 'count', 'min', 'max']
                }).reset_index()
                
                # Flatten multi-level columns
                payment_stats.columns = [f'{col[0]}_{col[1]}' if col[1] else f'{col[0]}' 
                                        for col in payment_stats.columns.values]
                
                # Find and rename columns
                id_column = payment_stats.columns[0]  # First column is the ID
                payment_stats = payment_stats.rename(columns={
                    id_column: 'loan_id',
                    f'{amount_col}_sum': 'total_paid',
                    f'{amount_col}_mean': 'avg_payment',
                    f'{amount_col}_std': 'payment_std',
                    f'{amount_col}_count': 'payment_count',
                    f'{amount_col}_min': 'min_payment',
                    f'{amount_col}_max': 'max_payment'
                })
            else:
                st.warning("⚠️ Could not find amount column in payments. Using basic payment features.")
                payment_stats = pd.DataFrame({'loan_id': loans[loan_id_col].unique()})
                payment_stats['payment_count'] = payments.groupby(payment_loan_id_col).size().values
        else:
            st.info("ℹ️ Payment data not available or missing loan ID. Proceeding without payment features.")
        
        # ============================================
        # 🔗 STEP 4: DATA MERGING
        # ============================================
        st.subheader("🔗 Data Integration")
        st.markdown("""
        **What we're doing:** Combining customer, loan, and payment data into a single dataset.
        **Why it's important:** Machine learning models need all relevant information in one place.
        """)
        
        progress_bar.progress(70, text="Merging datasets...")
        
        # Step 1: Merge loans with customers
        merged_data = pd.merge(
            loans, 
            customers, 
            left_on=loan_customer_id_col,
            right_on=customer_id_col,
            how='left',
            suffixes=('_loan', '_customer'),
            indicator=True
        )
        
        # Check merge success
        merge_stats = merged_data['_merge'].value_counts()
        st.info(f"""
        **Merge Statistics:**
        - Successfully matched: {merge_stats.get('both', 0):,} loans
        - Loans without customer data: {merge_stats.get('left_only', 0):,}
        - Customers without loans: {merge_stats.get('right_only', 0):,}
        """)
        
        # Remove merge indicator
        merged_data = merged_data.drop('_merge', axis=1)
        
        # Step 2: Merge with payment statistics if available
        if payment_stats is not None:
            merged_data = pd.merge(
                merged_data,
                payment_stats,
                left_on=loan_id_col,
                right_on='loan_id',
                how='left',
                suffixes=('', '_payment')
            )
        
        progress_bar.progress(90, text="Finalizing merged dataset...")
        
        # Display merged dataset info
        st.success(f"✅ **Merged Dataset Created:** {len(merged_data):,} records with {len(merged_data.columns):,} features")
        
        with st.expander("📋 View Merged Dataset Structure"):
            st.dataframe(merged_data.head(), use_container_width=True)
            st.metric("Total Features", len(merged_data.columns))
        
        progress_bar.progress(100, text="Feature engineering complete!")
        
        # ============================================
        # 🎯 STEP 5: TARGET VARIABLE PREPARATION
        # ============================================
        st.subheader("🎯 Target Variable Preparation")
        st.markdown("""
        **What we're doing:** Identifying and converting the loan default status to a binary target.
        **Why it's important:** Our models need a clear binary label (0=Non-default, 1=Default) to learn from.
        """)
        
        # Find target column
        target_candidates = ['default_flag', 'default', 'is_default', 'status', 'loan_status', 
                            'default_status', 'loan_default', 'performance_status']
        target_col = None
        
        for candidate in target_candidates:
            if candidate in merged_data.columns:
                target_col = candidate
                break
        
        if not target_col:
            st.error("❌ **Critical Error:** Could not find default status column.")
            st.info(f"**Available columns:** {merged_data.columns.tolist()}")
            return None, None
        
        st.success(f"✅ **Target Column Found:** '{target_col}'")
        
        # Show target distribution before processing
        st.markdown("#### 📈 Original Target Distribution")
        original_counts = merged_data[target_col].value_counts()
        fig_target_orig = px.bar(
            x=original_counts.index.astype(str),
            y=original_counts.values,
            title="Original Target Values Distribution",
            labels={'x': 'Target Value', 'y': 'Count'},
            color=original_counts.values,
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_target_orig, use_container_width=True)
        
        # Convert target to binary
        def create_binary_target(value):
            """Convert various default indicators to binary (0/1)"""
            if pd.isna(value):
                return 0
            
            str_val = str(value).lower()
            
            # Default indicators
            default_indicators = [
                'default', 'delinquent', 'charged off', 'charged_off', 
                'bad', 'write off', 'non-performing', 'np', '1', 'yes', 
                'true', 'failed', 'defaulted'
            ]
            
            # Non-default indicators
            non_default_indicators = [
                'current', 'active', 'paid', 'completed', 'good',
                'performing', '0', 'no', 'false', 'ok', 'closed'
            ]
            
            # Check for default
            if any(indicator in str_val for indicator in default_indicators):
                return 1
            # Check for non-default
            elif any(indicator in str_val for indicator in non_default_indicators):
                return 0
            # Try numeric conversion
            else:
                try:
                    num_val = float(value)
                    return 1 if num_val > 0.5 else 0
                except:
                    return 0  # Default to non-default if unclear
        
        merged_data['target'] = merged_data[target_col].apply(create_binary_target)
        
        # Show target distribution after conversion
        st.markdown("#### 🔄 Converted Binary Target")
        target_counts = merged_data['target'].value_counts()
        target_percentages = merged_data['target'].value_counts(normalize=True) * 100
        
        fig_target = make_subplots(
            rows=1, cols=2,
            subplot_titles=('📊 Count Distribution', '📈 Percentage Distribution'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Bar chart
        fig_target.add_trace(
            go.Bar(
                x=['Non-Default (0)', 'Default (1)'],
                y=[target_counts.get(0, 0), target_counts.get(1, 0)],
                marker_color=['green', 'red'],
                text=[f"{target_counts.get(0, 0):,}", f"{target_counts.get(1, 0):,}"],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig_target.add_trace(
            go.Pie(
                labels=['Non-Default', 'Default'],
                values=[target_percentages.get(0, 0), target_percentages.get(1, 0)],
                marker=dict(colors=['green', 'red']),
                hole=0.4
            ),
            row=1, col=2
        )
        
        fig_target.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_target, use_container_width=True)
        
        # Class imbalance warning
        if target_percentages.get(1, 0) < 5:
            st.warning("⚠️ **Class Imbalance Alert:** Default cases are less than 5%. Consider using resampling techniques.")
        
        # ============================================
        # 🏗️ STEP 6: FEATURE PREPARATION
        # ============================================
        st.subheader("🏗️ Feature Preparation")
        st.markdown("""
        **What we're doing:** Selecting and preprocessing features for model training.
        **Why it's important:** Clean, relevant features improve model accuracy and prevent overfitting.
        """)
        
        # Identify columns to exclude
        exclude_cols = [
            'target', target_col,
            customer_id_col, loan_customer_id_col, loan_id_col,
            'loan_id'  # Added payment merge ID
        ]
        
        # Remove any duplicate columns and ensure they exist
        exclude_cols = [col for col in exclude_cols if col in merged_data.columns]
        
        # Separate numeric and categorical features
        numeric_features = merged_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = merged_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove excluded columns
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        categorical_features = [col for col in categorical_features if col not in exclude_cols]
        
        st.info(f"""
        **Feature Summary:**
        - Numeric features: {len(numeric_features):,}
        - Categorical features: {len(categorical_features):,}
        - Total features before encoding: {len(numeric_features) + len(categorical_features):,}
        """)
        
        # Handle missing values in numeric features
        if numeric_features:
            missing_numeric = merged_data[numeric_features].isnull().sum().sum()
            if missing_numeric > 0:
                st.info(f"🔄 Filling {missing_numeric:,} missing values in numeric features...")
                merged_data[numeric_features] = merged_data[numeric_features].fillna(
                    merged_data[numeric_features].median()
                )
        
        # Encode categorical features
        if categorical_features:
            st.info(f"🔄 Encoding {len(categorical_features):,} categorical features...")
            merged_data_encoded = pd.get_dummies(
                merged_data, 
                columns=categorical_features, 
                drop_first=True,
                prefix_sep='_'
            )
            
            # Update numeric features list
            all_features = merged_data_encoded.select_dtypes(include=[np.number]).columns.tolist()
            features = [col for col in all_features if col not in exclude_cols and col != 'target']
        else:
            merged_data_encoded = merged_data.copy()
            features = numeric_features
        
        # Final feature count
        st.success(f"✅ **Final Feature Set:** {len(features):,} features after preprocessing")
        
        # Display feature importance preview
        with st.expander("📋 View Top 15 Features"):
            feature_df = pd.DataFrame({
                'Feature': features,
                'Type': ['Categorical' if '_' in f else 'Numeric' for f in features]
            })
            st.dataframe(feature_df.head(15), use_container_width=True)
        
        # Prepare X and y
        X = merged_data_encoded[features]
        y = merged_data_encoded['target']
        
        # ============================================
        # 🤖 STEP 7: MODEL TRAINING
        # ============================================
        st.subheader("🤖 Model Training")
        st.markdown("""
        **What we're doing:** Training multiple machine learning models to predict loan defaults.
        **Why it's important:** Different models have different strengths; comparing them helps us choose the best one.
        """)
        
        # Split data with stratification (preserves class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        st.info(f"""
        **Data Split:**
        - Training samples: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)
        - Testing samples: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)
        - Features: {len(features):,}
        """)
        
        # Define models with optimized parameters
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'color': 'blue'
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'color': 'green'
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=7,
                    learning_rate=0.1,
                    num_leaves=31,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'color': 'orange'
            }
        }
        
        # Train and evaluate models
        results = {}
        best_model_name = None
        best_roc_auc = 0
        
        progress_text = st.empty()
        model_progress = st.progress(0, text="Starting model training...")
        
        for i, (name, model_info) in enumerate(models.items()):
            progress_text.text(f"Training {name}...")
            model_progress.progress((i) / len(models), text=f"Training {name}...")
            
            try:
                # Train model
                model = model_info['model']
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'color': model_info['color']
                }
                
                # Update best model
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_name = name
                    
                st.success(f"✅ {name} trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error training {name}: {str(e)}")
                continue
        
        model_progress.progress(1.0, text="Model training complete!")
        progress_text.empty()
        
        if not results:
            st.error("❌ No models were trained successfully.")
            return None, None
        
        # ============================================
        # 📊 STEP 8: MODEL EVALUATION
        # ============================================
        st.subheader("📊 Model Performance Comparison")
        st.markdown("""
        **What we're doing:** Comparing the performance of all trained models.
        **Why it's important:** Helps us select the best model for deployment.
        """)
        
        # Create comparison dataframe
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results],
            'Precision': [results[m]['precision'] for m in results],
            'Recall': [results[m]['recall'] for m in results],
            'F1-Score': [results[m]['f1'] for m in results],
            'ROC-AUC': [results[m]['roc_auc'] for m in results]
        })
        
        # Display metrics table
        st.markdown("#### 📋 Performance Metrics Table")
        st.dataframe(
            metrics_df.style
            .background_gradient(subset=['Accuracy', 'ROC-AUC'], cmap='YlOrRd')
            .format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'ROC-AUC': '{:.3f}'
            }),
            use_container_width=True
        )
        
        # Visual comparison
        st.markdown("#### 📈 Visual Comparison")
        
        # Bar chart for metrics comparison
        fig_comparison = go.Figure()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        for metric in metrics_to_plot:
            fig_comparison.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Model'],
                y=metrics_df[metric],
                text=[f'{v:.2%}' if metric != 'ROC-AUC' else f'{v:.3f}' for v in metrics_df[metric]],
                textposition='auto',
            ))
        
        fig_comparison.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1.05]),
            height=500
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Highlight best model
        st.success(f"🏆 **Best Model:** {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.3f})")
        
        # ============================================
        # 🎯 STEP 9: DETAILED ANALYSIS OF BEST MODEL
        # ============================================
        st.subheader(f"🎯 Detailed Analysis: {best_model_name}")
        
        best_model_results = results[best_model_name]
        
        # Confusion Matrix
        st.markdown("#### 📊 Confusion Matrix")
        st.markdown("""
        **What we're seeing:** How well the model distinguishes between defaults and non-defaults.
        **Why it's important:** Shows the types of errors the model makes (false positives vs false negatives).
        """)
        
        cm = confusion_matrix(y_test, best_model_results['predictions'])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Non-Default', 'Default'],
            y=['Non-Default', 'Default'],
            title=f'{best_model_name} - Confusion Matrix',
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC Curve
        st.markdown("#### 📈 ROC Curve")
        st.markdown("""
        **What we're seeing:** The trade-off between true positive rate and false positive rate.
        **Why it's important:** Shows how well the model separates the classes at different thresholds.
        """)
        
        fpr, tpr, thresholds = roc_curve(y_test, best_model_results['probabilities'])
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            line=dict(color=results[best_model_name]['color'], width=3),
            name=f'{best_model_name} (AUC = {best_model_results["roc_auc"]:.3f})'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Random Classifier'
        ))
        
        fig_roc.update_layout(
            title=f'ROC Curve - {best_model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # ============================================
        # 🔍 STEP 10: FEATURE IMPORTANCE
        # ============================================
        st.subheader("🔍 Feature Importance Analysis")
        st.markdown("""
        **What we're seeing:** Which features have the most impact on predictions.
        **Why it's important:** Helps understand what drives loan defaults and validates model logic.
        """)
        
        if hasattr(best_model_results['model'], 'feature_importances_'):
            # Get feature importances
            importances = best_model_results['model'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(20)
            
            # Create visualization
            fig_importance = px.bar(
                feature_importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Top 20 Feature Importances - {best_model_name}',
                color='Importance',
                color_continuous_scale='viridis'
            )
            
            fig_importance.update_layout(
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Display table
            with st.expander("📋 View Complete Feature Importance Table"):
                st.dataframe(
                    feature_importance_df.style
                    .background_gradient(subset=['Importance'], cmap='viridis')
                    .format({'Importance': '{:.4f}'}),
                    use_container_width=True
                )
        
        # ============================================
        # 💾 STEP 11: MODEL SAVING & DEPLOYMENT
        # ============================================
        st.subheader("💾 Model Deployment")
        st.markdown("""
        **What we're doing:** Saving the best model for future use.
        **Why it's important:** Allows you to use the trained model on new data without retraining.
        """)
        
        try:
            # Save the best model
            model_filename = f'best_loan_default_model_{best_model_name.lower().replace(" ", "_")}.pkl'
            joblib.dump(best_model_results['model'], model_filename)
            
            # Save feature names
            feature_filename = 'model_features.pkl'
            joblib.dump(features, feature_filename)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ **Model saved as:** `{model_filename}`")
                st.download_button(
                    label="📥 Download Model",
                    data=open(model_filename, 'rb'),
                    file_name=model_filename,
                    mime="application/octet-stream"
                )
            
            with col2:
                st.success(f"✅ **Features saved as:** `{feature_filename}`")
                st.download_button(
                    label="📥 Download Features",
                    data=open(feature_filename, 'rb'),
                    file_name=feature_filename,
                    mime="application/octet-stream"
                )
            
            # Save performance report
            report_data = {
                'best_model': best_model_name,
                'performance_metrics': {
                    'accuracy': best_model_results['accuracy'],
                    'precision': best_model_results['precision'],
                    'recall': best_model_results['recall'],
                    'f1_score': best_model_results['f1'],
                    'roc_auc': best_model_results['roc_auc']
                },
                'dataset_info': {
                    'total_samples': len(merged_data),
                    'training_samples': len(X_train),
                    'testing_samples': len(X_test),
                    'features_count': len(features),
                    'class_distribution': {
                        'non_default': int(y.value_counts().get(0, 0)),
                        'default': int(y.value_counts().get(1, 0))
                    }
                },
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            report_df = pd.DataFrame([report_data['performance_metrics']])
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Performance Report",
                data=csv,
                file_name="model_performance_report.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.warning(f"⚠️ Could not save model files: {str(e)}")
        
        # ============================================
        # 📋 STEP 12: FINAL SUMMARY
        # ============================================
        st.subheader("📋 Executive Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.metric("🏆 Best Model", best_model_name)
            st.metric("📈 ROC-AUC Score", f"{best_model_results['roc_auc']:.3f}")
            st.metric("🎯 Accuracy", f"{best_model_results['accuracy']:.2%}")
        
        with summary_col2:
            st.metric("📊 Data Coverage", f"{len(merged_data):,} records")
            st.metric("🔧 Features Used", f"{len(features):,}")
            st.metric("⚖️ Default Rate", f"{target_percentages.get(1, 0):.1f}%")
        
        # Recommendations based on model performance
        st.markdown("#### 💡 Recommendations")
        
        if best_model_results['roc_auc'] > 0.8:
            st.success("✅ **Excellent Performance:** Model shows strong predictive power. Ready for deployment.")
        elif best_model_results['roc_auc'] > 0.7:
            st.info("📊 **Good Performance:** Model is reliable but could benefit from more data or feature engineering.")
        else:
            st.warning("⚠️ **Moderate Performance:** Consider collecting more data, engineering additional features, or trying different algorithms.")
        
        # Return results
        return results, {
            'best_model': best_model_name,
            'best_model_object': best_model_results['model'],
            'accuracy': best_model_results['accuracy'],
            'roc_auc': best_model_results['roc_auc'],
            'precision': best_model_results['precision'],
            'recall': best_model_results['recall'],
            'f1_score': best_model_results['f1'],
            'features_used': len(features),
            'sample_size': len(merged_data),
            'default_rate': target_percentages.get(1, 0)
        }
        
    except Exception as e:
        st.error(f"❌ **Critical Error in Loan Default Prediction:** {str(e)}")
        
        # Provide debugging information
        with st.expander("🔧 Debug Information"):
            import traceback
            st.code(traceback.format_exc())
        
        return None, None