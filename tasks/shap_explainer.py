import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from matplotlib.colors import LinearSegmentedColormap

def explain_with_shap(cleaned_data, predictions):
    """SHAP-like explanations without PyTorch dependencies"""
    
    st.subheader("🔍 Model Explainability Analysis")
    
    # Load the best model
    try:
        model = joblib.load('best_loan_default_model.pkl')
    except FileNotFoundError:
        st.warning("Model not found. Please run prediction first.")
        return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Prepare data
    customers = cleaned_data.get('customers')
    loans = cleaned_data.get('loans')
    payments = cleaned_data.get('payments')
    
    if customers is None or loans is None:
        st.error("Missing required data (customers or loans).")
        return
    
    # Get expected features from the model
    try:
        expected_features = model.feature_names_in_
    except AttributeError:
        # For older scikit-learn models, try to get from training data
        expected_features = None
        st.info("Could not get expected features from model. Will infer from data.")
    
    # Feature engineering - MUST MATCH what was used during training
    merged_data = pd.merge(loans, customers[['customer_id', 'age', 'annual_income_usd', 
                                             'credit_score', 'credit_history_years']], 
                          on='customer_id', how='left')
    
    # Add payment-related features if payment data exists
    payment_features_added = False
    if payments is not None and len(payments) > 0:
        # Check what payment columns are available
        payment_cols = payments.columns.tolist()
        
        # Find payment amount column (could have different names)
        payment_amount_col = None
        possible_payment_cols = ['payment_amount_usd', 'payment_amount', 'amount_usd', 
                                'amount', 'payment', 'payment_usd']
        
        for col in possible_payment_cols:
            if col in payment_cols:
                payment_amount_col = col
                break
        
        if payment_amount_col:
            try:
                # Calculate payment statistics
                payment_stats = payments.groupby('loan_id').agg({
                    payment_amount_col: ['mean', 'std', 'sum']
                }).reset_index()
                
                # Flatten column names
                payment_stats.columns = ['loan_id', 'avg_payment', 'payment_std', 'total_paid']
                
                # Merge with loan data
                merged_data = pd.merge(merged_data, payment_stats, on='loan_id', how='left')
                
                # Fill missing payment stats (for loans with no payments)
                merged_data['avg_payment'] = merged_data['avg_payment'].fillna(0)
                merged_data['payment_std'] = merged_data['payment_std'].fillna(0)
                merged_data['total_paid'] = merged_data['total_paid'].fillna(0)
                
                payment_features_added = True
                
            except Exception as e:
                st.warning(f"Could not calculate payment statistics: {e}")
                payment_features_added = False
    
    # Define base features
    base_features = ['loan_amount_usd', 'interest_rate_pct', 'loan_term_months',
                     'age', 'annual_income_usd', 'credit_score', 'credit_history_years']
    
    # Add payment features if they exist
    payment_features = []
    if payment_features_added:
        payment_features = ['avg_payment', 'payment_std', 'total_paid']
    
    all_features = base_features + payment_features
    
    # If we know expected features from model, use them as the definitive list
    if expected_features is not None:
        # Check if model expects payment features that we couldn't add
        expected_payment_feats = [feat for feat in expected_features 
                                 if feat in ['avg_payment', 'payment_std', 'total_paid']]
        
        if expected_payment_feats and not payment_features_added:
            st.warning(f"Model expects payment features {expected_payment_feats} but payment data is missing or incomplete.")
            # We'll still proceed, but these will be filled with 0
        
        all_features = list(expected_features)
    
    # Prepare X with all expected features
    X = pd.DataFrame()
    
    for feature in all_features:
        if feature in merged_data.columns:
            X[feature] = merged_data[feature]
        else:
            # Feature missing from data but expected by model - fill with appropriate default
            if feature in ['avg_payment', 'payment_std', 'total_paid']:
                # Payment features get 0
                X[feature] = 0
            elif feature in ['age', 'credit_history_years', 'loan_term_months']:
                # Count features get median of existing values or 0
                X[feature] = merged_data.get(feature, pd.Series([0]))
            elif feature in ['loan_amount_usd', 'annual_income_usd']:
                # Monetary features get median
                if len(merged_data) > 0:
                    X[feature] = merged_data[feature].median() if feature in merged_data.columns else 0
                else:
                    X[feature] = 0
            elif feature in ['interest_rate_pct', 'credit_score']:
                # Rate/score features get median
                if len(merged_data) > 0:
                    X[feature] = merged_data[feature].median() if feature in merged_data.columns else 0
                else:
                    X[feature] = 0
            else:
                # Unknown feature type
                X[feature] = 0
    
    # Fill NaN values with appropriate values
    for col in X.columns:
        if X[col].isna().any():
            if col in ['avg_payment', 'payment_std', 'total_paid']:
                X[col] = X[col].fillna(0)
            elif col in ['age', 'credit_history_years', 'loan_term_months']:
                X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
            elif col in ['loan_amount_usd', 'annual_income_usd']:
                X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
            elif col in ['interest_rate_pct', 'credit_score']:
                X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
            else:
                X[col] = X[col].fillna(0)
    
    # Ensure the order matches what the model expects
    if expected_features is not None:
        # Reorder columns to match model's expected order
        missing_from_expected = set(X.columns) - set(expected_features)
        extra_in_expected = set(expected_features) - set(X.columns)
        
        if missing_from_expected:
            st.warning(f"Data has extra features not expected by model: {missing_from_expected}")
        
        if extra_in_expected:
            st.warning(f"Model expects features not in data: {extra_in_expected}")
            # Add missing expected features with default values
            for feat in extra_in_expected:
                if feat not in X.columns:
                    X[feat] = 0
        
        # Ensure all expected features are present
        for feat in expected_features:
            if feat not in X.columns:
                X[feat] = 0
        
        # Reorder to match expected order
        X = X[expected_features]
    
    # Display feature summary
    st.info(f"**Using {len(X.columns)} features:** {', '.join(X.columns.tolist())}")
    
    # Check model type
    model_type = str(type(model)).lower()
    
    # OPTION 1: Tree-based models - implement SHAP-like calculations manually
    if any(x in model_type for x in ['randomforest', 'xgboost', 'lgbm', 'gradientboosting', 'decisiontree']):
        st.success("✅ Tree-based model detected - using TreeSHAP approximation")
        explain_tree_model(model, X, X.columns.tolist())
    
    # OPTION 2: Linear models
    elif any(x in model_type for x in ['linear', 'logistic', 'ridge', 'lasso']):
        st.info("📊 Linear model detected - using coefficient-based explanations")
        explain_linear_model(model, X, X.columns.tolist())
    
    # OPTION 3: Neural networks or other models
    else:
        st.warning("⚠️ Model type may not support full explainability")
        explain_generic_model(model, X, X.columns.tolist())

def explain_tree_model(model, X, feature_names):
    """Custom SHAP-like explanations for tree-based models"""
    
    st.subheader("📈 Feature Importance Analysis")
    
    # Ensure X has the exact features the model expects
    X_aligned = align_features_with_model(model, X, feature_names)
    if X_aligned is None:
        return
    
    # Use aligned X and feature names
    X = X_aligned
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    
    # 1. Get feature importances from the model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Ensure the number of importances matches the number of feature names
        if len(importances) != len(feature_names):
            st.warning(f"Model has {len(importances)} feature importances but {len(feature_names)} features in data.")
            # Truncate or pad to match
            min_length = min(len(importances), len(feature_names))
            importances = importances[:min_length]
            feature_names = feature_names[:min_length]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Model Feature Importances')
        ax.invert_yaxis()  # Highest importance at top
        
        # Color bars by importance
        if len(importance_df) > 0:
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(importance_df['Importance'].min(), importance_df['Importance'].max())
            for bar in bars:
                bar.set_color(cmap(norm(bar.get_width())))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Display as table
        st.dataframe(importance_df)
    else:
        st.info("This model doesn't have feature importances attribute.")
    
    # 2. Permutation Importance (SHAP alternative)
    st.subheader("🎯 Permutation Importance")
    
    try:
        # Simple permutation importance implementation
        if hasattr(model, 'predict'):
            y_pred_original = model.predict(X)
        elif hasattr(model, 'predict_proba'):
            y_pred_original = model.predict_proba(X)[:, 1]
        else:
            st.warning("Model doesn't have predict or predict_proba methods")
            return
        
        baseline_score = np.mean(y_pred_original)
        
        permutation_scores = []
        np.random.seed(42)
        
        # Limit to reasonable number of features for performance
        features_to_test = min(10, len(feature_names))
        if len(feature_names) > 0:
            selected_indices = np.random.choice(range(len(feature_names)), 
                                               min(features_to_test, len(feature_names)), 
                                               replace=False)
        else:
            st.warning("No features to test for permutation importance")
            return
        
        with st.spinner("Calculating permutation importance..."):
            for i in selected_indices:
                X_permuted = X.copy()
                # Permute the feature
                col_name = X.columns[i]
                X_permuted[col_name] = np.random.permutation(X_permuted[col_name].values)
                
                if hasattr(model, 'predict'):
                    y_pred_permuted = model.predict(X_permuted)
                else:
                    y_pred_permuted = model.predict_proba(X_permuted)[:, 1]
                
                perm_score = np.mean(y_pred_permuted)
                importance = baseline_score - perm_score
                permutation_scores.append((feature_names[i], importance))
        
        if permutation_scores:
            perm_df = pd.DataFrame(permutation_scores, columns=['Feature', 'Permutation Importance'])
            perm_df = perm_df.sort_values('Permutation Importance', ascending=False)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars2 = ax2.barh(perm_df['Feature'], perm_df['Permutation Importance'])
            ax2.set_xlabel('Decrease in Score When Feature is Permuted')
            ax2.set_title('Permutation Importance')
            ax2.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
    
    except Exception as e:
        st.info(f"Permutation importance skipped: {str(e)}")
    
    # 3. Individual Prediction Explanations
    st.subheader("👤 Individual Prediction Analysis")
    
    if len(X) > 0:
        sample_idx = st.slider("Select sample to analyze", 0, min(50, len(X)-1), 0)
        
        # Get prediction for this sample
        sample = X.iloc[[sample_idx]]
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(sample)[0]
                if len(proba) >= 2:
                    st.write(f"**Prediction probabilities:** Default: {proba[1]:.3f}, No Default: {proba[0]:.3f}")
                else:
                    st.write(f"**Prediction probability:** {proba[0]:.3f}")
            elif hasattr(model, 'predict'):
                pred = model.predict(sample)[0]
                st.write(f"**Predicted class:** {pred}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return
        
        # Show feature contributions (simplified)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Feature Values:**")
            sample_values = pd.DataFrame({
                'Feature': feature_names,
                'Value': sample.values[0]
            })
            st.dataframe(sample_values)
        
        with col2:
            # Create simple feature impact visualization
            st.markdown("**Feature Impact Analysis:**")
            
            if hasattr(model, 'feature_importances_'):
                # Ensure lengths match
                if len(model.feature_importances_) == len(feature_names):
                    # Normalize feature importances for this visualization
                    sample_vals = sample.values[0]
                    mean_vals = X.mean().values
                    
                    # Ensure arrays have same length
                    min_len = min(len(sample_vals), len(mean_vals), len(model.feature_importances_))
                    sample_vals = sample_vals[:min_len]
                    mean_vals = mean_vals[:min_len]
                    importances = model.feature_importances_[:min_len]
                    feature_names_adj = feature_names[:min_len]
                    
                    normalized_impact = importances * (sample_vals - mean_vals)
                    
                    impact_df = pd.DataFrame({
                        'Feature': feature_names_adj,
                        'Impact': normalized_impact
                    }).sort_values('Impact', key=abs, ascending=False)
                    
                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    colors = ['red' if x < 0 else 'green' for x in impact_df['Impact']]
                    ax3.barh(impact_df['Feature'], impact_df['Impact'], color=colors)
                    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
                    ax3.set_xlabel('Feature Impact on Prediction')
                    ax3.set_title(f'Feature Contributions for Sample {sample_idx}')
                    ax3.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)
                else:
                    st.info("Feature importances don't match feature names")
    else:
        st.warning("No data available for individual prediction analysis")
    
    # 4. Partial Dependence Plots (simplified)
    st.subheader("📊 Partial Dependence Analysis")
    
    if len(feature_names) > 0:
        selected_feature = st.selectbox("Select feature for partial dependence plot", feature_names)
        
        try:
            # Create PDP manually
            if selected_feature in X.columns:
                unique_vals = X[selected_feature].unique()
                if len(unique_vals) > 20:
                    # Sample for performance
                    unique_vals = np.percentile(X[selected_feature], np.linspace(0, 100, 20))
                
                pdp_values = []
                
                for val in unique_vals:
                    X_temp = X.copy()
                    X_temp[selected_feature] = val
                    
                    if hasattr(model, 'predict_proba'):
                        preds = model.predict_proba(X_temp)[:, 1]
                    else:
                        preds = model.predict(X_temp)
                    
                    pdp_values.append(preds.mean())
                
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                ax4.plot(unique_vals, pdp_values, 'b-', linewidth=2)
                ax4.scatter(unique_vals, pdp_values, color='red', s=50)
                ax4.set_xlabel(selected_feature)
                ax4.set_ylabel('Average Prediction')
                ax4.set_title(f'Partial Dependence Plot for {selected_feature}')
                ax4.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)
            else:
                st.warning(f"Selected feature '{selected_feature}' not found in data.")
            
        except Exception as e:
            st.info(f"Partial dependence plot not available: {str(e)}")
    else:
        st.warning("No features available for partial dependence analysis")

def explain_linear_model(model, X, feature_names):
    """Explain linear models using coefficients"""
    
    st.subheader("📊 Model Coefficients Analysis")
    
    # Ensure X has the exact features the model expects
    X_aligned = align_features_with_model(model, X, feature_names)
    if X_aligned is None:
        return
    
    X = X_aligned
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        
        # Handle different coefficient shapes
        if len(coefs.shape) > 1:
            coefs = coefs[0]
        
        # Ensure the number of coefficients matches the number of feature names
        if len(coefs) != len(feature_names):
            st.warning(f"Model has {len(coefs)} coefficients but {len(feature_names)} features in data.")
            # Use the minimum length
            min_length = min(len(coefs), len(feature_names))
            coefs = coefs[:min_length]
            feature_names = feature_names[:min_length]
        
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        # Plot coefficients
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if x < 0 else 'green' for x in coef_df['Coefficient']]
        bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Model Coefficients (Red = Negative, Green = Positive)')
        ax.invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width >= 0 else width
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', ha='left' if width >= 0 else 'right',
                   va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Display table
        st.dataframe(coef_df)
        
        # Interpretation
        st.subheader("🎯 Interpretation")
        st.markdown("""
        - **Positive coefficients**: Increase in this feature INCREASES the predicted probability of default
        - **Negative coefficients**: Increase in this feature DECREASES the predicted probability of default
        - **Larger absolute values**: Stronger influence on the prediction
        """)
    
    # Show intercept if available
    if hasattr(model, 'intercept_'):
        intercept = model.intercept_
        if isinstance(intercept, (list, np.ndarray)):
            intercept = intercept[0]
        st.info(f"**Model intercept:** {intercept:.4f}")
    else:
        st.info("No intercept available for this model")

def explain_generic_model(model, X, feature_names):
    """Generic model explanation for unsupported models"""
    
    st.subheader("📋 Model Information")
    
    # Display model type
    st.write(f"**Model type:** {type(model).__name__}")
    
    # Check what methods are available
    available_methods = []
    for method in ['predict', 'predict_proba', 'score', 'feature_importances_', 'coef_']:
        if hasattr(model, method):
            available_methods.append(method)
    
    st.write(f"**Available methods:** {', '.join(available_methods)}")
    
    # Ensure X has the exact features the model expects
    X_aligned = align_features_with_model(model, X, feature_names)
    if X_aligned is None:
        return
    
    X = X_aligned
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    
    # Simple feature correlation with predictions
    st.subheader("📊 Feature Correlation with Predictions")
    
    try:
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'predict'):
            predictions = model.predict(X)
        else:
            st.warning("Model doesn't support prediction methods")
            return
        
        correlations = []
        for feature in feature_names:
            if feature in X.columns and len(X[feature].unique()) > 1:  # Avoid constant features
                try:
                    corr = np.corrcoef(X[feature], predictions)[0, 1]
                    if not np.isnan(corr):
                        correlations.append((feature, corr))
                except:
                    continue
        
        if correlations:
            corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = ['red' if x < 0 else 'green' for x in corr_df['Correlation']]
            bars = ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors)
            
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Correlation with Predictions')
            ax.set_title('Feature Correlation Analysis')
            ax.set_xlim(-1, 1)
            ax.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.info("""
            **Interpretation:**
            - Positive correlation: Higher feature values associated with higher predicted risk
            - Negative correlation: Higher feature values associated with lower predicted risk
            """)
        else:
            st.warning("Could not calculate correlations for any features")
    
    except Exception as e:
        st.warning(f"Could not calculate correlations: {str(e)}")
    
    # Sample predictions
    st.subheader("👤 Sample Predictions")
    
    if len(X) > 0:
        max_idx = min(20, len(X)-1)
        sample_idx = st.slider("Select sample to examine", 0, max_idx, 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Feature Values:**")
            sample_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': X.iloc[sample_idx].values
            })
            st.dataframe(sample_df)
        
        with col2:
            st.markdown("**Prediction:**")
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X.iloc[[sample_idx]])[0]
                    if len(proba) >= 2:
                        st.write(f"**Class probabilities:**")
                        st.write(f"- No Default: {proba[0]:.3f}")
                        st.write(f"- Default: {proba[1]:.3f}")
                        
                        # Create a simple gauge
                        fig, ax = plt.subplots(figsize=(6, 2))
                        ax.barh(['Default Probability'], [proba[1]], color='red' if proba[1] > 0.5 else 'green')
                        ax.set_xlim(0, 1)
                        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
                        ax.set_xlabel('Probability')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.write(f"**Probability:** {proba[0]:.3f}")
                elif hasattr(model, 'predict'):
                    pred = model.predict(X.iloc[[sample_idx]])[0]
                    st.write(f"**Predicted class:** {pred}")
                else:
                    st.warning("Model doesn't support prediction")
            except Exception as e:
                st.error(f"Could not get prediction: {str(e)}")
    else:
        st.warning("No data available for sample predictions")

def align_features_with_model(model, X, feature_names):
    """
    Ensure X has exactly the features the model expects, in the right order
    """
    try:
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            
            # Create a new DataFrame with all expected features
            X_aligned = pd.DataFrame(index=X.index)
            
            for feature in expected_features:
                if feature in X.columns:
                    X_aligned[feature] = X[feature]
                else:
                    # Feature expected by model but not in data
                    st.warning(f"Feature '{feature}' expected by model but not in data. Filling with 0.")
                    X_aligned[feature] = 0
            
            # Ensure correct order
            X_aligned = X_aligned[expected_features]
            return X_aligned
        
        return X
    
    except Exception as e:
        st.error(f"Error aligning features with model: {e}")
        return None

# Alternative: Use SHAP with try-catch to handle PyTorch errors gracefully
def try_shap_explanation(model, X, feature_names):
    """Try to use SHAP if available, fallback to custom explanations"""
    
    try:
        import shap
        
        # Check if we can import without PyTorch errors
        test_import = shap.__version__
        
        # Align features first
        X_aligned = align_features_with_model(model, X, feature_names)
        if X_aligned is None:
            return False, None, None
        
        # Try to create explainer
        if hasattr(model, '_Booster'):  # XGBoost
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'estimators_'):  # Random Forest
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_aligned)
        
        shap_values = explainer(X_aligned)
        
        return True, explainer, shap_values
    except Exception as e:
        return False, None, None