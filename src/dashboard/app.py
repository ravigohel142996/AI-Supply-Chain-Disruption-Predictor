"""Main Streamlit dashboard for AI Supply Chain Disruption Predictor."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.utils.logger import log, app_logger
from src.utils.config import config
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor, FeatureEngineering
from src.data.sample_generator import generate_sample_data
from src.models.trainer import ModelTrainer
from src.models.predictor import RiskPredictor
from src.models.explainer import ModelExplainer
from src.business.impact_simulator import BusinessImpactSimulator
from src.utils.alerts import AlertSystem

# Initialize logger
app_logger.setup()

# Page configuration
st.set_page_config(
    page_title="AI Supply Chain Disruption Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-critical {
        color: #d62728;
        font-weight: bold;
    }
    .risk-high {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffbb00;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'df_train' not in st.session_state:
    st.session_state.df_train = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📦 AI Supply Chain Disruption Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict logistics delays and business risks before they impact customers")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/supply-chain.png", width=150)
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "📊 Data Upload & Training", "🔮 Predictions", 
             "📈 Analytics", "💼 Business Impact", "🚨 Alerts", "📚 About"]
        )
        
        st.markdown("---")
        st.markdown("### Model Status")
        if st.session_state.model_trained:
            st.success("✅ Model Trained")
            if st.session_state.trainer:
                st.info(f"Type: {st.session_state.trainer.model_type}")
                if st.session_state.trainer.metrics:
                    st.metric("Accuracy", f"{st.session_state.trainer.metrics.get('accuracy', 0):.2%}")
                    st.metric("AUC", f"{st.session_state.trainer.metrics.get('roc_auc', 0):.3f}")
        else:
            st.warning("⚠️ No Model Trained")
    
    # Render selected page
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Upload & Training":
        show_data_training()
    elif page == "🔮 Predictions":
        show_predictions()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "💼 Business Impact":
        show_business_impact()
    elif page == "🚨 Alerts":
        show_alerts()
    elif page == "📚 About":
        show_about()


def show_home():
    """Home page with overview."""
    st.header("Welcome to AI Supply Chain Disruption Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Key Features
        - **ML-Powered Predictions**: XGBoost & Random Forest models
        - **Risk Scoring**: Low, Medium, High, Critical categories
        - **Real-time Analytics**: Interactive dashboards
        - **Business Impact**: Revenue, SLA, and churn analysis
        """)
    
    with col2:
        st.markdown("""
        ### 📋 How It Works
        1. **Upload Data**: CSV/Excel files with order information
        2. **Train Model**: Automated ML training pipeline
        3. **Make Predictions**: Batch or single predictions
        4. **Analyze Results**: Visualizations and insights
        """)
    
    with col3:
        st.markdown("""
        ### 🚀 Quick Start
        1. Go to **Data Upload & Training**
        2. Use sample data or upload your own
        3. Train the model
        4. Navigate to **Predictions** to start
        """)
    
    st.markdown("---")
    
    # System metrics
    st.subheader("📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("App Version", config.app.version)
    with col2:
        st.metric("Model Type", config.model.type.upper())
    with col3:
        st.metric("Environment", config.app.environment.upper())
    with col4:
        if st.session_state.df_train is not None:
            st.metric("Training Samples", len(st.session_state.df_train))
        else:
            st.metric("Training Samples", "N/A")


def show_data_training():
    """Data upload and model training page."""
    st.header("📊 Data Upload & Model Training")
    
    tab1, tab2 = st.tabs(["📂 Data Upload", "🤖 Model Training"])
    
    with tab1:
        st.subheader("Upload Training Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            data_source = st.radio(
                "Select Data Source",
                ["Use Sample Data", "Upload CSV/Excel"]
            )
        
        with col2:
            if data_source == "Use Sample Data":
                n_samples = st.slider("Number of Samples", 100, 5000, 1000, 100)
        
        if st.button("Load Data", type="primary"):
            with st.spinner("Loading data..."):
                try:
                    if data_source == "Use Sample Data":
                        df = generate_sample_data(n_samples=n_samples)
                        st.success(f"✅ Generated {len(df)} sample records")
                    else:
                        st.info("Please upload a file below")
                        df = None
                    
                    if df is not None:
                        st.session_state.df_train = df
                        
                        # Show data preview
                        st.subheader("Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", len(df))
                        with col2:
                            st.metric("Features", len(df.columns) - 1)
                        with col3:
                            st.metric("Delay Rate", f"{df['is_delayed'].mean():.1%}")
                        
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        if data_source == "Upload CSV/Excel":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel file with supply chain data"
            )
            
            if uploaded_file is not None:
                try:
                    ingestion = DataIngestion()
                    df = ingestion.upload_handler(uploaded_file)
                    st.session_state.df_train = df
                    st.success(f"✅ Loaded {len(df)} records from {uploaded_file.name}")
                    st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Train ML Model")
        
        if st.session_state.df_train is None:
            st.warning("⚠️ Please load training data first")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["xgboost", "random_forest", "gradient_boosting"],
                help="Select machine learning algorithm"
            )
        
        with col2:
            validation_split = st.checkbox("Use Validation Split", value=True)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes"):
                try:
                    # Preprocess data
                    preprocessor = DataPreprocessor()
                    feature_eng = FeatureEngineering()
                    
                    df = st.session_state.df_train.copy()
                    
                    # Feature engineering
                    df = feature_eng.create_features(df)
                    
                    # Clean data
                    df = preprocessor.clean_data(df)
                    
                    # Select features
                    target = config.features.get('target', 'is_delayed')
                    X = df.drop(columns=[target, 'order_id', 'order_date'], errors='ignore')
                    y = df[target]
                    
                    # Encode categorical variables
                    X = preprocessor.encode_categorical(X, fit=True)
                    
                    # Train model
                    trainer = ModelTrainer(model_type=model_type)
                    metrics = trainer.train(X, y, validation_split=validation_split)
                    
                    # Save to session state
                    st.session_state.trainer = trainer
                    st.session_state.predictor = RiskPredictor(trainer.model)
                    st.session_state.explainer = ModelExplainer(trainer.model, X.sample(min(100, len(X))))
                    st.session_state.model_trained = True
                    st.session_state.df_processed = X
                    
                    # Save model
                    model_path = trainer.save_model()
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.2%}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.2%}")
                    with col4:
                        st.metric("AUC-ROC", f"{metrics['roc_auc']:.3f}")
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    cm = np.array(metrics['confusion_matrix'])
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['No Delay', 'Delay'],
                        y=['No Delay', 'Delay'],
                        text_auto=True,
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"Model saved to: {model_path}")
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    log.error(f"Training error: {str(e)}", exc_info=True)


def show_predictions():
    """Predictions page."""
    st.header("🔮 Make Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Data Upload & Training' page")
        return
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        show_single_prediction()
    
    with tab2:
        show_batch_prediction()


def show_single_prediction():
    """Single prediction interface."""
    st.subheader("Single Order Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        order_value = st.number_input("Order Value ($)", min_value=0.0, value=5000.0)
        shipping_distance = st.number_input("Shipping Distance (km)", min_value=0.0, value=500.0)
        lead_time = st.number_input("Lead Time (days)", min_value=0.0, value=5.0)
        supplier_reliability = st.slider("Supplier Reliability Score", 0.0, 1.0, 0.8)
        inventory_level = st.number_input("Inventory Level", min_value=0.0, value=500.0)
        demand_forecast = st.number_input("Demand Forecast", min_value=0.0, value=600.0)
    
    with col2:
        weather_risk = st.slider("Weather Risk Index", 0.0, 1.0, 0.3)
        shipping_mode = st.selectbox("Shipping Mode", ["Air", "Sea", "Road", "Rail"])
        supplier_region = st.selectbox("Supplier Region", ["Asia", "Europe", "North America", "South America"])
        product_category = st.selectbox("Product Category", ["Electronics", "Automotive", "Textiles", "Food", "Chemicals"])
        season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
        carrier = st.selectbox("Carrier", ["Carrier_A", "Carrier_B", "Carrier_C", "Carrier_D"])
    
    if st.button("Predict", type="primary"):
        try:
            # Create input DataFrame
            input_data = pd.DataFrame([{
                'order_value': order_value,
                'shipping_distance': shipping_distance,
                'lead_time': lead_time,
                'supplier_reliability_score': supplier_reliability,
                'inventory_level': inventory_level,
                'demand_forecast': demand_forecast,
                'weather_risk_index': weather_risk,
                'shipping_mode': shipping_mode,
                'supplier_region': supplier_region,
                'product_category': product_category,
                'season': season,
                'carrier': carrier
            }])
            
            # Feature engineering
            feature_eng = FeatureEngineering()
            input_data = feature_eng.create_features(input_data)
            
            # Preprocess
            preprocessor = DataPreprocessor()
            input_data = preprocessor.encode_categorical(input_data, fit=False)
            
            # Align with training features
            for col in st.session_state.trainer.feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[st.session_state.trainer.feature_names]
            
            # Predict
            result = st.session_state.predictor.predict_with_risk(input_data)
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Delay Probability", f"{result['delay_probability'].iloc[0]:.1%}")
            with col2:
                risk_cat = result['risk_category'].iloc[0]
                risk_class = f"risk-{risk_cat.lower()}"
                st.markdown(f'<p class="{risk_class}">Risk: {risk_cat}</p>', unsafe_allow_html=True)
            with col3:
                prediction = "DELAY EXPECTED" if result['prediction'].iloc[0] == 1 else "ON TIME"
                st.metric("Prediction", prediction)
            
            # Explanation
            if st.session_state.explainer:
                st.subheader("Prediction Explanation")
                try:
                    top_features = st.session_state.explainer.get_top_features(input_data, 0, top_n=5)
                    
                    fig = go.Figure(go.Bar(
                        x=top_features['shap_value'],
                        y=top_features['feature'],
                        orientation='h',
                        marker=dict(
                            color=top_features['shap_value'],
                            colorscale='RdYlGn_r'
                        )
                    ))
                    fig.update_layout(
                        title="Top Contributing Features",
                        xaxis_title="SHAP Value (Impact on Prediction)",
                        yaxis_title="Feature",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info("Explanation not available")
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


def show_batch_prediction():
    """Batch prediction interface."""
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload file for batch prediction",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            ingestion = DataIngestion()
            df = ingestion.upload_handler(uploaded_file)
            
            st.info(f"Loaded {len(df)} records")
            
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Making predictions..."):
                    # Preprocess
                    feature_eng = FeatureEngineering()
                    df_feat = feature_eng.create_features(df)
                    
                    preprocessor = DataPreprocessor()
                    df_feat = preprocessor.clean_data(df_feat)
                    df_feat = preprocessor.encode_categorical(df_feat, fit=False)
                    
                    # Align features
                    for col in st.session_state.trainer.feature_names:
                        if col not in df_feat.columns:
                            df_feat[col] = 0
                    X = df_feat[st.session_state.trainer.feature_names]
                    
                    # Predict
                    results = st.session_state.predictor.predict_with_risk(X)
                    
                    # Combine with original data
                    df_results = pd.concat([df.reset_index(drop=True), results], axis=1)
                    st.session_state.predictions = df_results
                    
                    st.success("✅ Predictions complete!")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Orders", len(df_results))
                    with col2:
                        st.metric("Predicted Delays", (results['prediction'] == 1).sum())
                    with col3:
                        st.metric("High Risk", (results['risk_category'] == 'High').sum())
                    with col4:
                        st.metric("Critical Risk", (results['risk_category'] == 'Critical').sum())
                    
                    # Show results
                    st.subheader("Prediction Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Download button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


def show_analytics():
    """Analytics and visualizations page."""
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ No predictions available. Please make batch predictions first.")
        return
    
    df = st.session_state.predictions
    
    # Summary KPIs
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Orders", len(df))
    with col2:
        delay_rate = (df['prediction'] == 1).sum() / len(df)
        st.metric("Delay Rate", f"{delay_rate:.1%}")
    with col3:
        avg_prob = df['delay_probability'].mean()
        st.metric("Avg Delay Prob", f"{avg_prob:.1%}")
    with col4:
        high_risk = ((df['risk_category'] == 'High') | (df['risk_category'] == 'Critical')).sum()
        st.metric("High/Critical Risk", high_risk)
    with col5:
        if 'order_value' in df.columns:
            at_risk_value = df[df['prediction'] == 1]['order_value'].sum()
            st.metric("At-Risk Value", f"${at_risk_value:,.0f}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        st.subheader("Risk Distribution")
        risk_counts = df['risk_category'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                'Low': '#2ca02c',
                'Medium': '#ffbb00',
                'High': '#ff7f0e',
                'Critical': '#d62728'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Delay probability distribution
        st.subheader("Delay Probability Distribution")
        fig = px.histogram(
            df,
            x='delay_probability',
            nbins=50,
            title="",
            labels={'delay_probability': 'Delay Probability'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # More visualizations
    if 'shipping_mode' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk by Shipping Mode")
            risk_by_mode = df.groupby('shipping_mode')['delay_probability'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=risk_by_mode.index,
                y=risk_by_mode.values,
                labels={'x': 'Shipping Mode', 'y': 'Avg Delay Probability'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'supplier_region' in df.columns:
                st.subheader("Risk by Supplier Region")
                risk_by_region = df.groupby('supplier_region')['delay_probability'].mean().sort_values(ascending=False)
                fig = px.bar(
                    x=risk_by_region.index,
                    y=risk_by_region.values,
                    labels={'x': 'Supplier Region', 'y': 'Avg Delay Probability'}
                )
                st.plotly_chart(fig, use_container_width=True)


def show_business_impact():
    """Business impact analysis page."""
    st.header("💼 Business Impact Analysis")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ No predictions available. Please make batch predictions first.")
        return
    
    df = st.session_state.predictions
    
    if 'order_value' not in df.columns:
        st.error("Order value data not available for business impact analysis")
        return
    
    # Calculate business impact
    simulator = BusinessImpactSimulator()
    
    predictions_df = df[['prediction', 'delay_probability', 'risk_category']].copy()
    order_values = df['order_value']
    
    total_impact = simulator.calculate_total_impact(predictions_df, order_values)
    
    # Display impact summary
    st.subheader("Financial Impact Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💰 Revenue Impact")
        st.metric("Expected Loss", f"${total_impact['revenue']['expected_loss']:,.2f}")
        st.metric("Worst Case", f"${total_impact['revenue']['worst_case_loss']:,.2f}")
        st.metric("Orders at Risk", total_impact['revenue']['at_risk_orders'])
    
    with col2:
        st.markdown("### 📋 SLA Impact")
        st.metric("Expected Penalties", f"${total_impact['sla']['expected_penalty']:,.2f}")
        st.metric("Max Penalties", f"${total_impact['sla']['max_penalty']:,.2f}")
        st.metric("High Risk Orders", total_impact['sla']['high_risk_orders'])
    
    with col3:
        st.markdown("### 👥 Customer Churn Impact")
        st.metric("Expected LTV Loss", f"${total_impact['churn']['expected_ltv_loss']:,.2f}")
        st.metric("High Risk Customers", total_impact['churn']['high_risk_customers'])
        st.metric("Avg Churn Prob", f"{total_impact['churn']['avg_churn_probability']:.1%}")
    
    st.markdown("---")
    
    # Total impact
    st.subheader("Total Business Impact")
    st.error(f"### Total Expected Loss: ${total_impact['overall']['total_expected_loss']:,.2f}")
    
    # Detailed breakdown
    impact_summary = simulator.generate_impact_summary(total_impact)
    st.dataframe(impact_summary, use_container_width=True)
    
    # Recommendations
    st.subheader("Top Risk Orders - Mitigation Recommendations")
    recommendations = simulator.create_mitigation_recommendations(predictions_df, top_n=10)
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"Order #{rec['order_index']} - {rec['risk_category']} Risk ({rec['delay_probability']:.1%})"):
            st.write(f"**Recommendation:** {rec['recommendation']}")


def show_alerts():
    """Alerts and reporting page."""
    st.header("🚨 Alerts & Reports")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ No predictions available. Please make batch predictions first.")
        return
    
    df = st.session_state.predictions
    predictions_df = df[['prediction', 'delay_probability', 'risk_category']].copy()
    
    # Alert system
    alert_system = AlertSystem()
    
    # Generate alerts
    st.subheader("Alert Generation")
    threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.7, 0.05)
    
    if st.button("Generate Alerts"):
        # Update threshold temporarily
        original_threshold = config.alerts.get('high_risk_threshold')
        config.alerts['high_risk_threshold'] = threshold
        
        alerts = alert_system.check_thresholds(predictions_df)
        
        if alerts:
            st.success(f"✅ Generated {len(alerts)} alerts")
            
            # Show alerts
            st.subheader("Active Alerts")
            alerts_df = pd.DataFrame(alerts)
            st.dataframe(alerts_df, use_container_width=True)
            
            # Export alerts
            col1, col2 = st.columns(2)
            
            with col1:
                csv = alerts_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Alerts (CSV)",
                    data=csv,
                    file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Generate report
                if st.button("📄 Generate PDF Report"):
                    with st.spinner("Generating report..."):
                        # Calculate business impact
                        simulator = BusinessImpactSimulator()
                        order_values = df['order_value'] if 'order_value' in df.columns else pd.Series([5000] * len(df))
                        business_impact = simulator.calculate_total_impact(predictions_df, order_values)
                        
                        # Generate report
                        top_risks = predictions_df.nlargest(10, 'delay_probability')
                        report_text = alert_system.generate_alert_report(
                            predictions_df, business_impact, top_risks
                        )
                        
                        # Export to PDF
                        pdf_path = alert_system.export_to_pdf(report_text)
                        
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=f,
                                file_name=f"alert_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
        else:
            st.info("No alerts generated with current threshold")


def show_about():
    """About page."""
    st.header("📚 About This Application")
    
    st.markdown("""
    ## AI Supply Chain Disruption Predictor
    
    ### Overview
    This enterprise-grade platform uses machine learning to predict supply chain disruptions,
    helping businesses prevent delays and minimize losses before they impact customers.
    
    ### Key Capabilities
    
    #### 🎯 Predictive Analytics
    - **Advanced ML Models**: XGBoost, Random Forest, and Gradient Boosting
    - **Risk Scoring**: Multi-level risk categorization (Low, Medium, High, Critical)
    - **Probability Calibration**: Accurate delay probability estimates
    
    #### 📊 Data Processing
    - **Automated Data Ingestion**: Support for CSV and Excel formats
    - **Schema Validation**: Ensure data quality and completeness
    - **Feature Engineering**: 15+ engineered features for better predictions
    - **Missing Value Handling**: Intelligent imputation strategies
    
    #### 🔍 Explainability
    - **SHAP Values**: Understand feature contributions to predictions
    - **Root Cause Analysis**: Identify key risk drivers
    - **Interactive Visualizations**: Explore model decisions
    
    #### 💼 Business Impact
    - **Revenue Loss Estimation**: Calculate financial impact of delays
    - **SLA Breach Analysis**: Assess penalty risks
    - **Customer Churn Prediction**: Estimate retention impact
    - **Mitigation Recommendations**: Actionable insights
    
    #### 🚨 Alert Management
    - **Threshold-based Alerts**: Configurable risk thresholds
    - **PDF Report Generation**: Professional documentation
    - **CSV Export**: Data portability
    - **Email/Slack Integration**: Ready for notifications (config-based)
    
    #### 🌐 API Layer
    - **REST API**: FastAPI-powered endpoints
    - **Batch Prediction**: Process large datasets efficiently
    - **Authentication Ready**: JWT token support
    - **API Documentation**: Auto-generated Swagger docs
    
    ### Technical Stack
    - **Backend**: Python 3.9+
    - **ML Framework**: Scikit-learn, XGBoost
    - **Web Framework**: Streamlit, FastAPI
    - **Visualization**: Plotly, Matplotlib
    - **Explainability**: SHAP
    - **Data Processing**: Pandas, NumPy
    
    ### Deployment
    - **Docker Support**: Containerized deployment
    - **Cloud Ready**: Render, AWS, Azure compatible
    - **Environment Configuration**: Flexible config management
    - **Logging**: Comprehensive logging with Loguru
    
    ### Version Information
    - **Version**: 1.0.0
    - **Release Date**: 2024
    - **License**: MIT
    
    ### Contact & Support
    For questions, issues, or feature requests, please refer to the project repository.
    
    ---
    
    **Built with ❤️ for supply chain professionals**
    """)


if __name__ == "__main__":
    main()
