import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Path Setup
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_DIR)

from src.models.predict import CGNIVEXPredictor

st.set_page_config(
    page_title="CGNIVEX — Mental Health Analytics Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E2E;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #313244;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return CGNIVEXPredictor()

@st.cache_data
def load_dataset():
    data_path = os.path.join(BASE_DIR, "data", "mental_health_dataset.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_data
def load_metrics():
    bench_path = os.path.join(BASE_DIR, "models", "benchmark_results.json")
    eval_path = os.path.join(BASE_DIR, "models", "test_evaluation_results.json")
    
    bench = json.load(open(bench_path)) if os.path.exists(bench_path) else {}
    test_eval = json.load(open(eval_path)) if os.path.exists(eval_path) else {}
    return bench, test_eval

def main():
    st.markdown('<div class="main-header">🧠 CGNIVEX Mental Health Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">NLP & Machine Learning Powered Social Media Mental Health Trend & Sentiment Classifier</div>', unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/isometric-headers/100/brain.png", width=70)
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio(
        "Select View",
        ["📊 Dataset Overview", "🧪 Model Diagnostics", "💬 Live Text Predictor", "📁 Batch CSV Processing"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Model**: Logistic Regression (TF-IDF 5k Features)\n\n🎯 **Classes**: 7 Mental Health Categories")

    dataset = load_dataset()
    benchmarks, test_eval = load_metrics()
    
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading prediction model: {e}")
        predictor = None

    # TAB 1: DATASET OVERVIEW
    if menu == "📊 Dataset Overview":
        st.header("📊 Dataset Overview & Class Distribution")
        
        if dataset is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Social Media Samples", f"{len(dataset):,}")
            with col2:
                st.metric("Unique Mental Health Classes", f"{dataset['status'].nunique()}")
            with col3:
                st.metric("Avg Post Length (Words)", f"{dataset['text'].apply(lambda x: len(x.split())).mean():.1f}")
                
            st.markdown("---")
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("Class Distribution")
                class_counts = dataset['status'].value_counts().reset_index()
                class_counts.columns = ['Status', 'Count']
                fig_bar = px.bar(
                    class_counts, x='Status', y='Count', color='Status',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title="Samples per Mental Health Category"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with col_chart2:
                st.subheader("Class Proportion Breakdown")
                fig_pie = px.pie(
                    class_counts, names='Status', values='Count',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("🔍 Dataset Sample Viewer")
            selected_class = st.selectbox("Filter by Category", ["All"] + list(dataset['status'].unique()))
            if selected_class != "All":
                filtered_df = dataset[dataset['status'] == selected_class]
            else:
                filtered_df = dataset
            st.dataframe(filtered_df.head(50), use_container_width=True)
        else:
            st.warning("Dataset not found at `data/mental_health_dataset.csv`.")

    # TAB 2: MODEL DIAGNOSTICS
    elif menu == "🧪 Model Diagnostics":
        st.header("🧪 Model Diagnostics & Benchmark Comparison")
        
        if benchmarks:
            st.subheader("Validation Benchmark Results")
            bench_df = pd.DataFrame(benchmarks).T.reset_index()
            bench_df.columns = ["Model Architecture", "Accuracy", "Macro F1 Score", "Weighted F1 Score"]
            st.table(bench_df.style.highlight_max(axis=0, subset=["Accuracy", "Macro F1 Score"], color="#2B4C7E"))
            
            fig_bench = px.bar(
                bench_df, x="Model Architecture", y=["Accuracy", "Macro F1 Score"],
                barmode="group", title="Validation Benchmarks Across Algorithms",
                color_discrete_sequence=["#6366F1", "#10B981"]
            )
            st.plotly_chart(fig_bench, use_container_width=True)

        if test_eval:
            st.markdown("---")
            st.subheader("🏆 Best Model Holdout Test Performance")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Test Accuracy", f"{test_eval.get('test_accuracy', 0):.4f}")
            with col_m2:
                st.metric("Test Macro F1", f"{test_eval.get('test_macro_f1', 0):.4f}")
            with col_m3:
                st.metric("Test Weighted F1", f"{test_eval.get('test_weighted_f1', 0):.4f}")

            if "confusion_matrix" in test_eval and "labels" in test_eval:
                st.subheader("Confusion Matrix")
                cm = np.array(test_eval["confusion_matrix"])
                labels = test_eval["labels"]
                
                fig_cm = px.imshow(
                    cm, x=labels, y=labels, text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Class", y="Actual True Class", color="Sample Count")
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    # TAB 3: LIVE PREDICTOR
    elif menu == "💬 Live Text Predictor":
        st.header("💬 Real-Time Mental Health & Sentiment Classifier")
        st.markdown("Enter a social media post or user reflection to analyze mental health indicators.")
        
        user_input = st.text_area(
            "Social Media Post / User Text:",
            height=120,
            placeholder="e.g., I'm feeling completely overwhelmed by my work deadlines and can't sleep at night..."
        )
        
        if st.button("🔍 Analyze Text", type="primary"):
            if user_input.strip() and predictor:
                result = predictor.predict_text(user_input)
                
                st.markdown("---")
                st.subheader("Classification Results")
                
                pred_status = result["predicted_status"]
                st.success(f"**Predicted Category**: `{pred_status}`")
                
                if result["probabilities"]:
                    st.subheader("Confidence Distribution Across Categories")
                    probs = result["probabilities"]
                    prob_df = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
                    prob_df = prob_df.sort_values(by="Probability", ascending=False)
                    
                    fig_prob = px.bar(
                        prob_df, x="Probability", y="Category", orientation="h",
                        color="Category", color_discrete_sequence=px.colors.qualitative.Bold,
                        range_x=[0, 1]
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
            elif not user_input.strip():
                st.warning("Please enter some text to analyze.")

    # TAB 4: BATCH CSV PROCESSING
    elif menu == "📁 Batch CSV Processing":
        st.header("📁 Batch CSV Mental Health Analysis")
        st.markdown("Upload a CSV file containing social media posts for automated multi-class classification.")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None and predictor:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:", df_upload.head())
            
            text_cols = list(df_upload.columns)
            selected_col = st.selectbox("Select Column Containing Text:", text_cols)
            
            if st.button("🚀 Process Batch Predictions"):
                with st.spinner("Processing batch predictions..."):
                    res_df = predictor.predict_batch(df_upload, text_column=selected_col)
                    st.success(f"Successfully processed {len(res_df)} rows!")
                    st.dataframe(res_df, use_container_width=True)
                    
                    csv_data = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="cgnivex_batch_predictions.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
