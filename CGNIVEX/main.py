import os
import argparse
import pandas as pd

from config import DISCLAIMER
from scraper import IntelligentScraper
from preprocessing import PreprocessingPipeline
from database import CGNIVEXDatabase
from graph import ContextualGraphBuilder
from models import TransformerEmbeddingModel, GNNRepresentationModel, CGNIVEXHybridModel, BaselineEvaluator
from analytics import SentimentAnalyzer, EmotionAnalyzer, TopicExtractor, TrendAnalyzer, AnomalyDetector, RiskScoreCalculator
from explainability import ModelExplainer
from continual_learning import ContinualLearningEngine

def run_pipeline(mock_mode: bool = True):
    print("=" * 70)
    print("CGNIVEX: Mental Health Trend Analysis & Prediction System")
    print("=" * 70)
    print(DISCLAIMER)
    print("-" * 70)

    # 1. Database Initialization
    db = CGNIVEXDatabase()

    # 2. Intelligent Scraping / Data Ingestion
    scraper = IntelligentScraper(mock_mode=mock_mode)
    raw_df = scraper.fetch_data(limit=100)

    # 3. Preprocessing & Privacy Scrubbing
    cleaner = PreprocessingPipeline()
    processed_df, quality_stats = cleaner.process_dataframe(raw_df)
    print(f"\n[Quality Assurance Stats] {quality_stats}")

    # Save processed posts to SQLite database
    db.save_posts(processed_df)

    # 4. Contextual Graph Construction
    graph_builder = ContextualGraphBuilder()
    hetero_graph = graph_builder.build_graph(processed_df)

    # 5. Extract Feature Embeddings
    print("\n[Embedding Engine] Generating Contextual Text & Graph Structural Representations...")
    transformer_model = TransformerEmbeddingModel(use_transformer=True)
    texts = processed_df["cleaned_text"].tolist()
    text_embeddings = transformer_model.generate_embedding(texts)

    gnn_model = GNNRepresentationModel()
    gnn_embeddings_dict = gnn_model.extract_graph_embeddings(hetero_graph)

    graph_embeddings = []
    for _, row in processed_df.iterrows():
        g_emb = gnn_model.get_post_graph_embedding(hetero_graph, str(row.get("post_id", "")), gnn_embeddings_dict)
        graph_embeddings.append(g_emb)
    graph_embeddings = pd.DataFrame(graph_embeddings).values

    # 6. Fit Proposed Hybrid CGNIVEX Model
    print("\n[Proposed Model] Fine-Tuning CGNIVEX Hybrid MLP Classifier...")
    hybrid_model = CGNIVEXHybridModel()

    emotions = processed_df["label"].tolist() if "label" in processed_df.columns else ["Neutral"] * len(processed_df)
    topics = processed_df["topic"].tolist() if "topic" in processed_df.columns else ["General wellbeing"] * len(processed_df)
    
    # Compute initial risk labels for training target
    risk_labels = []
    for i in range(len(processed_df)):
        r = RiskScoreCalculator.calculate_risk_indicator("Negative" if emotions[i] in ["Anxiety/Stress", "Sadness"] else "Neutral", emotions[i], topics[i])
        risk_labels.append(r["risk_indicator"])

    hybrid_model.fit(text_embeddings, graph_embeddings, emotions, topics, risk_labels, epochs=10)

    # 7. Evaluate Baseline Models vs Proposed CGNIVEX
    print("\n[Model Evaluation] Running Baseline Model Comparisons...")
    eval_results = BaselineEvaluator.evaluate_baselines(text_embeddings[:15], text_embeddings[15:], emotions[:15], emotions[15:])
    for model_name, metrics in eval_results.items():
        print(f" -> {model_name}: Accuracy={metrics['accuracy']}, F1={metrics['f1_score']}")
        db.save_metrics(model_name, metrics)

    # 8. Analytics & Anomaly Detection
    print("\n[Analytics Engine] Executing Sentiment, Topic, and Trend Spikes Analysis...")
    trend_analyzer = TrendAnalyzer()
    trend_df = trend_analyzer.analyze_temporal_trends(processed_df, freq="D")
    
    anomaly_detector = AnomalyDetector()
    trend_anomalies_df = anomaly_detector.detect_spikes(trend_df)

    # 9. Perform Inference & Save Prediction to DB
    sample_text = "I am feeling extremely overwhelmed by my upcoming university final exams and cannot sleep."
    sample_prep = cleaner.process_text(sample_text)
    sample_text_emb = transformer_model.generate_embedding([sample_prep["cleaned_text"]])
    sample_graph_emb = gnn_embeddings_dict.get("post_P1001", list(gnn_embeddings_dict.values())[0]).reshape(1, -1)

    prediction_res = hybrid_model.predict(sample_text_emb, sample_graph_emb)
    sample_sentiment = transformer_model.predict_sentiment(sample_text)
    
    risk_info = RiskScoreCalculator.calculate_risk_indicator(
        sample_sentiment["sentiment"],
        prediction_res["predicted_emotion"],
        prediction_res["predicted_topic"],
        sample_text
    )

    explanation = ModelExplainer.explain_prediction(
        sample_text,
        sample_sentiment["sentiment"],
        prediction_res["predicted_emotion"],
        prediction_res["predicted_topic"],
        risk_info,
        graph_context={"degree": hetero_graph.degree("post_P1001") if hetero_graph.has_node("post_P1001") else 3, "neighbors": ["Academic pressure", "Anxiety/Stress"]}
    )

    db.save_prediction(
        post_id="P_LIVE_001",
        text=sample_text,
        sentiment=sample_sentiment["sentiment"],
        emotion=prediction_res["predicted_emotion"],
        topic=prediction_res["predicted_topic"],
        risk=risk_info["risk_indicator"],
        explanation=explanation
    )

    # 10. Continual Learning Simulation
    cl_engine = ContinualLearningEngine()
    cl_result = cl_engine.process_incremental_batch(hybrid_model, transformer_model, gnn_model, graph_builder, processed_df.tail(5))

    print("\n" + "=" * 70)
    print("CGNIVEX Pipeline Completed Successfully!")
    print(f"Saved Database Record at: {db.db_path}")
    print("Run `streamlit run dashboard/app.py` to open the interactive visualization dashboard.")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGNIVEX Pipeline Execution")
    parser.add_argument("--mode", type=str, default="sample", help="Execution mode: sample or live")
    args = parser.parse_args()
    
    run_pipeline(mock_mode=(args.mode == "sample"))
