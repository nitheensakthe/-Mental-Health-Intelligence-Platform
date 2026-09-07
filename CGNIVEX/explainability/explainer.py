class ModelExplainer:
    """
    Inductive Variation Explainer Engine for CGNIVEX.
    Provides feature attribution, keyword importance, graph topology context,
    and signal breakdown for model predictions.
    """

    @staticmethod
    def explain_prediction(text: str, sentiment: str, emotion: str, topic: str, risk_info: dict, graph_context: dict = None) -> dict:
        """
        Generates structured explanation breakdown for a prediction.
        """
        words = text.lower().split()

        # Identify important keywords driving the prediction
        key_stress_words = ["exam", "stress", "overwhelmed", "sleepless", "lonely", "panic", "dread", "burnout", "helpless", "failing", "work"]
        key_pos_words = ["happy", "peace", "joy", "mindfulness", "grateful", "wonderful", "win", "good"]

        important_words = [w for w in words if w in key_stress_words or w in key_pos_words]
        if not important_words:
            important_words = [w for w in words if len(w) > 4][:5]

        # Extract graph topological context if provided
        graph_summary = "Standalone post embedding analysis."
        if graph_context:
            degree = graph_context.get("degree", 0)
            neighbors = graph_context.get("neighbors", [])
            graph_summary = f"Connected to {degree} graph nodes in community (Neighbor Topics: {', '.join(neighbors[:3])})."

        return {
            "prediction_summary": f"{risk_info.get('risk_indicator', 'Moderate')} Mental Health Risk Indicator",
            "important_words": important_words,
            "sentiment_signal": sentiment,
            "detected_emotion": emotion,
            "topic_category": topic,
            "graph_context_summary": graph_summary,
            "contributing_reasons": risk_info.get("contributing_factors", ["General language evaluation."]),
            "research_disclaimer": risk_info.get("disclaimer", "")
        }
