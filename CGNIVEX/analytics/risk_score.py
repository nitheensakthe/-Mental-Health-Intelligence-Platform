from config import DISCLAIMER

class RiskScoreCalculator:
    """
    Computes a research-oriented Mental Health Risk Indicator based on multi-factor signals:
    - Sentiment polarity
    - Detected emotion
    - Topic domain (e.g. Academic/Work pressure, Loneliness, Sleep)
    - Keyword density & temporal repeat signals
    
    IMPORTANT: This system is a research analytics prototype and does NOT provide medical diagnoses.
    """

    @staticmethod
    def calculate_risk_indicator(sentiment: str, emotion: str, topic: str, text: str = "") -> dict:
        """
        Computes composite numerical risk score (0.0 to 10.0) and assigns category:
        - Low (0.0 - 3.5)
        - Moderate (3.6 - 7.0)
        - High (7.1 - 10.0)
        """
        score = 0.0
        contributing_factors = []

        # 1. Sentiment Signal
        if sentiment == "Negative":
            score += 3.0
            contributing_factors.append("Negative sentiment detected")
        elif sentiment == "Neutral":
            score += 1.0

        # 2. Emotion Signal
        if emotion in ["Anxiety/Stress", "Fear"]:
            score += 3.5
            contributing_factors.append(f"High-stress emotion signal: {emotion}")
        elif emotion == "Sadness":
            score += 3.0
            contributing_factors.append("Sadness / Depressive emotion signal")
        elif emotion == "Anger":
            score += 2.0
            contributing_factors.append("Frustration / Anger signal")

        # 3. Topic Domain Signal
        if topic in ["Academic pressure", "Work pressure", "Depression", "Sleep"]:
            score += 2.5
            contributing_factors.append(f"Elevated concern topic domain: {topic}")
        elif topic in ["Loneliness", "Relationships"]:
            score += 1.5
            contributing_factors.append(f"Social stress topic: {topic}")

        # 4. Text Keyword Signal
        text_lower = text.lower()
        high_risk_words = ["helpless", "dread", "panic", "breaking point", "insomnia", "sleepless", "fail"]
        matched_words = [w for w in high_risk_words if w in text_lower]
        if matched_words:
            score += min(len(matched_words) * 0.5, 2.0)
            contributing_factors.append(f"Stress-indicative language: {', '.join(matched_words)}")

        # Cap score at 10.0
        score = round(min(score, 10.0), 2)

        if score <= 3.5:
            indicator = "Low"
        elif score <= 7.0:
            indicator = "Moderate"
        else:
            indicator = "High"

        return {
            "risk_score": score,
            "risk_indicator": indicator,
            "contributing_factors": contributing_factors,
            "disclaimer": DISCLAIMER
        }
