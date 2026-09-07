import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.privacy import PrivacyScrubber
from preprocessing.cleaner import PreprocessingPipeline

class TestPreprocessing(unittest.TestCase):

    def test_privacy_scrubbing(self):
        scrubber = PrivacyScrubber()
        raw = "Reach out at test@example.com or call 555-123-4567 or visit https://example.com"
        scrubbed = scrubber.scrub_text(raw)
        self.assertIn("[EMAIL_REMOVED]", scrubbed)
        self.assertIn("[PHONE_REMOVED]", scrubbed)
        self.assertIn("[URL_REMOVED]", scrubbed)

    def test_user_id_anonymization(self):
        scrubber = PrivacyScrubber()
        anon_id = scrubber.anonymize_user_id("user_john")
        self.assertTrue(anon_id.startswith("usr_"))
        self.assertNotEqual(anon_id, "user_john")

    def test_preprocessing_pipeline(self):
        pipeline = PreprocessingPipeline()
        res = pipeline.process_text("I am feeling 😊 and very stressed out!")
        self.assertIn("happy", res["cleaned_text"])
        self.assertIn("stressed", res["tokens"])

if __name__ == "__main__":
    unittest.main()
