import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

class TestFastAPI(unittest.TestCase):
    
    def test_root(self):
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    def test_predict_churn(self):
        """
        Test the /predict endpoint against the REAL Production Model.
        Note: This test will fail (503) if no model is tagged as @Production in MLflow.
        """
        # Create Client (Triggers startup -> loads REAL model from MLflow)
        with TestClient(app) as client:
            payload = {
                "tenure": 12,
                "monthly_charges": 70.5,
                "total_charges": 846.0,
                "contract": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "tech_support": "No",
                "online_security": "No",
                "support_calls": 2
            }
            
            response = client.post("/predict", json=payload)
            
            # Check for Service Unavailable (Missing Model)
            if response.status_code == 503:
                self.fail("API returned 503. Cause: No Model found in MLflow Registry with alias '@Production'.")
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertIn("prediction", data)
            self.assertIn("probability", data)
            self.assertIsInstance(data["probability"], float)

if __name__ == "__main__":
    unittest.main()
