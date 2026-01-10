from fastapi.testclient import TestClient
from app.main import app
import unittest

class TestFastAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_read_main(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Churn Prediction API", response.json()["message"])

    def test_predict_churn(self):
        # Sample payload matching CustomerData schema
        payload = {
            "tenure": 12,
            "monthly_charges": 70.0,
            "total_charges": 840.0,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "tech_support": "No",
            "online_security": "No",
            "support_calls": 2
        }
        
        response = self.client.post("/predict", json=payload)
        
        # 1. Check Status Code
        if response.status_code != 200:
            print(f"❌ API Error: {response.json()}")
            
        self.assertEqual(response.status_code, 200)
        
        # 2. Check JSON Structure
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        
        # 3. Check Logic Validity
        self.assertIn(data["prediction"], ["Churn", "No Churn"])
        self.assertTrue(0.0 <= data["probability"] <= 1.0)

if __name__ == "__main__":
    unittest.main()
