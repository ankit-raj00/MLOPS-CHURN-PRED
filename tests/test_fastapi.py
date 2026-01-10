import unittest
from fastapi.testclient import TestClient
from app.main import app

class TestFastAPI(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Industry Grade Setup:
        Initialize the TestClient ONCE for the entire class.
        Manually trigger 'lifespan' (startup) events to load the ML model.
        This saves time by loading the model only once, not per test.
        """
        cls.client = TestClient(app)
        cls.client.__enter__() # Triggers app startup (loads model)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up resources (shutdown event).
        """
        cls.client.__exit__(None, None, None)

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    def test_predict_churn(self):
        """
        Test the /predict endpoint against the REAL Production Model.
        """
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
        
        response = self.client.post("/predict", json=payload)
        
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
