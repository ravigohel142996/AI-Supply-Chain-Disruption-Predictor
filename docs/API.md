# API Documentation

## Base URL

- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

Currently, the API does not require authentication. For production deployments, implement JWT token authentication:

```python
# Future implementation
headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN"
}
```

## Endpoints

### 1. Root Endpoint

**GET /**

Returns API information and available endpoints.

**Response:**
```json
{
    "message": "AI Supply Chain Disruption Predictor API",
    "version": "1.0.0",
    "endpoints": [
        "/health",
        "/predict",
        "/batch-predict",
        "/model/info"
    ]
}
```

---

### 2. Health Check

**GET /health**

Check API health and model status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00",
    "model_loaded": true
}
```

**Status Codes:**
- `200`: API is healthy
- `503`: Service unavailable

---

### 3. Single Prediction

**POST /predict**

Make a prediction for a single order.

**Request Body:**
```json
{
    "order_value": 5000.0,
    "shipping_distance": 500.0,
    "lead_time": 5.0,
    "supplier_reliability_score": 0.8,
    "inventory_level": 500.0,
    "demand_forecast": 600.0,
    "weather_risk_index": 0.3,
    "shipping_mode": "Road",
    "supplier_region": "Asia",
    "product_category": "Electronics",
    "season": "Summer",
    "carrier": "Carrier_A"
}
```

**Response:**
```json
{
    "prediction": 0,
    "delay_probability": 0.23,
    "risk_level": 1,
    "risk_category": "Medium"
}
```

**Status Codes:**
- `200`: Successful prediction
- `422`: Validation error
- `500`: Server error
- `503`: Model not loaded

**Example (Python):**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "order_value": 5000,
    "shipping_distance": 500,
    "lead_time": 5,
    "supplier_reliability_score": 0.8,
    "inventory_level": 500,
    "demand_forecast": 600,
    "weather_risk_index": 0.3,
    "shipping_mode": "Road",
    "supplier_region": "Asia",
    "product_category": "Electronics",
    "season": "Summer",
    "carrier": "Carrier_A"
}

response = requests.post(url, json=data)
print(response.json())
```

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "order_value": 5000,
    "shipping_distance": 500,
    "lead_time": 5,
    "supplier_reliability_score": 0.8,
    "inventory_level": 500,
    "demand_forecast": 600,
    "weather_risk_index": 0.3,
    "shipping_mode": "Road",
    "supplier_region": "Asia",
    "product_category": "Electronics",
    "season": "Summer",
    "carrier": "Carrier_A"
  }'
```

---

### 4. Batch Prediction

**POST /batch-predict**

Make predictions for multiple orders from a CSV file.

**Request:**
- Content-Type: `multipart/form-data`
- File parameter: `file`
- Supported formats: CSV

**CSV Format:**
```csv
order_value,shipping_distance,lead_time,supplier_reliability_score,inventory_level,demand_forecast,weather_risk_index,shipping_mode,supplier_region,product_category,season,carrier
5000,500,5,0.8,500,600,0.3,Road,Asia,Electronics,Summer,Carrier_A
7500,800,7,0.6,300,800,0.5,Sea,Europe,Automotive,Winter,Carrier_B
```

**Response:**
```json
{
    "predictions": [
        {
            "prediction": 0,
            "delay_probability": 0.23,
            "risk_level": 1,
            "risk_category": "Medium"
        },
        {
            "prediction": 1,
            "delay_probability": 0.78,
            "risk_level": 2,
            "risk_category": "High"
        }
    ],
    "total_count": 2,
    "high_risk_count": 1,
    "processing_time": 0.15
}
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/batch-predict"
files = {"file": open("orders.csv", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -F "file=@orders.csv"
```

**Status Codes:**
- `200`: Successful batch prediction
- `422`: Invalid file or format
- `500`: Server error
- `503`: Model not loaded

---

### 5. Model Information

**GET /model/info**

Get information about the loaded model.

**Response:**
```json
{
    "model_type": "xgboost",
    "metrics": {
        "accuracy": 0.87,
        "precision": 0.84,
        "recall": 0.82,
        "f1": 0.83,
        "roc_auc": 0.91
    },
    "feature_names": [
        "order_value",
        "shipping_distance",
        "lead_time",
        "..."
    ]
}
```

**Status Codes:**
- `200`: Model information retrieved
- `503`: No model loaded

---

### 6. Load Model

**POST /model/load**

Load a trained model from disk.

**Request Body:**
```json
{
    "model_path": "models/xgboost_v20240115.joblib"
}
```

**Response:**
```json
{
    "status": "success",
    "message": "Model loaded from models/xgboost_v20240115.joblib"
}
```

**Status Codes:**
- `200`: Model loaded successfully
- `404`: Model file not found
- `500`: Error loading model

---

## Data Types

### FeatureInput

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| order_value | float | Yes | > 0 | Order value in dollars |
| shipping_distance | float | Yes | > 0 | Shipping distance in km |
| lead_time | float | Yes | > 0 | Lead time in days |
| supplier_reliability_score | float | Yes | 0-1 | Supplier reliability score |
| inventory_level | float | Yes | > 0 | Current inventory level |
| demand_forecast | float | Yes | > 0 | Forecasted demand |
| weather_risk_index | float | Yes | 0-1 | Weather risk index |
| shipping_mode | string | Yes | - | Air, Sea, Road, or Rail |
| supplier_region | string | Yes | - | Geographic region |
| product_category | string | Yes | - | Product category |
| season | string | Yes | - | Winter, Spring, Summer, Fall |
| carrier | string | Yes | - | Carrier name |

### PredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| prediction | int | Binary prediction (0=No delay, 1=Delay) |
| delay_probability | float | Probability of delay (0-1) |
| risk_level | int | Risk level (0=Low, 1=Medium, 2=High, 3=Critical) |
| risk_category | string | Risk category name |

### BatchPredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| predictions | array | Array of PredictionResponse objects |
| total_count | int | Total number of predictions |
| high_risk_count | int | Number of high/critical risk predictions |
| processing_time | float | Processing time in seconds |

---

## Error Handling

### Error Response Format

```json
{
    "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Status Code | Meaning | Common Causes |
|-------------|---------|---------------|
| 400 | Bad Request | Invalid input data |
| 404 | Not Found | Resource not found |
| 422 | Validation Error | Invalid field values or types |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Model not loaded |

### Example Error Response

```json
{
    "detail": "Model not loaded"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. For production:

- Recommended: 100 requests/minute per IP
- Batch predictions: 10 requests/minute
- Use API gateway for rate limiting

---

## Best Practices

### 1. Input Validation

Always validate input data before sending:

```python
def validate_input(data):
    assert data['order_value'] > 0, "Order value must be positive"
    assert 0 <= data['supplier_reliability_score'] <= 1, "Score must be 0-1"
    assert data['shipping_mode'] in ['Air', 'Sea', 'Road', 'Rail']
    return True
```

### 2. Error Handling

Implement proper error handling:

```python
try:
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

### 3. Batch Processing

For large datasets, process in chunks:

```python
def predict_batch(df, chunk_size=1000):
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df[i:i+chunk_size]
        chunk.to_csv('temp.csv', index=False)
        
        with open('temp.csv', 'rb') as f:
            response = requests.post(url, files={'file': f})
            results.extend(response.json()['predictions'])
    
    return results
```

### 4. Retry Logic

Implement exponential backoff:

```python
import time

def predict_with_retry(data, max_retries=3):
    for i in range(max_retries):
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(2 ** i)  # Exponential backoff
            else:
                raise
```

---

## Interactive Documentation

Once the API server is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide interactive API documentation where you can:
- View all endpoints
- See request/response schemas
- Test API calls directly
- Download OpenAPI specification

---

## SDK Examples

### Python SDK

```python
class SupplyChainClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, order_data):
        """Make single prediction."""
        response = requests.post(
            f"{self.base_url}/predict",
            json=order_data
        )
        response.raise_for_status()
        return response.json()
    
    def batch_predict(self, csv_path):
        """Make batch predictions."""
        with open(csv_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/batch-predict",
                files={'file': f}
            )
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = SupplyChainClient()
result = client.predict({
    "order_value": 5000,
    "shipping_distance": 500,
    # ... other fields
})
print(f"Risk: {result['risk_category']}")
```

### JavaScript SDK

```javascript
class SupplyChainClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async predict(orderData) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(orderData)
        });
        return await response.json();
    }
    
    async batchPredict(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${this.baseUrl}/batch-predict`, {
            method: 'POST',
            body: formData
        });
        return await response.json();
    }
}

// Usage
const client = new SupplyChainClient();
const result = await client.predict({
    order_value: 5000,
    shipping_distance: 500,
    // ... other fields
});
console.log(`Risk: ${result.risk_category}`);
```

---

## Changelog

### Version 1.0.0 (2024-01-15)
- Initial API release
- Single prediction endpoint
- Batch prediction endpoint
- Model management endpoints
- Health check endpoint

---

## Support

For issues or questions:
- GitHub Issues: [Link to repository]
- Email: support@example.com
- Documentation: [Link to docs]
