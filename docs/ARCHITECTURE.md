# Architecture Documentation

## System Architecture

The AI Supply Chain Disruption Predictor follows a modular, microservices-inspired architecture designed for scalability, maintainability, and production deployment.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard  │  REST API (FastAPI)  │  Mobile/Web      │
└──────────────┬──────────────────────┬─────────────────┬─────────┘
               │                      │                 │
               ▼                      ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │   Predictor   │  │  Explainer   │  │ Business Impact   │   │
│  │   Service     │  │   Service    │  │    Simulator      │   │
│  └───────────────┘  └──────────────┘  └───────────────────┘   │
│  ┌───────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │Alert Manager  │  │Data Ingestion│  │   Preprocessor    │   │
│  └───────────────┘  └──────────────┘  └───────────────────┘   │
└──────────────┬──────────────────────────────────────┬──────────┘
               │                                      │
               ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ML Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   XGBoost    │  │Random Forest │  │ Gradient Boosting    │ │
│  │   Model      │  │    Model     │  │      Model           │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │Feature Eng.  │  │Cross-Val     │  │Hyperparameter Tuning │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└──────────────┬──────────────────────────────────────┬──────────┘
               │                                      │
               ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Raw Data   │  │Processed Data│  │     Model Store      │ │
│  │   (CSV/XLS)  │  │   (Pickle)   │  │     (Joblib)         │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Overview

### 1. User Interface Layer

#### Streamlit Dashboard
- **Purpose**: Interactive web application for business users
- **Key Features**:
  - Data upload and visualization
  - Model training interface
  - Prediction workflows
  - Analytics dashboards
  - Business impact reports
  - Alert management
- **Technology**: Streamlit + Plotly
- **Port**: 8501

#### REST API (FastAPI)
- **Purpose**: Programmatic access for integrations
- **Key Features**:
  - Single prediction endpoint
  - Batch prediction endpoint
  - Model management
  - Health checks
- **Technology**: FastAPI + Uvicorn
- **Port**: 8000
- **Documentation**: Auto-generated Swagger UI

### 2. Application Layer

#### Data Ingestion Service
- **File**: `src/data/ingestion.py`
- **Responsibilities**:
  - CSV/Excel file loading
  - Schema validation
  - Data quality checks
  - Format standardization
- **Key Classes**: `DataIngestion`

#### Data Preprocessing Service
- **File**: `src/data/preprocessing.py`
- **Responsibilities**:
  - Missing value imputation
  - Outlier handling
  - Feature scaling
  - Categorical encoding
  - Feature engineering
- **Key Classes**: `DataPreprocessor`, `FeatureEngineering`

#### Model Training Service
- **File**: `src/models/trainer.py`
- **Responsibilities**:
  - Model instantiation
  - Cross-validation
  - Hyperparameter tuning
  - Model evaluation
  - Model versioning and persistence
- **Key Classes**: `ModelTrainer`
- **Supported Models**: XGBoost, Random Forest, Gradient Boosting

#### Prediction Service
- **File**: `src/models/predictor.py`
- **Responsibilities**:
  - Single/batch prediction
  - Risk categorization
  - Probability calibration
  - Confidence intervals
- **Key Classes**: `RiskPredictor`

#### Explainability Service
- **File**: `src/models/explainer.py`
- **Responsibilities**:
  - SHAP value calculation
  - Feature importance
  - Root cause analysis
  - Waterfall plots
- **Key Classes**: `ModelExplainer`
- **Technology**: SHAP library

#### Business Impact Simulator
- **File**: `src/business/impact_simulator.py`
- **Responsibilities**:
  - Revenue loss estimation
  - SLA breach analysis
  - Customer churn prediction
  - ROI calculation
  - Mitigation recommendations
- **Key Classes**: `BusinessImpactSimulator`

#### Alert Manager
- **File**: `src/utils/alerts.py`
- **Responsibilities**:
  - Threshold monitoring
  - Alert generation
  - Report creation (PDF/CSV)
  - Notification routing
- **Key Classes**: `AlertSystem`

### 3. ML Layer

#### Feature Engineering
- Time-based features (day of week, month, season)
- Ratio features (value per km, speed)
- Risk combinations
- Inventory pressure indicators
- High-value flags

#### Model Training Pipeline
```
Raw Data → Validation → Cleaning → Feature Engineering → 
Encoding → Scaling → Train/Test Split → Model Training → 
Evaluation → Hyperparameter Tuning → Model Saving
```

#### Prediction Pipeline
```
Input Data → Validation → Feature Engineering → 
Preprocessing → Encoding → Model Inference → 
Risk Categorization → Explanation → Output
```

### 4. Data Layer

#### Data Storage
- **Raw Data**: `data/raw/` - Original uploads
- **Processed Data**: `data/processed/` - Cleaned and transformed
- **Sample Data**: `data/sample/` - Demo datasets
- **Models**: `models/` - Trained model artifacts
- **Reports**: `reports/` - Generated reports and alerts
- **Logs**: `logs/` - Application logs

#### Model Persistence
- Format: Joblib (scikit-learn compatible)
- Versioning: Timestamp-based
- Metadata: Included in saved files
  - Model type
  - Feature names
  - Performance metrics
  - Training date

## Data Flow

### Training Flow
```
1. User uploads CSV/Excel → Data Ingestion
2. Schema validation and quality checks
3. Data cleaning (missing values, outliers)
4. Feature engineering (15+ features)
5. Encoding categorical variables
6. Train/validation split
7. Model training with cross-validation
8. Hyperparameter tuning (optional)
9. Model evaluation (metrics calculation)
10. Model saving with versioning
11. Results displayed to user
```

### Prediction Flow
```
1. User provides input (single or batch)
2. Data validation and preprocessing
3. Feature engineering (same as training)
4. Model inference
5. Probability calculation
6. Risk categorization (Low/Medium/High/Critical)
7. SHAP explanation generation
8. Business impact calculation
9. Results returned to user
```

### Alert Flow
```
1. Predictions generated
2. Threshold comparison
3. High-risk orders identified
4. Business impact calculated
5. Alert generation
6. Report creation (PDF/CSV)
7. Notification sent (email/Slack)
8. Alert logged
```

## Configuration Management

### Config Files
- **config/config.yaml**: Main configuration
  - App settings
  - Model parameters
  - Risk thresholds
  - Business metrics
  - API settings
  - Data validation rules
  - Logging configuration

- **.env**: Environment variables
  - API keys
  - Database URLs
  - SMTP settings
  - Cloud credentials

### Configuration Loading
```python
from src.utils.config import config

# Access configuration
model_type = config.model.type
risk_threshold = config.risk.high_threshold
api_port = config.api.port
```

## Logging

### Logging Strategy
- **Library**: Loguru
- **Format**: Structured JSON-like logs
- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Rotation**: 500 MB per file
- **Retention**: 10 days
- **Output**: Console + File

### Log Locations
- Application logs: `logs/app.log`
- Error logs: Captured in main log
- API logs: Included in application log

## Error Handling

### Strategies
1. **Validation Errors**: Early detection at input
2. **Processing Errors**: Graceful degradation
3. **Model Errors**: Fallback mechanisms
4. **API Errors**: HTTP status codes + messages
5. **Logging**: All errors logged with context

## Security Considerations

### Current Implementation
- Input validation (schema, types)
- File upload size limits
- CORS configuration
- Environment variable protection
- No hardcoded secrets

### Production Recommendations
- Implement JWT authentication
- Add rate limiting
- Enable HTTPS/TLS
- Use secrets management service
- Add API key authentication
- Implement role-based access control

## Scalability

### Horizontal Scaling
- Stateless application design
- Load balancer compatible
- Multiple instance support

### Vertical Scaling
- Efficient memory usage
- Batch processing optimization
- Lazy loading of models

### Performance Optimization
- Model caching
- Batch prediction support
- Async API endpoints
- Connection pooling ready

## Deployment Architecture

### Docker Deployment
```
┌─────────────────────────────────────────┐
│         Docker Host                      │
│  ┌────────────────┐  ┌────────────────┐ │
│  │   Dashboard    │  │      API       │ │
│  │   Container    │  │   Container    │ │
│  │   (Port 8501)  │  │  (Port 8000)   │ │
│  └────────────────┘  └────────────────┘ │
│           │                  │           │
│           └──────┬───────────┘           │
│                  │                       │
│         ┌────────▼────────┐              │
│         │  Shared Volumes │              │
│         │  (data, models) │              │
│         └─────────────────┘              │
└─────────────────────────────────────────┘
```

### Cloud Deployment (Render/AWS/Azure)
- Container registry for image storage
- Load balancer for traffic distribution
- Persistent volume for data/models
- Auto-scaling based on load
- Health check monitoring
- Logging aggregation

## Monitoring & Observability

### Metrics to Track
- Prediction latency
- Model accuracy over time
- API response times
- Error rates
- Alert frequency
- Data quality metrics

### Health Checks
- `/health` endpoint
- Model availability check
- Data store connectivity
- Memory usage monitoring

## Future Enhancements

### Short-term
- Real-time streaming predictions
- A/B testing framework
- Advanced model ensemble
- Custom alert rules engine

### Long-term
- Multi-tenancy support
- Advanced analytics dashboard
- Mobile application
- Integration marketplace
- AutoML capabilities
- Time series forecasting

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI | Streamlit | Interactive dashboard |
| API | FastAPI | REST endpoints |
| ML | XGBoost, scikit-learn | Predictive models |
| Explainability | SHAP | Model interpretation |
| Visualization | Plotly | Interactive charts |
| Data Processing | Pandas, NumPy | Data manipulation |
| Configuration | Pydantic, YAML | Settings management |
| Logging | Loguru | Structured logging |
| Deployment | Docker, Docker Compose | Containerization |
| Testing | Pytest | Unit/integration tests |

## Maintenance & Operations

### Regular Tasks
- Model retraining (weekly/monthly)
- Data quality monitoring
- Performance metric review
- Log analysis
- Security updates

### Backup Strategy
- Model versioning (automatic)
- Data backups (scheduled)
- Configuration backups
- Log archival

### Disaster Recovery
- Model rollback capability
- Data restoration procedures
- Service restart protocols
- Incident response plan
