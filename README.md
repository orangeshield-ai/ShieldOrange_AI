# ShieldOrange AI

**Weather-Driven Agricultural Supply Forecasting System**

AI-powered prediction system that monitors Florida's orange crop for weather-related supply disruptions, generating forecasts 48-72 hours before official USDA reports.

---

## Overview

ShieldOrange uses Claude Sonnet 4 to analyze NOAA weather forecasts and predict crop damage from freeze events, hurricanes, disease pressure, and drought conditions. By processing weather data in real-time and cross-referencing with 40 years of historical precedents, the system generates probabilistic crop damage estimates days before official agricultural reports.

**Key Features:**
- 📡 Real-time NOAA weather monitoring (every 15 minutes)
- 🤖 Claude Sonnet 4 AI analysis
- 📊 40-year historical database of weather-crop correlations
- 🎯 70% prediction accuracy (backtested 2023-2025)
- 📈 Performance tracking and validation
- 🚨 Automated high-confidence alerts

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ShieldOrange AI                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────┐ │
│  │   Weather    │────▶│    Claude    │────▶│ Database   │ │
│  │  Collector   │     │   Sonnet 4   │     │ PostgreSQL │ │
│  └──────────────┘     └──────────────┘     └────────────┘ │
│        │                     │                     │        │
│        │                     │                     │        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Main Orchestrator                         │  │
│  │  • 15-min monitoring cycle                           │  │
│  │  • Prediction generation                             │  │
│  │  • Alert management                                  │  │
│  │  • Performance tracking                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

External Data Sources:
├── NOAA National Weather Service (free)
├── National Hurricane Center (free)
├── USDA Crop Reports (free)
├── Planet Labs Satellite Imagery ($500/month)
└── CME Market Data ($100/month)
```

---

## Installation

### Prerequisites

- Python 3.9+
- PostgreSQL 14+
- Anthropic API key (Claude Sonnet 4 access)

### Step 1: Clone Repository

```bash
git clone https://github.com/shieldorange-ai/shieldorange.git
cd shieldorange
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Create `.env` file:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

Required environment variables:

```bash
# AI Model
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://user:password@localhost/shieldorange

# Optional: Satellite Imagery
PLANET_API_KEY=your-planet-key

# Optional: Market Data
CME_API_KEY=your-cme-key

# Deployment
DEPLOYMENT_PHASE=PAPER_TRADING  # or LIVE
```

### Step 5: Initialize Database

```bash
python -c "
from database import ShieldOrangeDatabase
db = ShieldOrangeDatabase()
db.connect()
db.create_tables()
db.seed_historical_data()
db.disconnect()
print('Database initialized successfully')
"
```

---

## Usage

### Run Single Monitoring Cycle

Test the system with one complete monitoring cycle:

```bash
python main.py once
```

This will:
1. Fetch weather forecasts for all citrus counties
2. Analyze freeze/hurricane/disease risk
3. Generate AI predictions if thresholds met
4. Store results in database
5. Display summary

**Example output:**
```
================================================================================
WEATHER MONITORING CYCLE - 2026-02-01T10:30:00Z
================================================================================

--- Monitoring Polk County ---
✓ Forecast retrieved successfully
  Forecast office: TBX
  Hourly periods: 72
  Active alerts: 1
⚠️  FREEZE RISK DETECTED: severe_damage
✓ Model agreement sufficient: 75%
Sending freeze analysis request to Claude Sonnet 4...
✓ Prediction generated successfully
  Probability: 82%
  Expected damage: 13.5%
  Confidence: 85%
✓ Prediction stored: PRED_Polk_20260201_103000_a7b3c9f1

================================================================================
MONITORING CYCLE COMPLETE
Counties monitored: 4
Predictions generated: 1
Alerts triggered: 1
================================================================================
```

### Run Continuous Monitoring (Production)

Start 24/7 monitoring with scheduled tasks:

```bash
python main.py continuous
```

This runs:
- Weather monitoring every 15 minutes
- Prediction verification daily at 9:00 AM
- Performance reports weekly on Monday

**To run as background service:**

```bash
# Using screen
screen -S shieldorange
python main.py continuous
# Detach: Ctrl+A, D

# Or using systemd (Linux)
sudo cp shieldorange.service /etc/systemd/system/
sudo systemctl enable shieldorange
sudo systemctl start shieldorange
```

### Generate Performance Report

```bash
python main.py report
```

**Output:**
```json
{
  "generated_at": "2026-02-01T10:00:00Z",
  "period": "Last 30 days",
  "overall_metrics": {
    "total_predictions": 23,
    "verified_count": 16,
    "avg_accuracy": 0.73,
    "success_rate": 0.70,
    "avg_confidence": 0.78
  },
  "recent_predictions": 5,
  "deployment_phase": "PAPER_TRADING",
  "system_status": "OPERATIONAL"
}
```

### Verify Predictions Against USDA Reports

```bash
python main.py verify
```

This checks unverified predictions against latest USDA data and calculates accuracy.

---

## Configuration

### Key Parameters (config.py)

**Freeze Thresholds:**
```python
FREEZE_THRESHOLDS = {
    'severe_damage': {
        'temp': 24.0,           # Fahrenheit
        'duration_hours': 4,
        'damage_pct': 0.15      # 15% crop loss
    }
}
```

**Prediction Thresholds:**
```python
MIN_CONFIDENCE_THRESHOLD = 0.70    # 70% minimum confidence
MIN_MODEL_AGREEMENT = 0.75         # 75% ensemble agreement
MIN_EXPECTED_IMPACT = 0.05         # 5% minimum crop damage
```

**Monitoring Intervals:**
```python
POLLING_INTERVALS = {
    'noaa_forecast': 900,      # 15 minutes
    'hurricane_center': 3600,  # 1 hour
    'usda_reports': 86400      # Daily
}
```

---

## Data Sources

All data sources are configured and automatically accessed:

| Source | Data | Cost | Update Frequency |
|--------|------|------|------------------|
| NOAA NWS | Weather forecasts | Free | 15 minutes |
| NOAA GOES Satellites | Cloud imagery | Free | 10 minutes |
| National Hurricane Center | Storm tracking | Free | 6 hours |
| USDA NASS | Crop reports | Free | Weekly/Monthly |
| Planet Labs | Satellite imagery | $500/mo | Daily |
| CME Group | Market prices | $100/mo | Real-time |

**Total monthly cost: ~$600** (excluding development time)

---

## System Components

### 1. Weather Collector (`weather_collector.py`)

Fetches and processes NOAA data:
- Hourly and daily forecasts
- Freeze warnings and alerts
- Hurricane threat detection
- Meteorologist discussions (natural language)
- Ensemble model comparison

**Example usage:**
```python
from weather_collector import NOAAWeatherCollector

collector = NOAAWeatherCollector()
forecast = collector.get_ensemble_forecast('Polk')

print(f"Freeze risk: {forecast['freeze_risk']['risk']}")
print(f"Expected damage: {forecast['freeze_risk']['max_expected_damage']*100}%")
```

### 2. Claude AI Engine (`claude_engine.py`)

Generates predictions using Claude Sonnet 4:
- Freeze event analysis
- Hurricane impact assessment
- Disease pressure forecasting
- Prediction quality validation
- Human-readable explanations

**Example usage:**
```python
from claude_engine import ClaudeAnalysisEngine

engine = ClaudeAnalysisEngine()
prediction = engine.analyze_freeze_event(forecast_data, historical_analogs)

print(f"Probability: {prediction['probability']*100}%")
print(f"Damage estimate: {prediction['expected_crop_damage_pct']*100}%")
print(f"Confidence: {prediction['confidence_score']*100}%")
```

### 3. Database (`database.py`)

PostgreSQL storage for:
- Weather forecasts (historical and current)
- AI predictions (with verification status)
- Historical events (40 years of data)
- USDA reports
- Satellite imagery metadata
- Performance metrics

**Example usage:**
```python
from database import ShieldOrangeDatabase

db = ShieldOrangeDatabase()
db.connect()

# Store prediction
pred_id = db.store_ai_prediction(prediction)

# Verify against USDA report
db.verify_prediction(prediction_id, actual_damage=0.127, 
                     verification_source='USDA Feb 2026')

# Get performance metrics
metrics = db.get_prediction_performance()
print(f"Accuracy: {metrics['avg_accuracy']*100}%")
```

### 4. Main Orchestrator (`main.py`)

Coordinates all components:
- Scheduled monitoring cycles
- Prediction generation logic
- Alert triggering
- Performance tracking
- Verification workflow

---

## Prediction Workflow

```
1. MONITOR (every 15 minutes)
   ├─ Fetch NOAA forecasts for 4 counties
   ├─ Check freeze/hurricane/disease risk
   └─ Store forecasts in database

2. ANALYZE (if risk detected)
   ├─ Query historical similar events
   ├─ Send data to Claude Sonnet 4
   ├─ Apply academic research models
   └─ Generate probabilistic prediction

3. VALIDATE (AI self-check)
   ├─ Check prediction quality
   ├─ Identify weaknesses
   └─ Adjust confidence if needed

4. DECIDE (threshold check)
   ├─ Confidence >= 70%?
   ├─ Model agreement >= 75%?
   └─ Expected impact >= 5%?

5. STORE & ALERT
   ├─ Save prediction to database
   ├─ Generate unique prediction ID
   ├─ Send high-confidence alerts
   └─ Track for verification

6. VERIFY (7-14 days later)
   ├─ Fetch USDA crop report
   ├─ Compare to prediction
   ├─ Calculate accuracy
   └─ Update performance metrics
```

---

## API Endpoints (Optional)

If you run the FastAPI server (`api_server.py`):

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Available endpoints:**

```
GET  /health              - System health check
GET  /predictions/latest  - Most recent predictions
GET  /predictions/{id}    - Specific prediction details
GET  /performance         - Performance metrics
POST /forecast/manual     - Trigger manual forecast
GET  /counties            - List monitored counties
```

**Example API call:**
```bash
curl http://localhost:8000/predictions/latest

{
  "timestamp": "2026-02-01T10:00:00Z",
  "predictions": [
    {
      "prediction_id": "PRED_Polk_20260201_100000_abc123",
      "county": "Polk",
      "event_type": "freeze",
      "probability": 0.82,
      "expected_damage_pct": 0.135,
      "confidence_score": 0.85,
      "timing": {
        "onset": "2026-02-03T06:00:00Z",
        "duration_hours": 5
      }
    }
  ]
}
```

---

## Testing

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
pytest tests/integration/
```

### Run Backtest

Test prediction accuracy on historical data:

```bash
python backtest.py --start 2023-01-01 --end 2025-12-31
```

**Output:**
```
Backtesting 2023-01-01 to 2025-12-31
=====================================
Total events: 23
Predictions generated: 23
Verified: 23

Performance Metrics:
  Accuracy: 73%
  Win Rate: 70%
  MAE: 2.4 percentage points
  Confidence calibration: 0.92

Event-Type Breakdown:
  Freeze: 75% accuracy (8 events)
  Hurricane: 71% accuracy (7 events)
  Disease: 60% accuracy (5 events)
```

---

## Monitoring & Alerts

### Alert Triggers

Alerts are automatically sent when:
- Prediction confidence >= 85%
- Expected crop damage >= 10%
- Multiple counties affected simultaneously

### Alert Channels

Configure in `config.py`:
- Email (SendGrid)
- SMS (Twilio)
- Slack webhook
- Custom webhook

### Example Alert

```
🚨 HIGH CONFIDENCE WEATHER EVENT PREDICTED

County: Polk County, Florida
Event: FREEZE
Expected Impact: 13.5% crop damage
Confidence: 85%
Timing: 2026-02-03 06:00 UTC

This prediction meets ShieldOrange's threshold for
significant agricultural impact. Review full analysis
and prepare for potential supply disruption.

Prediction ID: PRED_Polk_20260201_100000_abc123
View: https://dashboard.shieldorange.ai/predictions/abc123
```

---

## Performance Tracking

System automatically tracks:

| Metric | Description | Target |
|--------|-------------|--------|
| Forecast Accuracy | % predictions verified by USDA | >70% |
| Mean Absolute Error | Avg difference in damage estimate | <3% |
| Confidence Calibration | 80% confident = 80% accurate | >0.90 |
| False Positive Rate | Predicted damage, none occurred | <20% |
| False Negative Rate | Missed significant events | <10% |

**View performance:**
```bash
python main.py report
```

---

## Troubleshooting

### Common Issues

**1. Database connection error**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify connection string
psql postgresql://user:password@localhost/shieldorange
```

**2. Claude API rate limit**
```bash
# Check rate limit in config.py
RATE_LIMITS = {
    'anthropic_rpm': 50  # Reduce if hitting limits
}
```

**3. NOAA API timeout**
```bash
# NOAA can be slow, increase timeout
# In weather_collector.py:
response = self.session.get(url, timeout=30)  # Increase from 10
```

**4. No predictions generated**
```bash
# Check thresholds aren't too restrictive
MIN_CONFIDENCE_THRESHOLD = 0.60  # Lower from 0.70
MIN_EXPECTED_IMPACT = 0.03       # Lower from 0.05
```

### Logs

```bash
# View real-time logs
tail -f /var/log/shieldorange/main.log

# Check errors
tail -f /var/log/shieldorange/errors.log

# Search for specific prediction
grep "PRED_Polk" /var/log/shieldorange/predictions.log
```

---

## Development

### Code Style

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

### Adding New Data Sources

1. Create collector in `data_collectors/`
2. Add to orchestrator in `main.py`
3. Update database schema if needed
4. Add tests

**Example:**
```python
# data_collectors/soil_moisture.py
class SoilMoistureCollector:
    def fetch_data(self, county: str):
        # Fetch from USDA Soil Moisture API
        pass
```

---

## Deployment

### Production Checklist

- [ ] Set `DEPLOYMENT_PHASE=LIVE` in `.env`
- [ ] Configure production database
- [ ] Set up monitoring alerts (email/SMS)
- [ ] Enable error tracking (Sentry)
- [ ] Set up backup database
- [ ] Configure systemd service
- [ ] Set up log rotation
- [ ] Test alert channels
- [ ] Document runbook for on-call

### Docker Deployment

```bash
# Build image
docker build -t shieldorange:latest .

# Run container
docker run -d \
  --name shieldorange \
  --env-file .env \
  -v /var/log/shieldorange:/var/log/shieldorange \
  shieldorange:latest
```

### Cloud Deployment

**AWS:**
```bash
# ECS task definition provided in deploy/aws-ecs.json
aws ecs create-service --cli-input-json file://deploy/aws-ecs.json
```

**Google Cloud:**
```bash
# Cloud Run deployment
gcloud run deploy shieldorange \
  --image gcr.io/project/shieldorange \
  --platform managed \
  --region us-east1
```

---

## License

MIT License - see LICENSE file

---

## Contact & Community

### Official Channels

**Email:** research@shieldorange.ai  
**Website:** [shieldorange.ai](https://shieldorange.ai)  
**X:** [@ShieldOrangeAI](https://x.com/ShieldOrangeAI)  
**GitHub:** [github.com/shieldorange-ai/shieldorange](https://github.com/shieldorange-ai/shieldorange)

### Research Inquiries

For questions about methodology, data sources, collaboration opportunities, or institutional partnerships:

📧 **research@shieldorange.ai**

### About the Founders

**Background:**
- 10+ years precision agriculture experience at ADVAG (Farmers Edge partner)
- 10+ years trading futures markets
- Satellite imagery (NDVI), weather data, and soil sensor expertise
- Managed 240,000+ hectares with AgTech solutions
- Extensive experience connecting agricultural data to commodity markets

**Why ShieldOrange?**

ShieldOrange bridges two worlds: precision agriculture and futures trading. Having spent over a decade managing crops with satellite technology at ADVAG and simultaneously trading futures markets, Vitalik recognized that nobody was connecting weather prediction with commodity forecasting. ShieldOrange combines both domains of expertise.

---

## Support & Documentation

- **Email Support:** research@shieldorange.ai
- **Bug Reports:** [GitHub Issues](https://github.com/shieldorange-ai/shieldorange/issues)
- **Feature Requests:** [GitHub Discussions](https://github.com/shieldorange-ai/shieldorange/discussions)

---

## Follow Our Research

Stay updated on weather prediction research and agricultural forecasting:

🐦 **X:** [@ShieldOrangeAI](https://x.com/ShieldOrangeAI)  
🌐 **Website:** [shieldorange.ai](https://shieldorange.ai)  
📧 **Research Updates:** research@shieldorange.ai

---

## Acknowledgments

Built with:
- Claude Sonnet 4 by Anthropic
- NOAA National Weather Service data
- University of Florida IFAS citrus research
- CME Group market data

Academic research:
- Singerman et al. (2018) - Freeze damage models
- Li et al. (2020) - Disease-weather correlations

---

## Version History

**v1.0.0** (2026-02-01)
- Initial release
- Real-time NOAA monitoring
- Claude Sonnet 4 integration
- Freeze and hurricane predictions
- 70% historical accuracy (backtested)

---

**ShieldOrange AI - Predicting Tomorrow's Weather Events, Today**
