🛰️ Satellite Telemetry Anomaly Scoring
Real-time unsupervised anomaly detection for satellite health monitoring.

This project provides a full-stack ML pipeline to identify irregularities in satellite telemetry data. Using an Isolation Forest algorithm, it scores telemetry "buckets" to detect potential hardware failures, solar flare impacts, or software glitches before they become critical.

🚀 Key Features
Unsupervised ML Pipeline: Uses Scikit-learn's Isolation Forest to detect anomalies without needing labeled "failure" data.

Explainable Scoring: Doesn't just give a "yes/no"—it identifies the top contributors (e.g., "Battery Voltage is the primary reason for this anomaly").

Production-Ready API: Built with FastAPI to allow real-time scoring of incoming satellite data streams.

Data Simulation: Includes a sophisticated simulation engine that emulates satellite signals (CPU temp, battery voltage, gyro speed) and injects realistic anomalies for testing.

🛠️ Technical Stack
Language: Python 3.10+

ML Framework: Scikit-learn (Isolation Forest, Standard Scaler)

Data Handling: Pandas, NumPy

API Framework: FastAPI & Pydantic (v2)

Model Management: Joblib (Serialization)

📊 How It Works
Simulation (src/simulate.py): Generates 2,000 minutes of "normal" orbital telemetry and 400 minutes of test data with injected anomalies.

Training (src/train.py): Vectorizes the raw telemetry, builds a feature schema (handling missing data via indicators), and trains the Isolation Forest.

Inference (src/app.py): A REST API that accepts telemetry buckets and returns:

anomaly_score: 0.0 (Normal) to 1.0 (Critical Anomaly).

confidence: Reliability of the score based on training distribution.

top_contributors: Which specific signals caused the anomaly.

⚡ Quick Start
1. Install Dependencies

Bash
pip install fastapi scikit-learn pandas joblib pydantic
2. Generate Data & Train Model

Bash
python -m src.simulate
python -m src.train
3. Run the API

Bash
uvicorn src.app:app --reload
The API will be live at http://127.0.0.1:8000. You can view the interactive documentation at /docs.

🔭 Research Context
This project was developed as part of a focus on satellite telemetry processing and astrophysics research. It explores how unsupervised models can identify celestial-driven anomalies in satellite hardware, such as those caused by high-energy particle events or sensor degradation.
