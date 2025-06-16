# Investigation Agent 🔍

An AI-powered system for analyzing anomalies in metric data with an interactive web interface.

## Features 🌟

- **Interactive Web Interface**: Step-by-step guided analysis process
- **Smart Anomaly Detection**: Identifies unusual patterns in engagement metrics
- **Multi-metric Analysis**: Analyzes engagement, comments, shares, and more
- **Hypothesis Generation**: AI-powered hypothesis creation and validation
- **Visual Analytics**: Automated visualization of trends and anomalies
- **Report Generation**: Automated PowerPoint and text report creation
- **User Feedback System**: Continuous improvement through user feedback

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/investigation_agent.git
cd investigation_agent
```

2. Create and activate a virtual environment:
```bash
python -m venv ai_inv
source ai_inv/bin/activate  # On Windows: ai_inv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage 💡

1. Start the web interface:
```bash
streamlit run app.py
```

2. Navigate through the analysis steps:
   - Enter your investigation question
   - Review initial analysis
   - Select hypotheses to investigate
   - Get detailed analysis and visualizations
   - Download reports and provide feedback

## System Components 🔧

### Weight Optimization System
- Selection weight: 50%
- Execution weight: 20%
- Context weight: 20%
- Temporal weight: 5%
- Relationship weight: 5%

### AI Models
- **Text Analysis**: LLaMA 2 (7B), BERT-tiny, GPT2-small, DistilBERT, T5-small
- **Time Series**: Prophet, ARIMA, Isolation Forest, STUMPY
- **Pattern Recognition**: LightGBM, FastAI, XGBoost
- **NLP**: Spacy-small, NLTK, FastText

### Implementation Details
- Backend: FastAPI with SQLite
- Interfaces: Web UI (Streamlit) and CLI
- Containerization: Docker support
- Output: PowerPoint reports, visualizations, text summaries

## Project Structure 📁

```
investigation_agent/
├── app.py              # Streamlit web interface
├── demo.py             # Demo implementation
├── requirements.txt    # Project dependencies
├── README.md          # Documentation
├── app/               # Core application code
├── data/              # Data storage
└── templates/         # Report templates
```

## Example Investigation 📊

```python
# Via Python API
from demo import DemoInvestigationAgent

agent = DemoInvestigationAgent()
results = agent.investigate("Investigate the spike in engagement during April 2025")
```

Or use the web interface for a guided experience:
1. Enter your investigation question
2. Follow the step-by-step analysis process
3. Select relevant hypotheses
4. Review detailed analysis and visualizations
5. Download reports and provide feedback

## Output Examples 📈

- **Metric Analysis**: Engagement trends, anomaly detection
- **Visualizations**: Time series plots, anomaly highlights
- **Reports**: PowerPoint presentations, text summaries
- **Hypotheses**: AI-generated explanations with confidence scores

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details. 
