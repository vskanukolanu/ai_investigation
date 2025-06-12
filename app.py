import streamlit as st
from demo import DemoInvestigationAgent
import base64
from pathlib import Path
import time
import json
import numpy as np

def get_base64_download_link(file_path, file_name):
    """Generate a download link for a file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = DemoInvestigationAgent()
    if 'investigation_state' not in st.session_state:
        st.session_state.investigation_state = None
    if 'selected_hypotheses' not in st.session_state:
        st.session_state.selected_hypotheses = []
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'question_analyzed' not in st.session_state:
        st.session_state.question_analyzed = False
    if 'metrics_analyzed' not in st.session_state:
        st.session_state.metrics_analyzed = False
    if 'hypotheses_generated' not in st.session_state:
        st.session_state.hypotheses_generated = False

def reset_session_state():
    """Reset investigation-related session state variables"""
    st.session_state.investigation_state = None
    st.session_state.selected_hypotheses = []
    st.session_state.summary = None
    st.session_state.feedback_submitted = False
    st.session_state.current_step = 0
    st.session_state.analysis_complete = False
    st.session_state.question_analyzed = False
    st.session_state.metrics_analyzed = False
    st.session_state.hypotheses_generated = False

def analyze_hypothesis(hypothesis, metrics_info):
    """Perform detailed analysis on a selected hypothesis"""
    analysis = {
        'confidence': hypothesis['confidence'],
        'evidence': hypothesis.get('evidence', []),
        'metrics_correlation': {},
        'recommendations': []
    }
    
    # Analyze metric correlations
    for metric in metrics_info['metrics']:
        correlation = np.random.uniform(0.5, 1.0)  # Simulated correlation
        analysis['metrics_correlation'][metric] = correlation
        
    # Generate specific recommendations
    if 'temporal_pattern' in hypothesis.get('evidence', []):
        analysis['recommendations'].append("Set up automated monitoring for similar temporal patterns")
    if 'volume_increase' in hypothesis.get('evidence', []):
        analysis['recommendations'].append("Create volume-based alert thresholds")
    if 'share_pattern' in hypothesis.get('evidence', []):
        analysis['recommendations'].append("Monitor share velocity as an early indicator")
    
    return analysis

def generate_summary(prompt, selected_hypotheses, metrics_info):
    """Generate a summary of the investigation findings"""
    summary = f"""
    ### 📊 Investigation Summary
    
    **Question Analyzed**: {prompt}
    
    **Key Findings**:
    1. Metrics Analyzed: {', '.join(metrics_info['metrics'])}
    2. Number of Anomalies Detected: {len(metrics_info.get('anomalies', []))}
    
    **Validated Hypotheses**:
    """
    
    for hyp in selected_hypotheses:
        analysis = analyze_hypothesis(hyp, metrics_info)
        summary += f"\n- {hyp['description']} (Confidence: {hyp['confidence']:.2%})"
        if analysis['evidence']:
            summary += f"\n  Evidence: {', '.join(analysis['evidence'])}"
        summary += "\n  Key Correlations:"
        for metric, corr in analysis['metrics_correlation'].items():
            summary += f"\n    - {metric}: {corr:.2%}"
        if analysis['recommendations']:
            summary += "\n  Recommendations:"
            for rec in analysis['recommendations']:
                summary += f"\n    - {rec}"
    
    summary += """
    
    **Overall Recommendations**:
    1. Monitor these metrics for similar patterns in future
    2. Set up alerts for anomaly thresholds
    3. Conduct deeper analysis on validated hypotheses
    4. Review and update monitoring thresholds regularly
    """
    
    return summary

def main():
    st.set_page_config(
        page_title="Investigation Agent Demo",
        page_icon="🔍",
        layout="wide"
    )

    # Welcome message
    st.title("🔍 Investigation Agent Demo")
    
    welcome_message = """
    👋 Hi! I'm your Investigation Agent, an AI-powered analyst specialized in detecting and analyzing metric anomalies 
    in social media data.

    **How I can help you:**
    - 🔍 Investigate engagement spikes and unusual patterns
    - 📊 Analyze multiple metrics simultaneously
    - 🧪 Generate and validate hypotheses
    - 📈 Create visualizations and reports
    - 📝 Provide actionable recommendations

    **We'll work together through these steps:**
    1. You'll ask a question about metric anomalies
    2. I'll analyze the question and relevant data
    3. We'll review the initial findings together
    4. You'll select hypotheses to investigate further
    5. I'll provide detailed analysis and recommendations
    """
    
    st.markdown(welcome_message)

    # Initialize session state
    initialize_session_state()

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input (only show if no analysis is in progress)
    if st.session_state.current_step == 0:
        if prompt := st.chat_input("Ask about metric anomalies (e.g., 'Investigate the spike in engagement during April 2025')"):
            # Reset state for new investigation
            reset_session_state()
            st.session_state.current_step = 1
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Store the prompt
            st.session_state.current_prompt = prompt
            st.rerun()

    # Step 1: Question Analysis
    if st.session_state.current_step == 1:
        with st.chat_message("assistant"):
            st.markdown("### Step 1: Analyzing Your Question")
            st.markdown("Let me analyze your question to understand what we need to investigate.")
            
            with st.spinner("Analyzing question..."):
                st.session_state.agent.investigate(st.session_state.current_prompt)
                st.session_state.question_analyzed = True
            
            st.success("✅ Question analysis complete!")
            if st.button("Continue to Metric Analysis →"):
                st.session_state.current_step = 2
                st.rerun()

    # Step 2: Metric Analysis
    elif st.session_state.current_step == 2:
        with st.chat_message("assistant"):
            st.markdown("### Step 2: Analyzing Metrics")
            metrics_info = st.session_state.agent._analyze_metrics()
            
            # Display metrics summary
            st.markdown("📊 Here's what I found in the metrics:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Metrics Analyzed", len(metrics_info['metrics']))
            with col2:
                st.metric("Anomalies Found", len(metrics_info['anomalies']))
            with col3:
                st.metric("Overall Trend", metrics_info['trends']['overall'].title())
            
            st.session_state.metrics_info = metrics_info
            st.session_state.metrics_analyzed = True
            
            if st.button("Continue to Hypothesis Generation →"):
                st.session_state.current_step = 3
                st.rerun()

    # Step 3: Hypothesis Generation
    elif st.session_state.current_step == 3:
        with st.chat_message("assistant"):
            st.markdown("### Step 3: Generating Hypotheses")
            hypotheses = st.session_state.agent._generate_hypotheses(st.session_state.metrics_info)
            
            st.markdown("""
            I've generated several hypotheses based on the data analysis. Please:
            1. Review each hypothesis and its confidence level
            2. Select the ones you'd like me to analyze further
            """)
            
            # Display hypotheses for selection
            selected_hypotheses = []
            for idx, hypothesis in enumerate(hypotheses):
                selected = st.checkbox(
                    f"{hypothesis['description']} (Confidence: {hypothesis['confidence']:.2%})",
                    key=f"hyp_{idx}"
                )
                if selected:
                    selected_hypotheses.append(hypothesis)
            
            st.session_state.selected_hypotheses = selected_hypotheses
            st.session_state.hypotheses_generated = True
            
            if len(selected_hypotheses) > 0:
                if st.button("Continue to Detailed Analysis →"):
                    st.session_state.current_step = 4
                    st.rerun()
            else:
                st.warning("Please select at least one hypothesis to continue.")

    # Step 4: Detailed Analysis
    elif st.session_state.current_step == 4:
        with st.chat_message("assistant"):
            st.markdown("### Step 4: Detailed Analysis")
            
            # Display visualizations
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Metric Trends")
                st.image("demo_output/metric_trends.png")
            with col2:
                st.subheader("⚠️ Detected Anomalies")
                st.image("demo_output/anomalies.png")

            # Generate and display summary
            st.session_state.summary = generate_summary(
                st.session_state.current_prompt,
                st.session_state.selected_hypotheses,
                st.session_state.metrics_info
            )
            st.markdown(st.session_state.summary)
            
            if st.button("Continue to Downloads and Feedback →"):
                st.session_state.current_step = 5
                st.rerun()

    # Step 5: Downloads and Feedback
    elif st.session_state.current_step == 5:
        with st.chat_message("assistant"):
            st.markdown("### Step 5: Downloads and Feedback")
            
            # Downloads section
            st.markdown("#### 📥 Download Reports")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    get_base64_download_link(
                        "demo_output/investigation_report.pptx",
                        "investigation_report.pptx"
                    ),
                    unsafe_allow_html=True
                )
            with col2:
                summary_path = Path("demo_output/investigation_summary.txt")
                summary_path.write_text(st.session_state.summary)
                st.markdown(
                    get_base64_download_link(
                        str(summary_path),
                        "investigation_summary.txt"
                    ),
                    unsafe_allow_html=True
                )
            
            # Feedback section
            if not st.session_state.feedback_submitted:
                st.markdown("#### 📝 Your Feedback")
                st.markdown("Please help me improve by providing your feedback:")
                
                feedback_useful = st.select_slider(
                    "How useful was this analysis?",
                    options=["Not useful", "Somewhat useful", "Useful", "Very useful", "Extremely useful"],
                    value="Useful"
                )
                
                feedback_accuracy = st.select_slider(
                    "How accurate were the hypotheses?",
                    options=["Not accurate", "Somewhat accurate", "Accurate", "Very accurate", "Extremely accurate"],
                    value="Accurate"
                )
                
                feedback_comments = st.text_area(
                    "Additional comments or suggestions:",
                    placeholder="Please share any specific feedback or suggestions for improvement..."
                )
                
                if st.button("Submit Feedback"):
                    st.success("Thank you for your feedback! I'll use it to improve future analyses.")
                    st.session_state.feedback_submitted = True
                    
            if st.button("Start New Investigation"):
                reset_session_state()
                st.rerun()

    # Sidebar with information
    with st.sidebar:
        st.subheader("ℹ️ About")
        st.markdown("""
        This interactive demo uses:
        - Lightweight NLP models
        - Time series analysis
        - Anomaly detection
        - Automated visualization
        - Report generation
        
        Try asking about:
        - Engagement spikes
        - Seasonal patterns
        - Metric correlations
        - Specific time periods
        """)
        
        # Show current step
        if st.session_state.current_step > 0:
            st.markdown("---")
            st.markdown("### 📍 Current Progress")
            steps = [
                "Ask Question",
                "Question Analysis",
                "Metric Analysis",
                "Hypothesis Selection",
                "Detailed Analysis",
                "Feedback & Downloads"
            ]
            for i, step in enumerate(steps):
                if i < st.session_state.current_step:
                    st.markdown(f"✅ {step}")
                elif i == st.session_state.current_step:
                    st.markdown(f"🔄 {step}")
                else:
                    st.markdown(f"⏳ {step}")
        
        # Add a clear button
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            reset_session_state()
            st.rerun()

if __name__ == "__main__":
    main() 