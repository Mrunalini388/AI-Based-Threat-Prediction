# AI-Based Threat Prediction
 Machine Learning for Network Traffic Threat Detection

This project uses Artificial Intelligence to monitor network traffic and detect security threats such as hacking, malware, and other intrusions. By learning patterns of normal network behavior, the system can identify suspicious activity and issue real-time alerts.

# Features
Detects threats based on network traffic patterns

Learns normal behavior using machine learning models (Random Forest, SVM)

Sends real-time alerts for detected anomalies

Implements class balancing with SMOTE for better detection accuracy

Planned integration of deep learning models for enhanced performance

# Prerequisites
  Python 3.x
 Required packages: pandas, numpy, scikit-learn, imbalanced-learn

# Setup & Usage
1.Clone the repository:

```
git clone https://github.com/Mrunalini388/AI-Based-Threat-Prediction
cd AI-Based-Threat-Prediction
```
2.Create and activate a virtual environment:

macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```
Windows:
```
python -m venv venv
venv\Scripts\activate\
```
3.Install dependencies:
```
pip install -r requirements.txt
```
Place the dataset file UNSW-NB15_1.csv in the project root or update the file path in Ai threat detection.py.

4.Run the threat detection script:
```
python Ai\ threat\ detection.py
```
