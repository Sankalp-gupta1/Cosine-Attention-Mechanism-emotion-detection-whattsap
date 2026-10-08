# Real-Time Multilingual Emotion, Sarcasm & Facial Emotion System

A distributed AI communication prototype that combines **multilingual text emotion detection, sarcasm detection, real-time chat, facial emotion recognition, and multi-client coordination**.

The system is split into independent services so that model inference, chat transport, camera access, storage, and the user interface can run as separate components.

## Core Capabilities

- Real-time multi-user chat
- Multilingual text emotion classification
- Sarcasm classification
- XLM-RoBERTa based text models
- FastAPI inference service
- WebSocket chat server
- MySQL message persistence
- Facial emotion recognition workflow
- Shared/single-camera access coordination
- Streamlit-based monitoring and interaction UI
- Confusion-matrix and model-evaluation utilities

## Text Emotion Labels

The current text emotion service maps predictions to:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

Sarcasm is classified as:

- Sarcastic
- Not Sarcastic

## Architecture

```text
User / Client
     │
     ▼
WebSocket Chat Server :6789
     │
     ├──► FastAPI Model Service :8000
     │        ├── Emotion Model
     │        └── Sarcasm Model
     │
     ├──► MySQL message storage
     │
     └──► Broadcast enriched message
              │
              ▼
        Connected Clients

Facial Emotion Pipeline
     │
     ├── Camera Manager / Lock
     ├── Facial Emotion Server :6790
     └── Streamlit UI / Analytics
```

## Repository Structure

```text
Cosine-Attention-Mechanism-emotion-detection-whattsap/
├── chat_client/
├── chat_server/
│   └── server_node.py
├── model_service/
│   └── fastapi_model_service.py
├── facial_emotion/
├── web_ui/
├── data/
├── confusion_matrix_test.py
├── sarcasm_confusion.py
├── training_metrics.png
├── requirements.txt
└── runtime.txt
```

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- XLM-RoBERTa
- FastAPI
- Uvicorn
- WebSockets
- Streamlit
- OpenCV
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- MySQL

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The project also uses MySQL, so create/configure the local database expected by the chat and facial-emotion services.

## Local Configuration

Before running the system, review the local configuration in the source files:

- Set valid local paths for the trained emotion and sarcasm model directories.
- Configure the MySQL connection for your environment.
- Do not commit real database passwords or private credentials.
- Make sure the required trained model files are available locally.

## Run the Text Chat Pipeline

### 1. Start the model service

```bash
cd model_service
uvicorn fastapi_model_service:app --reload --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the WebSocket chat server

```bash
cd chat_server
python server_node.py
```

Chat server:

```text
ws://127.0.0.1:6789
```

### 3. Start one or more chat clients

```bash
cd chat_client
python client.py
```

Multiple terminals can be used to simulate multiple connected users.

Each message is enriched with emotion and sarcasm predictions before being broadcast.

## Facial Emotion Service

The facial-emotion subsystem contains:

- Camera coordination
- Face emotion client/server components
- Live webcam processing
- Model training/testing utilities
- A WebSocket face server on port `6790`

The server grants camera access to one active user at a time and releases the camera lock when the client disconnects.

## Streamlit Interface

The `web_ui/` package contains the project dashboard and supporting camera-management code.

From that directory:

```bash
streamlit run app.py
```

## Evaluation Utilities

The repository includes scripts for:

- Emotion confusion matrix testing
- Sarcasm confusion analysis
- Training metrics visualization

## Project Goal

The project explores how distributed communication systems can move beyond plain-message transport by attaching machine-understood emotional context to conversations while coordinating text and facial signals in real time.

## Limitations

This is a research/prototype system and currently depends on local model directories and local service configuration. Production deployment would require secure secret management, persistent service orchestration, model packaging, authentication, and more robust concurrency handling.

## Author

**Sankalp Gupta**

GitHub: https://github.com/Sankalp-gupta1
