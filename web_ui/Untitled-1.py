cd "d\model_service"

uvicorn fastapi_model_service:app --reload --port 8000

cd "web_ui"

streamlit run app.py