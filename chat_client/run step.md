 
Terminal 1 — Start model service (you already did this; if not):
cd "C:\Users\hp pc\emotion-detection-whatsapp\model_service"
uvicorn fastapi_model_service:app --reload --port 8000


Check http://127.0.0.1:8000/docs works.

Terminal 2 — Start chat server:
cd "C:\Users\hp pc\emotion-detection-whatsapp\chat_server"
python server_node.py

You should see:
WebSocket server started at ws://0.0.0.0:6789

Terminal 3 — Start one or more clients:

You can open multiple terminals (to simulate multiple users).

cd "C:\Users\hp pc\emotion-detection-whatsapp\chat_client"
python client.py


Type messages in any client terminal. After you send, all clients will receive the message enriched with emotion and sarcasm from your FastAPI service.

Example printed output:

➡ Message: I am so happy today!
   Emotion: Joy 😀
   Sarcasm: Not Sarcastic 🙂
