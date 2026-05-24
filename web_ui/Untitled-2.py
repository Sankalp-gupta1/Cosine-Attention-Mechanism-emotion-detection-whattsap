 # =========================================================
# FINAL COMPLETE app.py
# MULTI CLIENT + SHARED VIDEO + EMOTION SYSTEM
# =========================================================

import json
import os
import time
import cv2
import requests
import pandas as pd
import streamlit as st

from collections import deque

from webcam import (
    start_camera,
    stop_camera,
    detect_emotion
)

from camera_manager import (
    request_camera,
    release_camera,
    get_camera_status
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Distributed AI Emotion System",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]{
    background-color:#0b1120;
    color:white;
    font-family:'Segoe UI';
}

.block-container{
    padding-top:1rem;
}

.main-title{
    font-size:42px;
    font-weight:800;
    color:#00ffd5;
}

.sub-title{
    color:#8b949e;
    margin-bottom:20px;
}

.chat-card{
    background:#172033;
    padding:14px;
    border-radius:15px;
    margin-bottom:12px;
    border-left:4px solid #00ffd5;
}

.camera-active{
    color:#00ff99;
    font-weight:bold;
}

.camera-off{
    color:#ff4b4b;
    font-weight:bold;
}

.emotion-box{
    background:#111827;
    padding:10px;
    border-radius:12px;
    margin-bottom:10px;
    border:1px solid #1f2937;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FILES
# =========================================================

CHAT_FILE = "global_chat.json"
USERS_FILE = "online_users.json"

# =========================================================
# CREATE FILES
# =========================================================

if not os.path.exists(CHAT_FILE):
    with open(CHAT_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump([], f)

# =========================================================
# CHAT FUNCTIONS
# =========================================================

def load_messages():

    try:
        with open(CHAT_FILE, "r") as f:
            return json.load(f)

    except:
        return []

# =========================================================

def save_message(msg):

    messages = load_messages()

    messages.append(msg)

    # LAST 100 ONLY
    messages = messages[-100:]

    with open(CHAT_FILE, "w") as f:
        json.dump(messages, f, indent=4)

# =========================================================
# USER FUNCTIONS
# =========================================================

def load_users():

    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)

    except:
        return []

# =========================================================

def save_users(users):

    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# =========================================================
# SESSION STATE
# =========================================================

if "video_active" not in st.session_state:
    st.session_state.video_active = False

if "camera_user" not in st.session_state:
    st.session_state.camera_user = None

if "video_emotions" not in st.session_state:
    st.session_state.video_emotions = deque(maxlen=10)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🔥 Distributed AI Emotion Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        Live Multi-Client Chat + Facial Emotion + Sarcasm Detection
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# USERNAME
# =========================================================

username = st.sidebar.text_input(
    "Enter Username",
    value="Sankalp"
)

st.sidebar.success(f"Connected : {username}")

# =========================================================
# SAVE USER
# =========================================================

users = load_users()

if username not in users:

    users.append(username)
    save_users(users)

# =========================================================
# CAMERA STATUS
# =========================================================

status = get_camera_status()

if status["active"]:

    st.sidebar.warning(
        f"🎥 Camera Active : {status['user']}"
    )

else:

    st.sidebar.success(
        "🎥 Camera Available"
    )

# =========================================================
# ONLINE USERS
# =========================================================

st.sidebar.markdown("## 🟢 Online Users")

for user in users:

    if status["active"] and status["user"] == user:

        st.sidebar.markdown(
            f"""
            🎥 {user}
            <span class='camera-active'>
                (VIDEO ON)
            </span>
            """,
            unsafe_allow_html=True
        )

    else:

        st.sidebar.markdown(
            f"""
            📴 {user}
            <span class='camera-off'>
                (VIDEO OFF)
            </span>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# MAIN LAYOUT
# =========================================================

left, right = st.columns([2, 1])

# =========================================================
# LEFT SIDE CHAT
# =========================================================

with left:

    st.subheader("💬 Live Multi-Client Chat")

    chat_container = st.container(height=550)

    with chat_container:

        messages = load_messages()

        if len(messages) == 0:

            st.info("No messages yet")

        else:

            for msg in reversed(messages):

                st.markdown(
                    f"""
                    <div class="chat-card">

                        <b>👤 {msg['user']}</b>

                        <br><br>

                        💬 {msg['text']}

                        <br><br>

                        😄 Emotion : {msg['emotion']}

                        <br>

                        😂 Sarcasm : {msg['sarcasm']}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # =====================================================
    # CHAT INPUT
    # =====================================================

    text = st.chat_input(
        "Type your message..."
    )

    if text:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={
                    "text": text
                }
            )

            data = response.json()

            save_message({

                "user": username,
                "text": text,
                "emotion": data["emotion"],
                "sarcasm": data["sarcasm"]

            })

            st.rerun()

        except Exception as e:

            st.error(
                f"FastAPI Error : {e}"
            )

# =========================================================
# RIGHT SIDE VIDEO
# =========================================================

with right:

    st.subheader("🎥 Live Facial Emotion")

    # =====================================================
    # CAMERA AVAILABLE
    # =====================================================

    if not status["active"]:

        if st.button("🎥 Start Camera"):

            granted, message = request_camera(username)

            if granted:

                st.session_state.video_active = True
                st.session_state.camera_user = username

                st.rerun()

            else:

                st.error(message)

    # =====================================================
    # SOMEONE ELSE USING CAMERA
    # =====================================================

    elif status["user"] != username:

        st.warning(
            f"🎥 Live Camera : {status['user']}"
        )

        # =================================================
        # SHOW SHARED VIDEO
        # =================================================

        if os.path.exists("shared_camera.jpg"):

            image = cv2.imread(
                "shared_camera.jpg"
            )

            if image is not None:

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    image,
                    channels="RGB"
                )

        # =================================================
        # SHOW SHARED EMOTIONS
        # =================================================

        if os.path.exists("shared_emotion.json"):

            try:

                with open(
                    "shared_emotion.json",
                    "r"
                ) as f:

                    emotions = json.load(f)

                st.markdown(
                    "### 🎭 Live Emotions"
                )

                for emo in emotions[:10]:

                    st.markdown(
                        f"""
                        <div class="emotion-box">

                            {emo['emotion']}
                            ({emo['confidence']:.1f}%)

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except:
                pass

    # =====================================================
    # CURRENT USER CAMERA
    # =====================================================

    else:

        st.success("Camera Running")

        if st.button("❌ Stop Camera"):

            release_camera(username)

            st.session_state.video_active = False
            st.session_state.camera_user = None

            st.rerun()

        # =================================================
        # START CAMERA
        # =================================================

        cap, msg = start_camera()

        if cap is None:

            st.error(msg)

        else:

            FRAME_WINDOW = st.empty()

            emotion_placeholder = st.empty()

            while True:

                current_status = get_camera_status()

                if (
                    not current_status["active"]
                    or current_status["user"] != username
                ):
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                # =========================================
                # DETECT EMOTION
                # =========================================

                try:

                    frame, emotions = detect_emotion(frame)

                except Exception as e:

                    st.error(
                        f"Emotion Error : {e}"
                    )

                    break

                # =========================================
                # RGB
                # =========================================

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                FRAME_WINDOW.image(
                    frame,
                    channels="RGB"
                )

                # =========================================
                # LAST 10 EMOTIONS
                # =========================================

                if len(emotions) > 0:

                    emo = emotions[0]["emotion"]
                    conf = emotions[0]["confidence"]

                    latest = (
                        f"{emo} ({conf:.1f}%)"
                    )

                    if (
                        len(st.session_state.video_emotions) == 0
                        or
                        st.session_state.video_emotions[-1] != latest
                    ):

                        st.session_state.video_emotions.append(
                            latest
                        )

                # =========================================
                # SHOW LAST 10
                # =========================================

                with emotion_placeholder.container():

                    st.markdown(
                        "### 🎭 Recent Facial Emotions"
                    )

                    for item in reversed(
                        st.session_state.video_emotions
                    ):

                        st.markdown(
                            f"""
                            <div class="emotion-box">
                                {item}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                time.sleep(0.03)

            stop_camera(cap)

# =========================================================
# ANALYTICS
# =========================================================

st.divider()

st.subheader("📊 Analytics Dashboard")

messages = load_messages()

if len(messages) > 0:

    df = pd.DataFrame(messages)

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader(
        "Emotion Distribution"
    )

    emotion_counts = (
        df["emotion"]
        .value_counts()
    )

    st.bar_chart(
        emotion_counts
    )

    # =====================================================
    # EXPORT CHAT
    # =====================================================

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📂 Export Chat With Emotion",
        data=csv,
        file_name="emotion_chat_export.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No analytics yet"
    )