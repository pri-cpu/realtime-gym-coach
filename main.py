import os
import time

import pandas as pd
import streamlit as st
from groq import Groq
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from services.auth.login_wall import render_login_wall
from services.coaching.llm import LLMCoach
from services.coaching.tts import TextToSpeech
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio
from services.config.workout_config import EXERCISE_OPTIONS
from services.persistence.exercise_repository import get_users_exercises, init_db
from services.state.session_defaults import initial_session_defaults
from services.tracking.metrics import sync_metrics_update
from services.ui.style_loader import inject_local_font, inject_webrtc_styles, load_css
from services.vision.exercise_video_processor import VideoProcessorClass

# How often the live panels refresh while a workout is running.
REFRESH_SECONDS = 0.5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def init_voice_pipeline():
    """Create the Groq LLM + TTS pipeline once per session."""
    if "voice_pipeline" in st.session_state:
        return

    try:
        api_key = os.environ.get("GROQ_API_KEY", "")

        if not api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]

        # Stray spaces/newlines/quotes in the key cause "Connection error".
        api_key = str(api_key).strip().strip('"').strip("'")

        groq_client = Groq(api_key=api_key)
        llm_coach = LLMCoach(groq_client)
        tts = TextToSpeech(groq_client)
        st.session_state.voice_pipeline = VoicePipeline(llm_coach, tts)
    except Exception as e:
        print(f"[voice] pipeline disabled: {e}")
        st.session_state.voice_pipeline = None


def fire_voice_event(event, exercise):
    """Send a workout event to the voice pipeline without ever crashing the UI."""
    pipeline = st.session_state.get("voice_pipeline")
    if not pipeline:
        return

    try:
        result = pipeline.process_event(event=event, exercise=exercise, metrics={})
    except Exception as e:
        print(f"[voice] {event} failed: {e}")
        return

    if result:
        st.session_state.audio_to_play, st.session_state.coach_feedback = result


def render_coach_output():
    """Show the latest coach audio + text."""
    audio = st.session_state.get("audio_to_play")
    if audio:
        autoplay_audio(audio)

    feedback = st.session_state.get("coach_feedback")
    if feedback:
        st.markdown("")
        st.success(f"🤖 **Coach:** {feedback}")


# --------------------------------------------------------------------------- #
# Live panels (fragments: only these blocks rerun, not the whole app)
# --------------------------------------------------------------------------- #
@st.fragment(run_every=REFRESH_SECONDS)
def render_progress_panel():
    """Sidebar progress + per-exercise metrics, refreshed on a timer."""
    get = st.session_state.get
    exercise = get("exercise_type")

    st.divider()
    st.subheader("Progress")

    st.metric("Total Reps", f"{get('reps', 0)}")
    st.metric("Current Set Reps", f"{get('current_set_reps', 0)} / {get('reps_per_set', 0)}")
    st.metric("Sets Completed", f"{get('sets_completed', 0)} / {get('target_sets', 0)}")

    st.divider()

    if exercise == "Squats":
        st.subheader("Squat Metrics")
        st.metric("Knee Angle", f"{get('knee_angle', '-')}°")
        st.metric("Back Angle", f"{get('back_angle', '-')}°")
        st.metric("Depth Status", get("depth_status", "-"))

    elif exercise == "Push-ups":
        st.subheader("Push-up Metrics")
        st.metric("Elbow Angle", f"{get('elbow_angle', '-')}°")
        st.metric("Body Alignment", get("body_alignment", "-"))
        st.metric("Hip Position", get("hip_status", "-"))

    elif exercise == "Biceps Curls (Dumbbell)":
        st.subheader("Curl Metrics")
        st.metric("Elbow Angle", f"{get('elbow_angle', '-')}°")
        st.metric("Shoulder Stability", get("shoulder_status", "-"))
        st.metric("Swing Detection", get("swing_status", "-"))

    elif exercise == "Shoulder Press":
        st.subheader("Shoulder Press Metrics")
        st.metric("Elbow Angle", f"{get('elbow_angle', '-')}°")
        st.metric("Arm Extension", get("extension_status", "-"))
        st.metric("Back Arch", get("back_arch_status", "-"))

    elif exercise == "Lunges":
        st.subheader("Lunge Metrics")
        st.metric("Front Knee Angle", f"{get('front_knee_angle', '-')}°")
        st.metric("Torso Angle", f"{get('torso_angle', '-')}°")
        st.metric("Balance Status", get("balance_status", "-"))


@st.fragment(run_every=REFRESH_SECONDS)
def live_coach_panel(context):
    """Pull the latest metrics from the video processor and show coach output."""
    try:
        sync_metrics_update(context)
    except Exception as e:
        print(f"[sync] sync_metrics_update failed: {e}")

    render_coach_output()


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(
        page_icon="🏋️‍♀️",
        page_title="AI Real-time GYM Coach",
        initial_sidebar_state="expanded",
        layout="centered",
    )

    # These emit UI elements, so they must run on every full rerun.
    # (Cache the file reading/encoding inside the helpers with @st.cache_data.)
    load_css(os.path.join(os.getcwd(), "static", "style.css"))
    inject_local_font(os.path.join(os.getcwd(), "static", "AdobeClean.otf"), "AdobeClean")

    # No UI output, so it's safe to run only once per session.
    if not st.session_state.get("db_ready"):
        init_db()
        st.session_state.db_ready = True

    if not render_login_wall():
        return

    initial_session_defaults()
    init_voice_pipeline()

    workout_started = st.session_state.get("workout_started", False)

    # ---------------------------- Sidebar ---------------------------------- #
    with st.sidebar:
        st.title("🏋️‍♂️ Apna AI Coach")

        if st.session_state.get("username"):
            st.caption(f"👤 Login as {st.session_state.username}")

        st.divider()
        st.subheader("Workout Plan")

        if not workout_started:
            plan_exercise = st.selectbox("Exercise", options=EXERCISE_OPTIONS, key="plan_exercise")
            plan_sets = st.number_input("Sets", min_value=0, max_value=50, key="plan_sets", step=1)
            plan_reps = st.number_input("Reps per Set", min_value=0, max_value=50, key="plan_reps", step=1)

            st.markdown("")

            if st.button("Start Workout", width="stretch", key="start_session_button"):
                st.session_state.exercise_type = plan_exercise
                st.session_state.target_sets = int(plan_sets)
                st.session_state.reps_per_set = int(plan_reps)
                st.session_state.reps = 0
                st.session_state.workout_started = True
                st.session_state.set_cycle_started_at = time.time()
                st.session_state.last_saved_sets_completed = 0
                st.session_state.last_notified_sets_completed = 0
                st.session_state.last_notified_workout_complete = False

                fire_voice_event("workout_started", plan_exercise)
                st.rerun()
        else:
            exercise = st.session_state.get("exercise_type")
            sets = st.session_state.get("target_sets")
            reps = st.session_state.get("reps_per_set")

            st.info(f"**{exercise}** -- {sets} Sets / {reps} Reps")

            if st.button("End Workout", key="end_session_button", width="stretch"):
                st.session_state.workout_started = False
                fire_voice_event("workout_completed", exercise)
                st.rerun()

            render_progress_panel()

    # ---------------------------- Main area -------------------------------- #
    st.title("AI Real-time GYM Coach")
    st.markdown("#### Real-time pose detection with proactive AI voice coaching")

    coach_slot = st.container()

    if not workout_started:
        with coach_slot:
            render_coach_output()

        st.markdown(
            """
            <div style="
                border: 10px dashed #444;
                border-radius: 0px;
                padding: 48px 32px;
                text-align: center;
                color: #888;
                margin-top: 32px;
                margin-bottom: 32px;
            ">
                <h2 style="color:#ccc; margin-bottom:8px;">👈 Set your workout plan</h2>
                <p style="font-size:1.05rem;">
                    Choose your exercise, sets and reps in the sidebar,<br>
                    then click <strong>Start Workout</strong> to activate the camera and AI coach.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        context = webrtc_streamer(
            key="exercise-analysis",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessorClass,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15},
                },
                "audio": False,
            },
            async_processing=True,
        )

        # Replaces the old `time.sleep(0.25); st.rerun()` loop.
        with coach_slot:
            live_coach_panel(context)

        inject_webrtc_styles()

    # ---------------------------- History ---------------------------------- #
    st.divider()
    st.markdown("#### Workout History")

    user_id = st.session_state.get("user_id", 0)

    if isinstance(user_id, int):
        history_rows = get_users_exercises(user_id)

        arr = [
            {
                "Exercise": row["exercise_name"],
                "Reps": row["reps"],
                "Sets": row["sets"],
                "Time (sec)": row["time"],
                "Date": row["created_at"],
            }
            for row in history_rows
        ]

        df = pd.DataFrame(arr)

        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            agg_df = (
                df.groupby(["Exercise", "Date"])
                .agg({"Reps": "sum", "Sets": "sum", "Time (sec)": "sum"})
                .reset_index()
            )
            agg_df.index += 1
            st.table(agg_df, border="horizontal")
        else:
            st.info("No workout history found.")


if __name__ == "__main__":
    main()