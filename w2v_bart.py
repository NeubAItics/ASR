import streamlit as st
from streamlit_mic_recorder import mic_recorder
from transformers import pipeline
import os
from pydub import AudioSegment
import io
import tempfile

# --- Configuration ---
ASR_MODEL_NAME = "facebook/wav2vec2-large-960h"
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"

st.set_page_config(layout="centered", page_title="Audio Processor")
st.title("🗣️ Audio to Transcript & Summary")
st.markdown("---")

# --- Load Models ---
@st.cache_resource
def load_asr_pipeline():
    with st.spinner(f"Loading ASR model ({ASR_MODEL_NAME})..."):
        return pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME)

@st.cache_resource
def load_summarization_pipeline(model_name):
    with st.spinner(f"Loading Summarization model ({model_name})..."):
        return pipeline("summarization", model=model_name)

asr_pipeline = load_asr_pipeline()
summarizer_pipeline = load_summarization_pipeline(SUMMARIZATION_MODEL_NAME)

# --- Session state for audio ---
if "audio_data_for_processing" not in st.session_state:
    st.session_state.audio_data_for_processing = None

st.markdown("---")
st.header("1. Choose Audio Input Method")

audio_source_option = st.radio(
    "Select input method:",
    ("Upload Audio File", "Record Live Audio"),
    key="audio_source_radio"
)

# --- Upload ---
if audio_source_option == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload an audio file (.mp3, .wav, .m4a, .flac)",
        type=["mp3", "wav", "m4a", "flac"]
    )

    if uploaded_file:
        st.audio(uploaded_file, format=uploaded_file.type)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_upload_file:
                temp_upload_file.write(uploaded_file.read())
                temp_path = temp_upload_file.name

            audio_segment = AudioSegment.from_file(temp_path)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1) 
            temp_wav = io.BytesIO()
            audio_segment.export(temp_wav, format="wav")
            temp_wav.seek(0)
            st.session_state.audio_data_for_processing = temp_wav
            st.success(f"Uploaded audio length: {audio_segment.duration_seconds:.2f} sec")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# --- Record ---
elif audio_source_option == "Record Live Audio":
    st.info("Click the mic to start/stop recording. Auto stops after 30s.")
    recorded = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=True,
        use_container_width=True,
        key="mic_recorder"
    )

    if recorded and 'bytes' in recorded:
        audio_bytes = recorded['bytes']
        st.audio(audio_bytes, format="audio/wav")

        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.name = "recorded_audio.wav"
        audio_buffer.seek(0)
        st.session_state.audio_data_for_processing = audio_buffer
        st.success("Audio recorded successfully!")
    elif recorded is None:
        st.info("No audio recorded yet.")
    else:
        st.warning("Recording failed or was empty.")

st.markdown("---")

# --- Process Audio ---
if st.button("Process Audio", use_container_width=True):
    audio_data_for_processing = st.session_state.audio_data_for_processing
    transcript = None

    if audio_data_for_processing:
        st.subheader("Processing Results:")
        with st.spinner("Transcribing audio..."):
            try:
                audio_data_for_processing.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(audio_data_for_processing.read())
                    temp_path = temp_file.name

                result = asr_pipeline(temp_path)
                transcript = result["text"]
                st.session_state.transcript = transcript
                st.success("Transcription Complete!")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if transcript:
            with st.spinner("Generating summary..."):
                try:
                    summary_result = summarizer_pipeline(
                        transcript,
                        max_length=150,
                        min_length=40,
                        do_sample=False
                    )
                    summary = summary_result[0]['summary_text']
                    st.session_state.summary = summary
                    st.success("Summary Complete!")
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
    else:
        st.warning("Please upload or record audio before processing.")

# --- Show stored results ---
if "transcript" in st.session_state:
    st.subheader("📝 Transcript:")
    st.code(st.session_state.transcript)
    st.download_button(
        label="📥 Download Transcript",
        data=st.session_state.transcript,
        file_name="transcript.txt",
        mime="text/plain"
    )

if "summary" in st.session_state:
    st.subheader("📄 Summary:")
    st.write(st.session_state.summary)
    st.download_button(
        label="📥 Download Summary",
        data=st.session_state.summary,
        file_name="summary.txt",
        mime="text/plain"
    )

st.markdown("---")
st.info("Built using Hugging Face Transformers, Streamlit, and PyDub.")
st.caption(f"ASR Model: {ASR_MODEL_NAME}, Summarization Model: {SUMMARIZATION_MODEL_NAME}")
