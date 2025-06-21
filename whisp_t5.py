import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
from transformers import pipeline
import os
from pydub import AudioSegment
import io
import tempfile
from fpdf import FPDF # Import FPDF for PDF generation

# --- Configuration ---
WHISPER_MODEL_NAME = "base"
SUMMARIZATION_MODEL_NAME = "t5-small"

st.set_page_config(layout="centered", page_title="Audio Processor")
st.title("🗣️ Audio to Transcript & Summary")
st.markdown("---")

# --- Load Models ---
@st.cache_resource
def load_whisper_model():
    with st.spinner(f"Loading Whisper model ({WHISPER_MODEL_NAME})..."):
        try:
            model = whisper.load_model(WHISPER_MODEL_NAME)
            st.success(f"Whisper model '{WHISPER_MODEL_NAME}' loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Failed to load Whisper model. Please check your internet connection and disk space. Error: {e}")
            st.stop()

@st.cache_resource
def load_summarization_pipeline(model_name):
    with st.spinner(f"Loading Summarization model ({model_name})..."):
        try:
            summarizer = pipeline("summarization", model=model_name)
            st.success(f"Summarization model '{model_name}' loaded successfully!")
            return summarizer
        except Exception as e:
            st.error(f"Failed to load summarization model. Please check your internet connection, system resources (RAM/GPU), and ensure 'transformers' and 'accelerate' are up-to-date. Error: {e}")
            st.stop()

whisper_model = load_whisper_model()
summarizer_pipeline = load_summarization_pipeline(SUMMARIZATION_MODEL_NAME)

# --- Session state for audio and results ---
if "audio_data_for_processing" not in st.session_state:
    st.session_state.audio_data_for_processing = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "summary_text" not in st.session_state:
    st.session_state.summary_text = None

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
        temp_path = None # Initialize for finally block

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_upload_file:
                temp_upload_file.write(uploaded_file.read())
                temp_path = temp_upload_file.name

            audio_segment = AudioSegment.from_file(temp_path)
            temp_wav = io.BytesIO()
            audio_segment.export(temp_wav, format="wav")
            temp_wav.seek(0)
            st.session_state.audio_data_for_processing = temp_wav
            st.info(f"Uploaded audio length: {audio_segment.duration_seconds:.2f} sec")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}. Please ensure FFmpeg is installed and the file is not corrupted.")
            st.session_state.audio_data_for_processing = None # Reset on error
        finally:
            if temp_path and os.path.exists(temp_path):
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
        audio_buffer.name = "recorded_audio.wav" # For compatibility with Whisper
        audio_buffer.seek(0)
        st.session_state.audio_data_for_processing = audio_buffer
        st.success("Audio recorded successfully!")
    elif recorded is None:
        st.info("No audio recorded yet.")
    else:
        st.warning("Recording failed or was empty.")
        st.session_state.audio_data_for_processing = None # Reset on failure

st.markdown("---")

# --- Function to generate PDF ---
def generate_pdf(transcript, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add Transcript
    pdf.multi_cell(0, 10, "--- Transcript ---")
    pdf.ln(5) # Line break
    
    # Handle potential non-ASCII characters by encoding to latin-1 or similar if utf-8 fails in fpdf
    # FPDF's multi_cell will try to encode. If it fails with default fonts,
    # the fallback below handles it by replacing unrenderable chars.
    try:
        pdf.multi_cell(0, 10, transcript)
    except UnicodeEncodeError:
        # FPDF only supports certain encodings for built-in fonts (like latin-1).
        # This workaround replaces characters that cannot be encoded.
        st.warning("Some characters in the transcript could not be rendered in the PDF (due to font limitations). They have been replaced.")
        pdf.multi_cell(0, 10, transcript.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.ln(10) # Larger line break

    # Add Summary
    if summary:
        pdf.multi_cell(0, 10, "--- Summary ---")
        pdf.ln(5)
        try:
            pdf.multi_cell(0, 10, summary)
        except UnicodeEncodeError:
            st.warning("Some characters in the summary could not be rendered in the PDF (due to font limitations). They have been replaced.")
            pdf.multi_cell(0, 10, summary.encode('latin-1', 'replace').decode('latin-1'))

    return pdf.output(dest='S') # <--- THE FIX: Removed .encode('latin-1') here!

# --- Process Audio ---
if st.button("Process Audio", use_container_width=True):
    # Clear previous results
    st.session_state.transcript_text = None
    st.session_state.summary_text = None

    if st.session_state.audio_data_for_processing:
        st.subheader("Processing Results:")
        current_wav_temp_path = None # To store path for cleanup

        with st.spinner("Transcribing audio..."):
            try:
                st.session_state.audio_data_for_processing.seek(0)
                # Write BytesIO WAV data to a temporary .wav file on disk
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(st.session_state.audio_data_for_processing.read())
                    current_wav_temp_path = temp_file.name

                audio_np = whisper.load_audio(current_wav_temp_path)
                result = whisper_model.transcribe(audio_np)
                transcript = result['text']
                st.session_state.transcript_text = transcript # Store in session state
                st.success("Transcription Complete!")
                st.subheader("📝 Transcript:")
                st.code(transcript)

                st.download_button(
                    label="📥 Download Transcript",
                    data=st.session_state.transcript_text,
                    file_name="transcript.txt",
                    mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"Transcription failed: {e}. Please check the audio file and ensure Whisper model loaded correctly.")
            finally:
                if current_wav_temp_path and os.path.exists(current_wav_temp_path):
                    os.remove(current_wav_temp_path) # Clean up temp WAV file

        # --- Summarize ---
        if st.session_state.transcript_text:
            with st.spinner("Generating summary..."):
                try:
                    summary_result = summarizer_pipeline(
                        st.session_state.transcript_text,
                        max_length=150,
                        min_length=40,
                        do_sample=False
                    )
                    summary = summary_result[0]['summary_text']
                    st.session_state.summary_text = summary # Store in session state
                    st.success("Summary Complete!")
                    st.subheader("📄 Summary:")
                    st.write(summary)

                    st.download_button(
                        label="📥 Download Summary",
                        data=st.session_state.summary_text,
                        file_name="summary.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Summarization failed: {e}. Try adjusting length parameters or using a shorter transcript.")

        else:
            st.warning("Skipping summarization as no transcript was generated.")

    else:
        st.warning("Please upload or record audio before processing.")


st.markdown("---")
st.caption(f"Whisper Model: {WHISPER_MODEL_NAME.capitalize()}, Summarization Model: {SUMMARIZATION_MODEL_NAME}")