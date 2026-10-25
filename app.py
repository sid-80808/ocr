
import streamlit as st
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
from google import genai

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="NoteSnap AI",
    page_icon="📝",
    layout="centered"
)

st.title("📝 NoteSnap AI")
st.write(
    "Upload a photo of your notes or paste text "
    "to extract, summarize, and download your notes."
)

# ---------------- GEMINI CLIENT ----------------

@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def summarize_chunk(client, text):
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=(
            "Summarize the following study notes clearly. "
            "Use headings and bullet points where useful. "
            "Preserve important facts, definitions, names, "
            "dates, and formulas. Do not invent information.\n\n"
            f"NOTES:\n{text}"
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty summary.")

    return response.text.strip()


def summarize_text(text, client):
    # Split long notes into manageable chunks.
    words = text.split()
    chunk_size = 2500

    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    summaries = []

    for chunk in chunks:
        summaries.append(summarize_chunk(client, chunk))

    # If there are several summaries, combine them.
    if len(summaries) > 1:
        combined = "\n\n".join(summaries)
        return summarize_chunk(client, combined)

    return summaries[0]


# ---------------- INPUT ----------------

uploaded = st.file_uploader(
    "Upload an image of your notes",
    type=["png", "jpg", "jpeg"]
)

text_input = st.text_area(
    "Or paste your notes here",
    height=180,
    placeholder="Paste your notes here..."
)

image = None

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(
            image,
            caption="Uploaded notes",
            width="stretch"
        )
    except Exception:
        st.error("Could not open the image. Try another file.")
        st.stop()


# ---------------- PROCESS NOTES ----------------

if st.button("✨ Extract and Summarize", type="primary"):

    # Get text from pasted input or OCR.
    if text_input.strip():
        text = text_input.strip()

    elif image is not None:
        with st.spinner("Extracting text from image..."):
            try:
                processed = ImageOps.grayscale(image)
                processed = ImageEnhance.Contrast(
                    processed
                ).enhance(2)

                text = pytesseract.image_to_string(
                    processed,
                    config="--psm 6"
                ).strip()

            except Exception as e:
                st.error(f"OCR failed: {e}")
                st.stop()

    else:
        st.warning("Upload an image or paste your notes first.")
        st.stop()

    if not text:
        st.error(
            "No text detected. Try a clearer image "
            "or paste the text manually."
        )
        st.stop()

    # Display extracted text.
    st.subheader("📄 Extracted Text")
    st.text_area(
        "OCR output",
        value=text,
        height=220
    )

    # Generate the summary.
    st.subheader("✨ AI Summary")

    if len(text.split()) < 35:
        summary = text
        st.info(
            "These notes are short, so the extracted text "
            "is shown as the summary."
        )

    else:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = get_gemini_client(api_key)

            with st.spinner("Gemini is summarizing your notes..."):
                summary = summarize_text(text, client)

        except Exception as e:
            st.error(f"Summarization failed: {e}")
            st.stop()

    st.markdown(summary)

    # Download results.
    st.subheader("⬇️ Download Your Notes")

    st.download_button(
        "Download Summary",
        data=summary,
        file_name="notes_summary.txt",
        mime="text/plain"
    )

    st.download_button(
        "Download Extracted Text",
        data=text,
        file_name="extracted_notes.txt",
        mime="text/plain"
    )

    st.success("Your notes have been processed!")
