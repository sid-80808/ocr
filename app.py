
import time

import streamlit as st
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
from google import genai


# ---------------- PAGE CONFIGURATION ----------------

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


# ---------------- GEMINI SUMMARIZATION ----------------

def summarize_chunk(client, text):
    prompt = (
        "Summarize the following study notes clearly. "
        "Use headings and bullet points where useful. "
        "Preserve important facts, definitions, names, "
        "dates, and formulas. Do not invent information.\n\n"
        f"NOTES:\n{text}"
    )

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty summary."
                )

            return response.text.strip()

        except Exception as e:
            error_message = str(e).upper()

            temporary_error = any(
                code in error_message
                for code in [
                    "503",
                    "429",
                    "500",
                    "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED",
                    "INTERNAL"
                ]
            )

            if not temporary_error or attempt == 3:
                raise

            # Wait 4, 8, and 16 seconds before retrying.
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        "Gemini could not summarize the notes."
    )


def summarize_text(text, client):
    words = text.split()
    chunk_size = 2500

    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    summaries = []

    for index, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            st.write(
                f"Summarizing section {index} of {len(chunks)}..."
            )

        summaries.append(
            summarize_chunk(client, chunk)
        )

    # Combine section summaries if the input was long.
    if len(summaries) > 1:
        combined = "\n\n".join(summaries)
        return summarize_chunk(client, combined)

    return summaries[0]


# ---------------- USER INPUT ----------------

uploaded = st.file_uploader(
    "Upload an image of your notes",
    type=["png", "jpg", "jpeg"]
)

text_input = st.text_area(
    "Or paste your notes here",
    height=180,
    placeholder="Paste your notes here..."
)


# ---------------- IMAGE PREVIEW ----------------

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
        st.error(
            "Could not open the image. Please try another file."
        )
        st.stop()


# ---------------- PROCESS NOTES ----------------

if st.button(
    "✨ Extract and Summarize",
    type="primary"
):

    # Prefer pasted text when provided.
    if text_input.strip():
        text = text_input.strip()

    # Otherwise, extract text from the uploaded image.
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
        st.warning(
            "Please upload an image or paste your notes first."
        )
        st.stop()

    # Stop if no usable text was found.
    if not text:
        st.error(
            "No text detected. Try a clearer image "
            "or paste the text manually."
        )
        st.stop()

    # Display extracted or pasted text.
    st.subheader("📄 Extracted Text")

    st.text_area(
        "OCR output",
        value=text,
        height=220
    )

    # ---------------- CREATE SUMMARY ----------------

    st.subheader("✨ AI Summary")

    if len(text.split()) < 35:
        summary = text

        st.info(
            "These notes are short, so the extracted text "
            "is shown as the summary."
        )

    else:
        try:
            # Read the API key from Streamlit Secrets.
            api_key = st.secrets["GEMINI_API_KEY"]

            client = get_gemini_client(api_key)

            with st.spinner(
                "Gemini is summarizing your notes..."
            ):
                summary = summarize_text(text, client)

        except Exception as e:
            st.error(f"Summarization failed: {e}")

            st.info(
                "If this is a temporary Gemini server error, "
                "wait a little and try again."
            )

            st.stop()

    # ---------------- DISPLAY SUMMARY ----------------

    st.markdown(summary)

    # ---------------- DOWNLOAD FILES ----------------

    st.subheader("⬇️ Download Your Notes")

    st.download_button(
        label="Download Summary",
        data=summary,
        file_name="notes_summary.txt",
        mime="text/plain"
    )

    st.download_button(
        label="Download Extracted Text",
        data=text,
        file_name="extracted_notes.txt",
        mime="text/plain"
    )

    st.success("Your notes have been processed!")
