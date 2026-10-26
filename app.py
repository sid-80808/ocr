
import time

import streamlit as st
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
from google import genai


#PAGE_CONFIGURATION 

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


#GEMINI

@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


#GEMINI_SUMMARIZATION

def summarize_chunk(client, text):
    prompt = (
        "Summarize the following study notes clearly. "
        "Use headings and bullet points where useful. "
        "Preserve important facts, definitions, names, "
        "dates, and formulas. Do not invent information.\n\n"
        f"NOTES:\n{text}"
    )

    for attempt in range(3):
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

            # Do not retry quota or rate-limit errors.
            if any(code in error_message for code in [
                "429",
                "RESOURCE_EXHAUSTED",
                "QUOTA EXCEEDED"
            ]):
                raise RuntimeError(
                    "Gemini quota or rate limit reached. "
                    "Check https://ai.dev/rate-limit "
                    "or wait until your quota resets."
                ) from e

            # Retry temporary server errors.
            temporary_error = any(
                code in error_message
                for code in [
                    "503",
                    "500",
                    "UNAVAILABLE",
                    "INTERNAL"
                ]
            )

            if not temporary_error or attempt == 2:
                raise

            # Wait 4 seconds, then 8 seconds.
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        "Gemini could not summarize the notes."
    )


def summarize_text(text, client):
    words = text.split()
    chunk_size = 10000

    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

    summaries = []

    for index, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            st.write(
                f"Summarizing section {index} "
                f"of {len(chunks)}..."
            )

        summaries.append(
            summarize_chunk(client, chunk)
        )

    # Combine summaries locally without another API request.
    return "\n\n".join(summaries)


#USER_INPUT

uploaded = st.file_uploader(
    "Upload an image of your notes",
    type=["png", "jpg", "jpeg"]
)

text_input = st.text_area(
    "Or paste your notes here",
    height=180,
    placeholder="Paste your notes here..."
)


#IMAGE_PREVIEW 

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


#PROCESS_NOTES

if st.button(
    "✨ Extract and Summarize",
    type="primary"
):

    #Use pasted text if provided.
    if text_input.strip():
        text = text_input.strip()

    #Otherwise, extract text from the uploaded image.
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

    # Check whether text was extracted.
    if not text:
        st.error(
            "No text detected. Try a clearer image "
            "or paste the text manually."
        )
        st.stop()

    #DISPLAY EXTRACTED TEXTs

    st.subheader("📄 Extracted Text")

    st.text_area(
        "OCR output",
        value=text,
        height=220
    )

    #CREATE_SUMMARY

    st.subheader("✨ AI Summary")

    if len(text.split()) < 35:
        summary = text

        st.info(
            "These notes are short, so the extracted text "
            "is shown as the summary."
        )

    else:
        try:
            # Read API key from Streamlit Secrets.
            api_key = st.secrets["GEMINI_API_KEY"]

            client = get_gemini_client(api_key)

            with st.spinner(
                "Gemini is summarizing your notes..."
            ):
                summary = summarize_text(text, client)

        except Exception as e:
            st.error(f"Summarization failed: {e}")

            st.info(
                "If your quota is exhausted, wait until it "
                "resets or check your available quota. "
                "Temporary server errors may resolve later."
            )
            st.stop()

    #DISPLAY_SUMMARY

    st.markdown(summary)

    #DOWNLOAD_FILES

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
