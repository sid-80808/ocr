
import streamlit as st
import pytesseract
import requests
from PIL import Image, ImageOps, ImageEnhance

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="NoteSnap AI",
    page_icon="📝",
    layout="centered"
)

st.title("📝 NoteSnap AI")
st.write(
    "Upload a photo of your notes or paste text to "
    "extract, summarize, and download your notes."
)

# ---------------- SUMMARIZATION FUNCTION ----------------

def summarize_text(text):
    token = st.secrets["HF_TOKEN"]

    url = (
        "https://router.huggingface.co/"
        "hf-inference/models/facebook/bart-large-cnn"
    )

    headers = {
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "inputs": text[:10000],
        "parameters": {
            "max_length": 150,
            "min_length": 30,
            "do_sample": False
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()
    result = response.json()

    if isinstance(result, list) and result:
        return result[0]["summary_text"]

    raise ValueError(f"Unexpected API response: {result}")


# ---------------- INPUT SECTION ----------------

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
        st.error("Could not open the image. Try another file.")
        st.stop()

# ---------------- PROCESS NOTES ----------------

if st.button("✨ Extract and Summarize", type="primary"):

    # Extract text from the image or use pasted text
    if text_input.strip():
        text = text_input.strip()

    elif image is not None:
        with st.spinner("Extracting text from image..."):
            try:
                processed_image = ImageOps.grayscale(image)
                processed_image = ImageEnhance.Contrast(
                    processed_image
                ).enhance(2)

                text = pytesseract.image_to_string(
                    processed_image,
                    config="--psm 6"
                ).strip()

            except Exception as e:
                st.error(f"OCR failed: {e}")
                st.stop()

    else:
        st.warning("Please upload an image or paste your notes.")
        st.stop()

    if not text:
        st.error(
            "No text detected. Try a clearer image "
            "or paste the text manually."
        )
        st.stop()

    # Display extracted text
    st.subheader("📄 Extracted Text")
    st.text_area(
        "OCR output",
        value=text,
        height=220
    )

    # Generate summary
    st.subheader("✨ Summary")

    if len(text.split()) < 35:
        summary = text
        st.info(
            "These notes are short, so the extracted "
            "text is shown as the summary."
        )

    else:
        with st.spinner("Generating your summary..."):
            try:
                summary = summarize_text(text)
                st.write(summary)

            except requests.exceptions.Timeout:
                st.error(
                    "The summarization service timed out. "
                    "Please try again."
                )
                st.stop()

            except requests.exceptions.HTTPError as e:
                st.error(
                    f"Hugging Face API error: {e.response.status_code}. "
                    "Check your token, model availability, and API access."
                )
                st.stop()

            except Exception as e:
                st.error(f"Summarization failed: {e}")
                st.stop()

    if len(text.split()) < 35:
        st.write(summary)

    # ---------------- DOWNLOAD SECTION ----------------

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
