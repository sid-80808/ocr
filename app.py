
import streamlit as st
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
from transformers import pipeline
import io

# Uncomment and edit this line if Tesseract isn't detected on Windows:
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

st.set_page_config(page_title="NoteSnap AI", page_icon="📝")

st.title("📝 NoteSnap AI")
st.write("Upload your notes, extract text, and generate a summary.")

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

uploaded = st.file_uploader(
    "Upload a photo of your notes",
    type=["png", "jpg", "jpeg"]
)

text_input = st.text_area(
    "Or paste your notes here",
    height=150
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded notes", width="stretch")

if st.button("Extract and summarize"):
    with st.spinner("Processing your notes..."):
        if text_input.strip():
            text = text_input.strip()
        elif uploaded:
            image = ImageOps.grayscale(image)
            image = ImageEnhance.Contrast(image).enhance(2)
            text = pytesseract.image_to_string(
                image, config="--psm 6"
            )
        else:
            st.warning("Upload an image or paste some text.")
            st.stop()

        if not text.strip():
            st.error("No text detected. Try a clearer image.")
            st.stop()

        st.subheader("Extracted text")
        st.text_area("OCR output", text, height=200)

        if len(text.split()) < 35:
            summary = text
            st.info("Notes are short, so the extracted text is shown as the summary.")
        else:
            summarizer = load_summarizer()

            # Split long notes into manageable chunks.
            words = text.split()
            chunks = [
                " ".join(words[i:i + 400])
                for i in range(0, len(words), 400)
            ]

            summaries = []
            for chunk in chunks:
                result = summarizer(
                    chunk,
                    max_length=120,
                    min_length=25,
                    do_sample=False,
                    truncation=True
                )
                summaries.append(result[0]["summary_text"])

            summary = "\n\n".join(summaries)

        st.subheader("✨ Summary")
        st.write(summary)

        st.download_button(
            "Download summary",
            data=summary,
            file_name="notes_summary.txt",
            mime="text/plain"
        )

        st.download_button(
            "Download extracted text",
            data=text,
            file_name="extracted_notes.txt",
            mime="text/plain"
        )
