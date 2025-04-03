import streamlit as st
import time
import PyPDF2
from io import BytesIO
from mistralai import Mistral

# --- Rate Limiter Decorator ---
def rate_limited(min_interval):
    def decorator(func):
        last_called = [0.0]
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

# --- Retrieve API Keys from Streamlit Secrets ---
# (In production on Streamlit Community Cloud or locally via .streamlit/secrets.toml)
api_key = st.secrets["MISTRAL_API_KEY"]

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# --- App UI ---
st.title("PDF OCR Converter")
st.write("Upload a PDF file and receive the OCR'd text as a TXT file.")

# File uploader widget for PDFs
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Clear previous results if a new file is uploaded
if uploaded_file is not None:
    if "prev_file" not in st.session_state or st.session_state.prev_file != uploaded_file.name:
        st.session_state.prev_file = uploaded_file.name
        st.session_state.ocr_result = None

if uploaded_file is not None:
    # Only process if not already cached in session state.
    if st.session_state.get("ocr_result") is None:
        pdf_bytes = uploaded_file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        num_pages = len(pdf_reader.pages)
        st.write(f"Processing {num_pages} pages...")

        combined_ocr_text = ""
        progress_bar = st.progress(0)

        @rate_limited(2)  # Enforce a 2-second delay between API calls
        def process_page(page, page_number):
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(page)
            page_pdf_bytes = BytesIO()
            pdf_writer.write(page_pdf_bytes)
            page_pdf_bytes.seek(0)

            # Upload page for OCR processing
            uploaded_page = client.files.upload(
                file={
                    "file_name": f"page_{page_number}.pdf",
                    "content": page_pdf_bytes.getvalue(),
                },
                purpose="ocr"
            )
            signed_page_url = client.files.get_signed_url(file_id=uploaded_page.id)

            # Process OCR for this page
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_page_url.url
                }
            )
            return ocr_response

        # Process each page and accumulate OCR text.
        for i, page in enumerate(pdf_reader.pages):
            ocr_page_response = process_page(page, i + 1)
            combined_ocr_text += f"--- Page {i+1} ---\n{ocr_page_response}\n\n"
            progress_bar.progress((i + 1) / num_pages)

        st.text_area("OCR Text", combined_ocr_text, height=300)

        # Cache the OCR result in session state.
        st.session_state.ocr_result = combined_ocr_text

    # Provide a download button for the TXT file using the cached OCR result.
    st.download_button("Download Text", st.session_state.ocr_result, file_name="output.txt")

    # Halt further execution to prevent re-processing on re-runs.
    st.stop()
