# analysis.py
import os
import traceback
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompt_values import PromptValue
import re
from PIL import Image
import pytesseract
import base64

import config
from utils import calculate_hash
from database import check_duplicate, save_analysis_result

# --- Constants ---
MIN_RECEIPT_TEXT_LENGTH = 20
REQUIRED_KEYWORD_COUNT = 2 # Lowered threshold
RECEIPT_KEYWORDS = [
    'total', 'amount', 'invoice', 'receipt', 'merchant', 'tax', 'vat',
    'subtotal', 'cash', 'card', 'credit', 'payment', 'change', 'item',
    'price', 'qty', 'date', 'time', 'gst', 'hst', 'pst', 'paid', 'due',
    r'\$', r'€', r'£'
]
KEYWORD_PATTERNS = [re.compile(r'\b' + keyword + r'\b', re.IGNORECASE) for keyword in RECEIPT_KEYWORDS]

# --- Pydantic Models ---
class ReceiptExtractionResult(BaseModel):
    merchant_name: str = Field(description="The name of the merchant or vendor on the receipt. If unclear, state 'Unknown'.")
    total_amount: str = Field(description="The final total amount paid on the receipt, including currency symbol if possible. If unclear, state 'Unknown'.")

# --- Text Extraction Function --- <<< FULL BODY PRESENT >>>
def extract_text_from_file(uploaded_file) -> str:
    """Extracts text using Tesseract (with preprocessing for images)."""
    file_name = uploaded_file.name; file_type = uploaded_file.type; temp_file_path = os.path.join(config.UPLOAD_FOLDER, f"temp_{file_name}"); extracted_text = ""
    try:
        file_bytes = uploaded_file.getvalue();
        if not file_bytes: return ""
        with open(temp_file_path, "wb") as f: f.write(file_bytes)
        if file_type == "application/pdf":
            try: loader = PyPDFLoader(temp_file_path); pages = loader.load()
            except Exception as pdf_e: st.error(f"PDF Error {file_name}: {pdf_e}"); extracted_text = ""
            else:
                 if pages: extracted_text = "\n".join([doc.page_content for doc in pages if hasattr(doc, 'page_content')])
        elif file_type == "text/plain":
            try: loader = TextLoader(temp_file_path, encoding='utf-8'); docs = loader.load()
            except Exception as txt_e: st.error(f"TXT Error {file_name}: {txt_e}"); extracted_text = ""
            else:
                 if docs: extracted_text = docs[0].page_content
        elif file_type in ["image/jpeg", "image/png", "image/jpg"]:
            try:
                img = Image.open(temp_file_path); img_gray = img.convert('L'); threshold = 150; img_bw = img_gray.point(lambda x: 0 if x < threshold else 255, '1');
                extracted_text = pytesseract.image_to_string(img_bw, config='--psm 6');
                if not extracted_text.strip(): st.warning(f"Tesseract OCR found no significant text in image: {file_name}")
            except ImportError: st.error("Pytesseract/Tesseract not installed/found."); extracted_text = ""
            except Exception as ocr_e: st.error(f"Tesseract OCR Error for {file_name}: {ocr_e}"); st.error(traceback.format_exc()); extracted_text = ""
        else: st.warning(f"Unsupported file type '{file_type}' for {file_name}."); extracted_text = ""
    except Exception as e: st.error(f"Extraction Error for {file_name}: {e}"); st.error(traceback.format_exc()); extracted_text = ""
    finally:
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError as e: st.warning(f"Could not remove temp file {temp_file_path}: {e}")
    return extracted_text.strip()
# --- <<< END Text Extraction >>> ---


# --- Function: GPT-4o Vision OCR --- <<< FULL BODY PRESENT >>>
def extract_text_with_gpt4o_vision(image_bytes: bytes, file_name: str) -> str:
    """Uses GPT-4o Vision API to extract text from an image."""
    st.write(f"Escalating to GPT-4o Vision for OCR on {file_name}...")
    extracted_text = ""
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        if not client.api_key: st.error("OpenAI API key missing for Vision call."); return ""
        response = client.chat.completions.create( model="gpt-4o", messages=[ { "role": "user", "content": [ {"type": "text", "text": "Extract all text content from this receipt image. Output only the extracted text."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} ] } ], max_tokens=2000 )
        if response.choices: extracted_text = response.choices[0].message.content or ""; st.success(f"Advanced OCR successful for {file_name}.")
        else: st.warning(f"GPT-4o Vision returned no text for {file_name}.")
    except openai.APIError as e: st.error(f"OpenAI API Error (Vision OCR) for {file_name}: {e}"); st.error(traceback.format_exc())
    except Exception as e: st.error(f"Unexpected error (Vision OCR) for {file_name}: {e}"); st.error(traceback.format_exc())
    return extracted_text.strip()
# --- <<< END GPT-4o Vision Function >>> ---


# --- Q&A Function --- <<< FULL BODY PRESENT >>>
def answer_policy_question(question: str, policy_index: FAISS, llm: ChatOpenAI) -> str:
    """Answers a user question based on retrieved policy context."""
    if not question: return "Please enter a question."
    if not policy_index: return "Error: Policy index is not available for Q&A."
    if not llm: return "Error: LLM is not available for Q&A."
    try:
        qa_retriever = policy_index.as_retriever(search_kwargs={"k": 4})
        qa_template = """You are an AI assistant answering questions about a company expense policy based ONLY on the provided context... [rest of Q&A prompt] ... Answer:"""
        qa_prompt = ChatPromptTemplate.from_template(qa_template)
        qa_chain = ( {"context": qa_retriever, "question": RunnablePassthrough()} | qa_prompt | llm | StrOutputParser() )
        answer = qa_chain.invoke(question)
        return answer
    except Exception as e: st.error(f"Error during Q&A processing: {e}"); st.error(traceback.format_exc()); return "Sorry, an error occurred while trying to answer the question."
# --- <<< END Q&A Function >>> ---


# --- Receipt Processing Function (Multi-Step) --- <<< FULL BODY PRESENT >>>
def process_receipts(receipt_files: list, policy_index: FAISS, llm: ChatOpenAI, selected_policy_name: str) -> list:
    """Processes receipts via multi-step: DB check, OCR/Heuristics/Escalate, Extract, Classify, Save."""
    session_results = []
    if not llm: st.error("LLM invalid."); return []

    # --- Step 1 Setup: Extraction Chain ---
    extract_parser = JsonOutputParser(pydantic_object=ReceiptExtractionResult)
    extract_template = """
    You are an AI assistant tasked ONLY with extracting specific fields from expense receipt text.
    Analyze the provided receipt text and extract ONLY the merchant name and total amount paid.

    Expense Receipt Details:
    {receipt}

    Extraction Instructions:
    1. Extract the Merchant Name. If not found, output "Unknown".
    2. Extract the Total Amount Paid (include currency symbol if possible). If not found, output "Unknown".

    {format_instructions}

    Extract the information now and provide ONLY the JSON output. No other text.
    """
    extract_prompt = ChatPromptTemplate.from_template(
        extract_template, partial_variables={"format_instructions": extract_parser.get_format_instructions()}
    )
    extraction_chain = ( extract_prompt | llm | extract_parser )

    # --- Step 2 Setup: Classification Chain ---
    classify_parser = StrOutputParser()
    classify_template = """
    You are an AI expense analysis assistant. Your task is to analyze an expense receipt based ONLY on the provided company policy context and classify it.

    Policy Rules (context):
    {context}

    Expense Receipt Text:
    {receipt}

    Analysis Instructions:
    1. Review the receipt text against the policy context. Assume the expense purpose is valid.
    2. Prioritize specific rules (e.g., meal limits, forbidden items like alcohol) over general rules (e.g., 'personal items'). If an item like coffee or a simple meal appears and isn't explicitly forbidden, evaluate it against applicable limits (like meal allowance) rather than rejecting it solely based on a general 'personal items' clause.
    3. Classify into ONE category: 'Approved', 'Rejected', or 'On Hold'.
    4. Provide reasoning ONLY if status is 'Rejected' or 'On Hold', otherwise provide NO reasoning.
    5. Only reject based on clear, explicit policy violations found in the context provided.

    Output Format Instructions:
    ***VERY IMPORTANT: Format output as exactly two lines:***
    Status: [Approved/Rejected/On Hold]
    Reasoning: [Your reasoning here, or leave empty if Approved]
    ***Do NOT add any other text before or after these two lines.***

    Analyze the receipt now.
    """
    classify_prompt = ChatPromptTemplate.from_template(classify_template)
    # RAG chain specifically for classification (Requires policy_index)
    rag_chain_classify = None # Initialize
    if policy_index: # Check if index is valid before creating retriever
        retriever = policy_index.as_retriever(search_kwargs={"k": 3})
        rag_chain_classify = (
            {"context": retriever, "receipt": RunnablePassthrough()}
            | classify_prompt
            | llm
            | classify_parser
        )
    else:
         # This case should ideally not be reached if main.py checks index validity
         # but handle it defensively. Classification will fail later.
         st.warning("Policy index missing, classification step will likely fail.")


    total_receipts = len(receipt_files);

    for i, receipt_file in enumerate(receipt_files):
        current_file_name = receipt_file.name
        receipt_content = None
        stored_file_path = None
        merchant, amount, status, reasoning = "Error", "Error", "Error", "Initial Processing Error"

        try: # Outer try for the whole file processing
            # --- Duplicate Check & File Save ---
            try: receipt_content = receipt_file.getvalue()
            except Exception as e: raise ValueError(f"Read buffer error: {e}")
            if not receipt_content: raise ValueError("File is empty.")
            receipt_hash = calculate_hash(receipt_content)
            is_duplicate, existing_status, existing_timestamp = check_duplicate(receipt_hash)
            if is_duplicate:
                reasoning = f"Duplicate. Processed {existing_timestamp}. Status: {existing_status}"
                session_results.append({"Receipt Name": current_file_name, "Status": "Duplicate", "Reasoning": reasoning, "Merchant Name": "N/A", "Total Amount": "N/A"});
                continue;

            try: # Save original file
                _, extension = os.path.splitext(current_file_name); extension = extension if extension else f".{receipt_file.type.split('/')[-1]}" if receipt_file.type and '/' in receipt_file.type else ".bin"; safe_filename = f"{receipt_hash}{extension}"; stored_file_path = os.path.join(config.RECEIPT_STORAGE_DIR, safe_filename)
                with open(stored_file_path, "wb") as f_save: f_save.write(receipt_content)
            except Exception as save_e: st.error(f"Failed to save file for {current_file_name}: {save_e}"); stored_file_path = None

            # --- Text Extraction & Heuristics ---
            receipt_text = extract_text_from_file(receipt_file)
            original_extraction_failed = not receipt_text
            text_length = len(receipt_text); found_keywords = 0
            if text_length > 0:
                 for pattern in KEYWORD_PATTERNS:
                      if pattern.search(receipt_text): found_keywords += 1
            tesseract_passed_heuristics = (text_length >= MIN_RECEIPT_TEXT_LENGTH and found_keywords >= REQUIRED_KEYWORD_COUNT)

            # --- Escalation Logic ---
            proceed_to_analysis = False; status_override, reasoning_override = None, None
            if tesseract_passed_heuristics: proceed_to_analysis = True
            elif receipt_file.type in ["image/jpeg", "image/png", "image/jpg"]: # Escalate image
                st.warning(f"Low quality detected for {current_file_name}. Attempting advanced OCR...")
                receipt_text_vision = extract_text_with_gpt4o_vision(receipt_content, current_file_name)
                if not receipt_text_vision: status_override, reasoning_override = "Error", "Failed text extraction (Tesseract & Vision)."
                else: # Re-check heuristics on vision text
                     text_length_vision = len(receipt_text_vision); found_keywords_vision = 0
                     for pattern in KEYWORD_PATTERNS:
                         if pattern.search(receipt_text_vision): found_keywords_vision += 1
                     vision_passed_heuristics = (text_length_vision >= MIN_RECEIPT_TEXT_LENGTH and found_keywords_vision >= REQUIRED_KEYWORD_COUNT)
                     if vision_passed_heuristics: st.success(f"Advanced OCR successful."); receipt_text = receipt_text_vision; proceed_to_analysis = True
                     else: status_override, reasoning_override = "Error", f"File not likely receipt (Adv. OCR - Len: {text_length_vision}, Keywords: {found_keywords_vision})."; st.error(f"'{current_file_name}' flagged non-receipt post-Adv. OCR.")
            else: # Failed PDF/TXT or Tesseract on image
                status_override = "Error"; reasoning_override = f"File not likely a receipt (Initial Check - Len: {text_length}, Keywords: {found_keywords}).";
                if original_extraction_failed: reasoning_override = "Failed text extraction (or empty file)."
                st.warning(f"'{current_file_name}' flagged as non-receipt/unreadable.")

            if not proceed_to_analysis:
                raise ValueError(reasoning_override) # Use reasoning from failed check

            # --- If text is valid, proceed with multi-step analysis ---

            # --- STEP 1: Extraction ---
            st.write(f"Step 1: Extracting data for {current_file_name}...")
            extraction_input_dict = { "receipt": receipt_text }
            extracted_data = extraction_chain.invoke(extraction_input_dict)
            merchant = extracted_data.get("merchant_name", "Unknown")
            amount = extracted_data.get("total_amount", "Unknown")
            st.write(f" -> Extracted: Merchant='{merchant}', Amount='{amount}'")

            # --- STEP 2: RAG Classification ---
            if not rag_chain_classify: # Check if chain was created (depends on policy_index)
                 raise ValueError("Classification chain not initialized (Policy Index missing?).")

            st.write(f"Step 2: Classifying {current_file_name} against policy...")
            classification_response_str = rag_chain_classify.invoke(receipt_text)
            # st.write(f" -> Classification Response: {repr(classification_response_str)}") # Optional Debug

            # --- Parse Classification Response ---
            status = "Error"; reasoning = "Failed: Classification Parse Error."
            if classification_response_str:
                response_lines = classification_response_str.strip().split('\n')
                if len(response_lines) >= 1 and response_lines[0].startswith("Status:"):
                    status_candidate = response_lines[0].replace("Status:", "").strip();
                    if status_candidate in ["Approved", "Rejected", "On Hold"]:
                        status = status_candidate;
                        if status != "Approved" and len(response_lines) >= 2 and response_lines[1].startswith("Reasoning:"): reasoning = response_lines[1].replace("Reasoning:", "").strip()
                        elif status != "Approved" and (len(response_lines) < 2 or not response_lines[1].strip()): reasoning = "Reasoning required but not provided by LLM."
                        elif status == "Approved": reasoning = ""
                        elif status != "Approved": reasoning = f"Classification Parse Error: Expected 'Reasoning:', got line 2: {response_lines[1][:50]}..."
                    else: reasoning = f"Failed: Invalid Status value '{status_candidate}' from classification."
                else: reasoning = f"Failed: Bad Status line from classification. Got: {response_lines[0][:100] if response_lines else 'EMPTY'}"
            else: reasoning = "Failed: Empty response from classification LLM."
            # --- End Classification Parsing ---
            st.write(f" -> Final Result: Status='{status}', Merchant='{merchant}', Amount='{amount}'")

        # --- Outer Exception Handling ---
        except Exception as e:
            st.error(f"Error processing {current_file_name}: {e}")
            st.error(traceback.format_exc());
            # Use specific error if available from heuristics/extraction, else generic
            if status == "Error" and reasoning != "Initial Processing Error": pass
            else: status, reasoning = "Error", f"Unexpected error: {str(e)}"
            merchant, amount = "Error", "Error"

        # --- Save final result ---
        save_analysis_result(current_file_name, receipt_hash, selected_policy_name, status, reasoning, merchant, amount, stored_file_path);
        session_results.append({
            "Receipt Name": current_file_name, "Status": status, "Reasoning": reasoning,
            "Merchant Name": merchant, "Total Amount": amount
            });

    return session_results
# --- <<< END Multi-Step Function >>> ---