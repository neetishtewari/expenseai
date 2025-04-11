# main.py
import streamlit as st
import os
import json
import pandas as pd
import datetime
import traceback
import sqlite3
import base64

# Import AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

# Import from our refactored modules
import config
from database import init_db, get_historical_results, get_result_by_filename
from indexing import create_and_save_policy_index
from analysis import process_receipts, answer_policy_question

# Import LangChain components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# --- Manifest Helper Functions ---
def load_manifest():
    """Loads the policy index manifest file."""
    if os.path.exists(config.MANIFEST_FILE):
        try:
            with open(config.MANIFEST_FILE, 'r') as f: content = f.read(); return json.loads(content) if content else {}
        except json.JSONDecodeError: st.warning(f"Manifest file invalid JSON."); return {}
        except Exception as e: st.error(f"Manifest Load Error: {e}"); return {}
    return {}

def save_manifest(manifest_data):
    """Saves the policy index manifest file."""
    try:
        with open(config.MANIFEST_FILE, 'w') as f: json.dump(manifest_data, f, indent=4)
    except Exception as e: st.error(f"Manifest Save Error: {e}")

# --- Caching Core Resources ---
@st.cache_resource
def get_embeddings_model():
    """Loads and caches the embedding model."""
    try: return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    except Exception as e: st.error(f"Embedding Load Error: {e}"); print(f"ERROR embeddings: {e}\n{traceback.format_exc()}"); st.stop()

@st.cache_resource
def get_llm():
    """Loads and caches the LLM."""
    if not config.OPENAI_API_KEY: st.error("OpenAI API key missing."); st.stop();
    try: return ChatOpenAI(model_name=config.LLM_MODEL_NAME, temperature=0.0, api_key=config.OPENAI_API_KEY)
    except Exception as e: st.error(f"LLM Init Error: {e}"); print(f"ERROR LLM: {e}\n{traceback.format_exc()}"); st.stop()

# --- Display Logic ---
def display_results(results: list):
    """Displays results from the current session."""
    if results is None: st.info("Waiting for analysis results..."); return
    if not results: st.info("No new/updated results processed in this batch."); return
    st.subheader("Current Batch Analysis Results")
    if not isinstance(results, list) or (results and not isinstance(results[0], dict)): st.error("Internal Error: Invalid format for results."); print(f"ERROR: Invalid results type: {type(results)}"); return
    try:
        df = pd.DataFrame(results)
        if not df.empty:
            cols_order = ["Receipt Name", "Merchant Name", "Total Amount", "Status", "Reasoning"]; cols_to_show = [col for col in cols_order if col in df.columns];
            if cols_to_show: df_display = df[cols_to_show].copy(); st.dataframe(df_display, use_container_width=True)
            else: st.warning("Batch result structure missing columns."); st.dataframe(df, use_container_width=True)
        else: st.info("Result list was empty for this batch.")
    except Exception as e: st.error(f"Error displaying results DF: {e}"); print(f"ERROR creating/displaying results DF: {e}\n{traceback.format_exc()}"); st.write("Raw results:", results)

# --- Helper to display PDF ---
def display_pdf(file_path):
    """Displays PDF file in Streamlit using Base64 embedding in iframe."""
    try:
        with open(file_path, "rb") as f: base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" style="height:70vh;" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError: st.error(f"Error: Could not find stored PDF file at {file_path}")
    except Exception as e: st.error(f"Error displaying PDF: {e}"); st.error(traceback.format_exc())

# --- Main Streamlit App ---
st.set_page_config(layout="wide")
st.title("📄 Expense AI") # Version bump
st.markdown("Verify that the expenses comply with the company policies using Agentic RAG.")
init_db()

# --- Initialize Session State ---
if 'page_num' not in st.session_state: st.session_state.page_num = 1
if 'results' not in st.session_state: st.session_state.results = None
if 'qa_answer' not in st.session_state: st.session_state.qa_answer = ""
if 'qa_processing' not in st.session_state: st.session_state.qa_processing = False
if 'qa_target_filename' not in st.session_state: st.session_state.qa_target_filename = None

# --- Load Core Resources ---
with st.spinner("Loading AI models..."):
    embeddings = get_embeddings_model()
    llm = get_llm()
policy_manifest = load_manifest()

# --- Sidebar ---
st.sidebar.title("🏛️ Policy Management");
st.sidebar.markdown("Add or update company expense policies."); new_policy_file = st.sidebar.file_uploader("Upload Policy PDF", type=["pdf"], key="policy_manager_uploader"); new_policy_name = st.sidebar.text_input("Policy Name", placeholder="e.g., Global T&E v1.1", key="policy_name_input");
if st.sidebar.button("Add Policy & Index", key="add_policy_button", disabled=(not new_policy_file or not new_policy_name)):
    temp_policy_path = os.path.join(config.UPLOAD_FOLDER, new_policy_file.name);
    try:
        with open(temp_policy_path, "wb") as f: f.write(new_policy_file.getvalue());
        with st.spinner(f"Processing '{new_policy_name}'..."): success, msg_path = create_and_save_policy_index(temp_policy_path, new_policy_name, embeddings);
        if success: policy_manifest[new_policy_name] = msg_path; save_manifest(policy_manifest); st.sidebar.success(f"Policy '{new_policy_name}' added!"); st.rerun();
        else: st.sidebar.error(f"Failed: {msg_path}");
    except Exception as e: st.sidebar.error(f"Policy Add Error: {e}"); st.sidebar.error(traceback.format_exc())
    finally:
         if os.path.exists(temp_policy_path):
             try: os.remove(temp_policy_path)
             except OSError as e: st.warning(f"Could not remove temp policy file {temp_policy_path}: {e}")
st.sidebar.markdown("---"); st.sidebar.subheader("Indexed Policies");
if not policy_manifest: st.sidebar.info("No policies indexed yet.");
else:
    for name, path in policy_manifest.items(): st.sidebar.markdown(f"- **{name}** (`{os.path.basename(path) if path and isinstance(path, str) else 'N/A'}`)")

# --- Main Area ---
st.header("🧾 Receipt Analysis");
if not policy_manifest: st.warning("Please add a policy via the sidebar.");
else:
    policy_names = list(policy_manifest.keys());
    if not policy_names: st.warning("No policies found in manifest.");
    else:
        selected_policy_name = st.selectbox("Select Policy:", options=policy_names, index=len(policy_names)-1, key="policy_selector");
        st.subheader("1. Upload Receipts"); st.caption("Files remain until cleared manually or new files are uploaded.")
        receipt_files = st.file_uploader("Upload Receipts:", type=["pdf", "txt", "jpg", "jpeg", "png"], accept_multiple_files=True, key="receipt_uploader_main")
        st.subheader("2. Process Files"); analyze_button_disabled = (not selected_policy_name or not receipt_files); analyze_button = st.button("Analyze Expenses", key="analyze_button_main", disabled=analyze_button_disabled);

        if analyze_button:
            st.session_state.results = None; selected_index_path = policy_manifest.get(selected_policy_name);
            if not receipt_files: st.warning("No receipts uploaded.");
            elif not selected_index_path or not os.path.exists(selected_index_path): st.error(f"Index path invalid.");
            else:
                loaded_policy_index = None; status_placeholder = st.empty(); analysis_successful = False
                try:
                    status_placeholder.text(f"Loading policy '{selected_policy_name}'...")
                    with st.spinner("Loading index..."):
                        if not embeddings: raise ValueError("Embeddings missing.")
                        try: loaded_policy_index = FAISS.load_local(selected_index_path, embeddings, allow_dangerous_deserialization=True)
                        except Exception as faiss_e: st.error(f"FAISS Load Error: {faiss_e}"); raise
                    if not loaded_policy_index: raise ValueError("FAISS loaded as None.")
                    status_placeholder.success(f"Policy index loaded.")

                    status_placeholder.text(f"Analyzing {len(receipt_files)} receipts...")
                    # Ensure analysis.py is v8.0+ (Multi-Step)
                    results = process_receipts(receipt_files, loaded_policy_index, llm, selected_policy_name)
                    st.session_state.results = results
                    status_placeholder.success(f"Analysis complete.")
                    analysis_successful = True
                except Exception as e: status_placeholder.error(f"Workflow Error: {e}"); st.error(traceback.format_exc()); st.session_state.results = []
                finally: pass

                st.session_state.page_num = 1;
                st.session_state.qa_target_filename = None
                st.session_state.qa_answer = ""
                st.rerun();

# --- Display Current Batch Results ---
if st.session_state.results is not None: display_results(st.session_state.results)

st.markdown("---")

# --- Analysis History Section ---
st.header("📊 Analysis History")
current_page = st.session_state.get('page_num', 1); offset = (current_page - 1) * config.RESULTS_PER_PAGE
history_data = get_historical_results(limit=config.RESULTS_PER_PAGE, offset=offset); selected_row_data_from_grid = None

if not history_data: st.info("No historical analysis data found.")
else:
    st.write(f"Displaying Page {current_page}...")
    history_df = pd.DataFrame(history_data); display_columns = { "analysis_timestamp": "Date", "receipt_filename": "Receipt Name", "merchant_name": "Merchant", "total_amount": "Amount", "status": "Status", "reasoning": "Reasoning", "stored_file_path": "File Path"};
    cols_to_display = [col for col in display_columns.keys() if col in history_df.columns]; history_df_display = pd.DataFrame();
    if cols_to_display:
        history_df_display = history_df[cols_to_display].copy(); history_df_display.rename(columns=display_columns, inplace=True)
        if "Date" in history_df_display.columns: history_df_display["Date"] = pd.to_datetime(history_df_display["Date"], errors='coerce').dt.strftime('%Y-%m-%d').fillna('N/A')
        cols_to_fill = ["Merchant", "Amount", "Reasoning", "File Path"];
        for col in cols_to_fill:
             if col in history_df_display.columns: history_df_display[col] = history_df_display[col].fillna("N/A")

        df_for_aggrid = history_df_display.drop(columns=['File Path'], errors='ignore')

        # --- AgGrid Configuration ---
        gb = GridOptionsBuilder.from_dataframe(df_for_aggrid)
        # Configure columns with flex/width
        gb.configure_column("Receipt Name", flex=2, minWidth=150, sortable=True, filterable=True)
        gb.configure_column("Merchant", flex=1, minWidth=120, sortable=True, filterable=True)
        gb.configure_column("Reasoning", flex=3, minWidth=200, sortable=True, filterable=True)
        gb.configure_column("Date", width=110, flex=0, sortable=True, filterable=True)
        gb.configure_column("Amount", width=100, flex=0, sortable=True, filterable=True)
        gb.configure_column("Status", width=100, flex=0, sortable=True, filterable=True)
        # View Link Renderer
        view_link_renderer = JsCode(""" class LinkCellRenderer { init(params) { this.params = params; this.eGui = document.createElement('div'); this.eGui.innerHTML = `<a href="#" onclick="return false;" style="text-decoration: underline; cursor: pointer;">View</a>`; this.eLink = this.eGui.querySelector('a'); this.linkClickListener = this.linkClickListener.bind(this); this.eLink.addEventListener('click', this.linkClickListener); } linkClickListener(event) { this.params.api.selectIndex(this.params.rowIndex, false, false); } getGui() { return this.eGui; } destroy() { if (this.eLink) { this.eLink.removeEventListener('click', this.linkClickListener); } } } """)
        gb.configure_column("view_action", headerName="Action", cellRenderer=view_link_renderer, width=70, flex=0, minWidth=70, lockPosition='right', suppressMenu=True, filter=False, sortable=False)
        # Other options
        gb.configure_grid_options(rowHeight=35); gb.configure_selection('single', use_checkbox=False);
        gridOptions = gb.build()

        # --- Display AgGrid ---
        grid_response = AgGrid( df_for_aggrid, gridOptions=gridOptions, data_return_mode=DataReturnMode.AS_INPUT, update_mode=GridUpdateMode.SELECTION_CHANGED, allow_unsafe_jscode=True, height=400, width='100%', reload_data=True, key='history_grid', theme='alpine' )
        # --- Get Selected Row ---
        selected_rows_df = grid_response.get('selected_rows', None)
        if selected_rows_df is not None and not selected_rows_df.empty: selected_row_data_from_grid = selected_rows_df.iloc[0].to_dict()
        else: selected_row_data_from_grid = None
    else: st.warning("Could not prepare history columns.")

    # --- Pagination Controls ---
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=(current_page <= 1), key="prev_page_hist"): st.session_state.page_num -= 1; st.rerun()
    with col3:
        if st.button("Next ➡️", disabled=(len(history_data) < config.RESULTS_PER_PAGE), key="next_page_hist"): st.session_state.page_num += 1; st.rerun()
    with col2: st.write(f"Page {current_page}")

st.markdown("---")

# --- Receipt Viewer Section ---
if selected_row_data_from_grid:
    selected_filename = selected_row_data_from_grid.get("Receipt Name", None)
    if not selected_filename: st.error("Could not determine filename from selected row.")
    else:
        st.header(f"📄 Viewing: {selected_filename}")
        receipt_data = get_result_by_filename(selected_filename) # Fetch full data
        if not receipt_data: st.error(f"Could not retrieve full data for {selected_filename}.")
        else:
            stored_path = receipt_data.get("stored_file_path"); policy_name_used = receipt_data.get("policy_name_used", None); file_ext = os.path.splitext(stored_path)[1].lower() if stored_path and isinstance(stored_path, str) else None
            col_view1, col_view2 = st.columns([1, 1])
            with col_view1: # File Viewer
                st.subheader("Original File");
                # ---<<< CORRECTED ELIF SYNTAX >>>---
                if not stored_path:
                     st.warning("File path not found in record.") # Changed message
                elif not os.path.exists(stored_path): # Corrected indentation
                     st.error(f"Stored file not found at path: {stored_path}")
                elif file_ext == ".pdf": # Corrected indentation
                     display_pdf(stored_path)
                elif file_ext in [".jpg", ".jpeg", ".png"]: # Corrected indentation
                     st.image(stored_path)
                elif file_ext == ".txt": # Corrected indentation
                     try:
                         with open(stored_path, 'r', encoding='utf-8') as f: txt_content = f.read()
                         st.text_area("TXT Content", txt_content, height=500)
                     except Exception as e: st.error(f"Could not read TXT {stored_path}: {e}")
                else: # Corrected indentation
                     st.warning(f"Cannot display type: {file_ext}.")
                 # ---<<< END CORRECTION >>>---
            with col_view2: # Analysis Details + Q&A
                st.subheader("Analysis Details");
                ts = pd.to_datetime(receipt_data.get("analysis_timestamp")).strftime('%Y-%m-%d %H:%M:%S') if receipt_data.get("analysis_timestamp") else "N/A"; st.info(f"Analyzed on: {ts}"); st.info(f"Policy Used: {policy_name_used}")
                st.metric("Merchant", selected_row_data_from_grid.get("Merchant", "N/A")); st.metric("Total Amount", selected_row_data_from_grid.get("Amount", "N/A")); st.metric("Status", selected_row_data_from_grid.get("Status", "N/A"))
                reasoning = selected_row_data_from_grid.get("Reasoning", ""); status = selected_row_data_from_grid.get("Status")
                if reasoning and status not in ["Approved", "Duplicate", "N/A"]: st.subheader("Reasoning:"); st.warning(reasoning)
                elif status == "Duplicate": st.subheader("Details:"); st.info(reasoning)

                # --- Q&A Integration ---
                st.markdown("---"); st.subheader("❓ Ask About Policy")
                q_key = f"q_{selected_filename}";
                if st.session_state.qa_target_filename != selected_filename: st.session_state[q_key] = ""; st.session_state.qa_answer = ""; st.session_state.qa_processing = False; st.session_state.qa_target_filename = selected_filename
                user_question = st.text_input("Ask a question:", key=q_key)
                if st.button("Ask AI Assistant", key=f"ask_btn_{selected_filename}"):
                    current_input_value = st.session_state.get(q_key, "")
                    if current_input_value: st.session_state.qa_processing = True; st.session_state.qa_answer = "Thinking..."; st.rerun();
                    else: st.session_state.qa_answer = "Please enter question."; st.session_state.qa_processing = False; # Maybe rerun here too
                if st.session_state.qa_processing and st.session_state.qa_target_filename == selected_filename:
                    answer = "Error during Q&A."; current_question_to_process = st.session_state.get(q_key, "")
                    if current_question_to_process and policy_name_used and policy_manifest.get(policy_name_used):
                        policy_index_path = policy_manifest.get(policy_name_used)
                        if os.path.exists(policy_index_path):
                             try:
                                 with st.spinner("Loading policy & asking AI..."):
                                     qa_policy_index = FAISS.load_local(policy_index_path, embeddings, allow_dangerous_deserialization=True)
                                     if qa_policy_index: answer = answer_policy_question(current_question_to_process, qa_policy_index, llm)
                                     else: answer = "Error: Could not load policy index."
                             except Exception as qa_load_e: answer = f"Error loading/asking: {qa_load_e}"
                        else: answer = "Error: Policy index path not found."
                    elif not current_question_to_process: answer = "Internal Error: Question missing."
                    else: answer = "Error: Cannot answer (missing policy info)."
                    st.session_state.qa_answer = answer; st.session_state.qa_processing = False; st.rerun();
                if st.session_state.qa_answer and st.session_state.qa_target_filename == selected_filename:
                    if st.session_state.qa_answer == "Thinking...": st.info("Thinking...")
                    elif "Error" in st.session_state.qa_answer or "Please enter" in st.session_state.qa_answer: st.warning(st.session_state.qa_answer)
                    else: st.markdown("**Answer:**"); st.success(st.session_state.qa_answer)
                # --- END Q&A Integration ---

# --- Footer ---
st.markdown("---")
st.markdown("Analyzer v7.16 - Final Syntax Fix")