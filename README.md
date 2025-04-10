# Expense Analyzer (Agentic RAG)

## Overview

This project is a Python application designed to automatically analyze expense receipts against a company's expense policy. It utilizes a Retrieval-Augmented Generation (RAG) approach with AI language models to classify receipts and provide reasoning for compliance decisions. The application features a web interface built with Streamlit for easy interaction.

**Current Features:**

*   **Policy Management:** Upload company expense policies (PDF). Policies are indexed for efficient retrieval.
*   **Receipt Upload:** Upload receipts as PDF, TXT, JPG, or PNG files.
*   **Automated Analysis:**
    *   Extracts text from receipts (using OCR for images via Tesseract and GPT-4o Vision fallback).
    *   Detects potential non-receipt files based on heuristics.
    *   Retrieves relevant policy sections using a FAISS vector index.
    *   Uses an AI Language Model (currently configured for GPT-4o) to:
        *   Extract Merchant Name and Total Amount.
        *   Classify receipts as 'Approved', 'Rejected', or 'On Hold'.
        *   Provide reasoning for 'Rejected' or 'On Hold' statuses based on policy context.
    *   Detects and flags duplicate receipt submissions based on content hash.
*   **Persistent History:** Stores all analysis results (including merchant/amount) in a local SQLite database.
*   **Interactive History View:** Displays past analysis results in a paginated table (using AgGrid) with view links.
*   **Receipt Viewer:** Allows viewing the original uploaded receipt file alongside its analysis details.
*   **Policy Q&A:** Allows asking natural language questions about the relevant policy directly from the receipt viewer.
*   **Web Admin Interface:** Built with Streamlit for managing policies and viewing results.

## Project Structure
expense-analyzer/
├── venv/ # Virtual environment (ignored by git)
├── uploads/ # Temporary storage for uploads (ignored by git)
├── receipt_storage/ # Persistent storage for original receipts (ignored by git)
├── policy_indices/ # Stores FAISS indexes and manifest (ignored by git)
├── .env # Stores API keys (ignored by git)
├── .gitignore # Specifies files/folders for git to ignore
├── expense_analysis.db # SQLite database for results (ignored by git)
├── main.py # Streamlit UI application, orchestration
├── analysis.py # Core analysis logic (OCR, RAG, LLM calls, parsing)
├── config.py # Configuration variables, constants, path setup
├── database.py # SQLite database interactions
├── indexing.py # Policy PDF processing and FAISS indexing
├── utils.py # Utility functions (e.g., hashing)
├── requirements.txt # Python package dependencies for pip
└── packages.txt # System-level dependencies for Streamlit Cloud (e.g., Tesseract)
└── README.md # This file


## Setup and Running Locally

1.  **Prerequisites:**
    *   Python 3.9+
    *   `pip` (Python package installer)
    *   Git
    *   Tesseract OCR Engine:
        *   **macOS:** `brew install tesseract tesseract-lang`
        *   **(Other OS):** Follow official Tesseract installation guides.

2.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd expense-analyzer
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On macOS/Linux
    # venv\Scripts\activate # On Windows
    ```

4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the root directory.
    *   Add your OpenAI API key:
        ```
        OPENAI_API_KEY="your_openai_api_key_here"
        ```

6.  **Run the Streamlit App:**
    ```bash
    streamlit run main.py
    ```

7.  **Access the App:** Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Deployment (Streamlit Community Cloud)

This app can be deployed to Streamlit Community Cloud. Ensure:

1.  Code is pushed to a GitHub/GitLab/Bitbucket repository.
2.  `requirements.txt` is up-to-date.
3.  `packages.txt` includes `tesseract-ocr` and necessary language packs (e.g., `tesseract-ocr-eng`).
4.  The `OPENAI_API_KEY` is configured as a Secret in the Streamlit Cloud app settings.

**Note:** Due to the ephemeral filesystem on Streamlit Community Cloud's free tier, the SQLite database, saved receipts (`receipt_storage`), and policy indexes (`policy_indices`) will likely **not persist** between app restarts or sleeps. For persistent demos, consider alternative hosting or modifying the app to use cloud database/storage services.

## Future Enhancements (Roadmap)

*   Refine History/Pagination (Counts, Jump-to-Page, Filter/Sort)
*   Improve Duplicate Feedback UI
*   Error Handling & UX Polish (AgGrid Tuning)
*   Advanced Policy Management (View Snippets, Delete, Updates)
*   Email / Cloud Storage Integration
*   User Accounts / Multi-Tenancy

## Contributing

*(Add guidelines here if you plan for contributions later)*

## License

*(Specify license here if chosen, e.g., MIT License)*
