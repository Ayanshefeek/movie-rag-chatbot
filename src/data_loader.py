from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import CSVLoader
import sys
import os
sys.path.append(os.path.dirname(__file__))
def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load ALL CSV files from the data directory.
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # Auto-detect all CSV files
    csv_files = list(data_path.glob("**/*.csv"))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")

    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(file_path=str(csv_file), encoding="utf-8")
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load {csv_file}: {e}")

    print(f"[DEBUG] Total documents loaded: {len(documents)}")
    return documents


# Usage
if __name__ == "__main__":
    docs = load_all_documents("../data")
    print(f"\nTotal documents loaded: {len(docs)}")

    if docs:
        first_doc = docs[0]
        print("\n--- Example Document (Row 1) ---")
        print(f"Type     : {type(first_doc)}")
        print(f"Content  : {first_doc.page_content}")
        print(f"Metadata : {first_doc.metadata}")