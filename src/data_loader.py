from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import CSVLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load CSV file from the data directory and convert to LangChain document structure.
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # CSV file
    csv_file = data_path / "imdb.csv"
    print(f"[DEBUG] Loading CSV: {csv_file}")
    try:
        loader = CSVLoader(file_path=str(csv_file), encoding="utf-8")
        loaded = loader.load()
        print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
        documents.extend(loaded)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    return documents


# Usage
if __name__ == "__main__":
    docs = load_all_documents("../data")
    print(f"\nTotal documents loaded: {len(docs)}")

    # Example: print the first document
    if docs:
        first_doc = docs[0]
        print("\n--- Example Document (Row 1) ---")
        print(f"Type     : {type(first_doc)}")
        print(f"Content  : {first_doc.page_content}")
        print(f"Metadata : {first_doc.metadata}")