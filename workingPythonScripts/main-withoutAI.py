from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from docx import Document


DATA_FOLDER = Path("../data")


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_csv(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe.to_string(index=False)


def read_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def read_docx(file_path):
    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"

    return text


def convert_file_to_text(file_path):

    extension = file_path.suffix.lower()

    if extension == ".txt":
        return read_txt(file_path)

    elif extension == ".csv":
        return read_csv(file_path)

    elif extension == ".pdf":
        return read_pdf(file_path)

    elif extension == ".docx":
        return read_docx(file_path)

    return None


def load_documents():

    documents = []

    for file_path in DATA_FOLDER.iterdir():

        if not file_path.is_file():
            continue

        text = convert_file_to_text(file_path)

        if text is None:
            continue

        chunks = split_into_chunks(text)

        for chunk in chunks:

            documents.append({
                "filename": file_path.name,
                "content": chunk
            })

    return documents


def search_documents(documents, query):

    results = []

    query_words = query.lower().split()

    for document in documents:

        content = document["content"].lower()

        matches = 0

        for word in query_words:

            if word in content:
                matches += 1

        if matches > 0:

            results.append({
                "filename": document["filename"],
                "content": document["content"],
                "score": matches
            })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results


def main():

    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    print()

    query = input("What do you want to search for? ")

    results = search_documents(documents, query)

    print()
    print("SEARCH RESULTS")
    print("==============")

    if not results:
        print("No matching documents found.")
        return

    for result in results:

        print()
        print("FILE:", result["filename"])
        print("----------------------")
        print(result["content"])

def split_into_chunks(text):

    lines = text.splitlines()

    chunks = []

    for line in lines:

        line = line.strip()

        if line:
            chunks.append(line)

    return chunks

if __name__ == "__main__":
    main()