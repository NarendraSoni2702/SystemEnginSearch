from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from docx import Document
from openai import OpenAI


DATA_FOLDER = Path("../data")

client = OpenAI()


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

        documents.append({
            "filename": file_path.name,
            "content": text
        })

    return documents


def search_documents(documents, query):

    results = []

    query_words = query.lower().split()

    for document in documents:

        content = document["content"].lower()

        score = 0

        for word in query_words:

            if word in content:
                score += 1

        if score > 0:

            results.append({
                "filename": document["filename"],
                "content": document["content"],
                "score": score
            })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results


def ask_ai(question, search_results):

    context = ""

    for result in search_results:

        context += (
            f"\nSOURCE: {result['filename']}\n"
            f"{result['content']}\n"
        )

    prompt = f"""
You are an assistant that answers questions using the provided documents.

Answer the user's question using only the information in the documents.

If the answer cannot be found in the documents, say:
"I couldn't find that information in the documents."

Documents:
{context}

User question:
{question}
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return response.output_text


def main():

    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    print()

    question = input("Ask a question: ")

    search_results = search_documents(
        documents,
        question
    )

    if not search_results:

        print()
        print("No relevant information found.")

        return

    print()
    print("Sending relevant information to AI...")

    answer = ask_ai(
        question,
        search_results
    )

    print()
    print("AI ANSWER")
    print("=========")
    print(answer)


if __name__ == "__main__":
    main()