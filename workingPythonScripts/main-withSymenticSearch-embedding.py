from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from docx import Document
from openai import OpenAI


DATA_FOLDER = Path("../data")

client = OpenAI()


# --------------------------------
# FILE READERS
# --------------------------------

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


# --------------------------------
# CONVERT FILE TO TEXT
# --------------------------------

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


# --------------------------------
# LOAD DOCUMENTS
# --------------------------------

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


# --------------------------------
# SEARCH DOCUMENTS
# --------------------------------

def search_documents(documents, question):

    results = []

    words = question.lower().split()

    for document in documents:

        content = document["content"].lower()

        score = 0

        for word in words:

            if word in content:
                score += 1

        if score > 0:

            results.append({
                "filename": document["filename"],
                "content": document["content"],
                "score": score
            })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results


# --------------------------------
# ASK AI
# --------------------------------

def ask_ai(question, search_results):

    context = ""

    for result in search_results:

        context += (
            f"\nSOURCE: {result['filename']}\n"
            f"{result['content']}\n"
        )

    prompt = f"""
You are a helpful assistant that answers questions about the user's files.

Use ONLY the information provided in the CONTEXT.

If the answer is not present in the context, say:
"I couldn't find that information in the files."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    return response.output_text


# --------------------------------
# MAIN CHAT
# --------------------------------

def main():

    print("===================================")
    print("       AI FILE ASSISTANT")
    print("===================================")

    print()

    print("Loading files...")

    documents = load_documents()

    print(f"Loaded {len(documents)} files.")

    print()

    while True:

        question = input("Ask a question (or type 'exit'): ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question.strip():
            continue

        print()
        print("Searching files...")

        results = search_documents(
            documents,
            question
        )

        if not results:

            print()
            print("I couldn't find relevant information in your files.")
            print()

            continue

        print(
            f"Found information in {len(results)} file(s)."
        )

        print()
        print("Asking AI...")

        answer = ask_ai(
            question,
            results
        )

        print()
        print("AI:")
        print(answer)

        print()
        print("-----------------------------------")
        print()


if __name__ == "__main__":
    main()