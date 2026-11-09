# embedding_experiment.py
# STEP 4b: a tiny experiment to SEE what embeddings do, before we embed all 4,257 chunks.
#
# We embed a few short Danish sentences and a question, and print how similar each
# sentence is to the question. If the model understands meaning, the sentence that
# ANSWERS the question should score highest, even though it shares few words with it.
#
# Run from the backend/ folder:
#   .venv/bin/python -m rag.embedding_experiment

from rag.embeddings import (
    DIMENSIONS,
    EMBEDDING_MODEL,
    cosine_similarity,
    embed_documents,
    embed_query,
)

# Pretend these are chunks from the conditions. Two are about waiting periods
# (karenstid), one is about something else in the same document, one is nonsense
# for insurance. Try changing or adding sentences yourself!
DOCUMENTS = [
    "Forsikringen dækker ikke driftstab i karenstiden. Karenstiden er 12 timer.",
    "Der ydes ingen erstatning for de første 12 timer efter skaden er konstateret.",
    "Erstatningen opgøres på grundlag af nyværdiprincippet.",
    "Hunden skal vaccineres hvert år, og kattemaden står i skabet.",
]

# A question that uses DIFFERENT words from document 2 ("ventetid" instead of "12 timer").
QUESTION = "Hvor lang er ventetiden før cyberforsikringen begynder at dække mit tab?"


def main() -> None:
    print(f"Model: {EMBEDDING_MODEL}  (asking for {DIMENSIONS} numbers per vector)\n")

    # One request for the documents, one for the question.
    doc_vectors = embed_documents(DOCUMENTS)
    question_vector = embed_query(QUESTION)

    # What did we actually get back?
    print(f"Got {len(doc_vectors)} document vectors, each with {len(doc_vectors[0])} numbers.")
    print(f"The first 5 numbers of vector 1: {[round(x, 4) for x in doc_vectors[0][:5]]}")
    print("(Meaningless to a human, but that's the 'coordinates' of the text.)\n")

    print(f"QUESTION: {QUESTION}\n")

    # Score every document against the question.
    scored = [
        (cosine_similarity(question_vector, vector), text)
        for vector, text in zip(doc_vectors, DOCUMENTS)
    ]

    # Sort from most to least similar. reverse=True = highest first.
    scored.sort(reverse=True)
    for score, text in scored:
        print(f"  {score:.3f}   {text}")


if __name__ == "__main__":
    main()
