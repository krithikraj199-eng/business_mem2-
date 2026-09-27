from rapidfuzz.distance import Levenshtein

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Feature 1: Name Token Jaccard Similarity
def name_token_jaccard(
    name_a: str | None,
    name_b: str | None
) -> float:

    words_a = set((name_a or "").casefold().split())
    words_b = set((name_b or "").casefold().split())

    if not words_a or not words_b:
        return 0.0

    common_words = words_a.intersection(words_b)
    all_words = words_a.union(words_b)

    return len(common_words) / len(all_words)


# Feature 2: Name Levenshtein Similarity
def name_levenshtein_similarity(
    name_a: str | None,
    name_b: str | None
) -> float:

    name_a = (name_a or "").casefold().strip()
    name_b = (name_b or "").casefold().strip()

    if not name_a or not name_b:
        return 0.0

    return Levenshtein.normalized_similarity(name_a, name_b)


# Feature 3: Character N-gram TF-IDF Similarity
def name_char_tfidf_similarity(
    name_a: str | None,
    name_b: str | None,
    vectorizer: TfidfVectorizer
) -> float:

    name_a = (name_a or "").casefold().strip()
    name_b = (name_b or "").casefold().strip()

    if not name_a or not name_b:
        return 0.0

    # Convert both names into TF-IDF vectors
    vectors = vectorizer.transform([name_a, name_b])

    # Calculate cosine similarity
    score = cosine_similarity(
        vectors[0],
        vectors[1]
    )[0, 0]

    return float(score)


# Test all three features
if __name__ == "__main__":

    name1 = "Sri Krishna Textiles"
    name2 = "Sri Krishna Textile"

    # Example training names for our local test
    training_names = [
        "Sri Krishna Textiles",
        "Sree Krishna Textile",
        "Krishna Medical Store",
        "Sri Krishna Textile",
        "Kumar Textile"
    ]

    # Learn TF-IDF vocabulary from training names
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )

    vectorizer.fit(training_names)

    # Calculate similarity features
    jaccard_score = name_token_jaccard(name1, name2)

    levenshtein_score = name_levenshtein_similarity(
        name1, name2
    )

    tfidf_score = name_char_tfidf_similarity(
        name1,
        name2,
        vectorizer
    )

    print(f"Jaccard Similarity: {jaccard_score:.2f}")

    print(f"Levenshtein Similarity: {levenshtein_score:.2f}")

    print(f"Character TF-IDF Similarity: {tfidf_score:.4f}")