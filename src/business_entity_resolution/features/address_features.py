
import re

from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Feature 1: Address Token Jaccard Similarity
def address_token_jaccard(
    address_a: str | None,
    address_b: str | None
) -> float:

    tokens_a = set(
        re.findall(r"\w+", (address_a or "").casefold())
    )

    tokens_b = set(
        re.findall(r"\w+", (address_b or "").casefold())
    )

    if not tokens_a or not tokens_b:
        return 0.0

    common_tokens = tokens_a.intersection(tokens_b)
    all_tokens = tokens_a.union(tokens_b)

    return len(common_tokens) / len(all_tokens)


# Feature 2: Address Levenshtein Similarity
def address_levenshtein_similarity(
    address_a: str | None,
    address_b: str | None
) -> float:

    address_a = (address_a or "").casefold().strip()
    address_b = (address_b or "").casefold().strip()

    if not address_a or not address_b:
        return 0.0

    return Levenshtein.normalized_similarity(
        address_a,
        address_b
    )


# Feature 3: Address Character TF-IDF Similarity
def address_char_tfidf_similarity(
    address_a: str | None,
    address_b: str | None,
    vectorizer: TfidfVectorizer
) -> float:

    address_a = (address_a or "").casefold().strip()
    address_b = (address_b or "").casefold().strip()

    if not address_a or not address_b:
        return 0.0

    vectors = vectorizer.transform([
        address_a,
        address_b
    ])

    score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0, 0]

    return float(score)


# Feature 4: Address Numeric Overlap
def address_numeric_overlap(
    address_a: str | None,
    address_b: str | None
) -> float:

    numbers_a = set(
        re.findall(
            r"\b\d+[a-z]?\b",
            (address_a or "").casefold()
        )
    )

    numbers_b = set(
        re.findall(
            r"\b\d+[a-z]?\b",
            (address_b or "").casefold()
        )
    )

    if not numbers_a or not numbers_b:
        return 0.0

    common_numbers = numbers_a.intersection(numbers_b)
    all_numbers = numbers_a.union(numbers_b)

    return len(common_numbers) / len(all_numbers)


# Feature 5: Numeric Information Availability
def address_numeric_available(
    address_a: str | None,
    address_b: str | None
) -> int:

    has_numbers_a = bool(
        re.search(r"\d", address_a or "")
    )

    has_numbers_b = bool(
        re.search(r"\d", address_b or "")
    )

    return int(has_numbers_a and has_numbers_b)


# Test all five address features
if __name__ == "__main__":

    address1 = "12 Gandhi Road, Coimbatore"
    address2 = "12 Gandhi Rd, Coimbatore"

    training_addresses = [
        "12 Gandhi Road, Coimbatore",
        "12 Gandhi Rd, Coimbatore",
        "45 Nehru Street, Chennai",
        "88 MG Road, Bangalore",
        "10 Market Street, Coimbatore"
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )

    vectorizer.fit(training_addresses)

    jaccard_score = address_token_jaccard(
        address1, address2
    )

    levenshtein_score = address_levenshtein_similarity(
        address1, address2
    )

    tfidf_score = address_char_tfidf_similarity(
        address1, address2, vectorizer
    )

    numeric_score = address_numeric_overlap(
        address1, address2
    )

    numeric_available = address_numeric_available(
        address1, address2
    )

    print(
        f"Address Jaccard Similarity: {jaccard_score:.2f}"
    )

    print(
        f"Address Levenshtein Similarity: {levenshtein_score:.4f}"
    )

    print(
        f"Address TF-IDF Similarity: {tfidf_score:.4f}"
    )

    print(
        f"Address Numeric Overlap: {numeric_score:.2f}"
    )

    print(
        f"Numeric Information Available: {numeric_available}"
    )
