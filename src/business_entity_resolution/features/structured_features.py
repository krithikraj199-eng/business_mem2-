
import re


# Normalize a postal code without assuming a country format
def normalize_postal_code(postal_code: str | None) -> str:

    postal_code = (postal_code or "").casefold()

    # Remove spaces and hyphens
    return re.sub(r"[\s-]+", "", postal_code)


# Feature 1: Postal Code Match
def postal_code_match(
    postal_a: str | None,
    postal_b: str | None
) -> int:

    postal_a = normalize_postal_code(postal_a)
    postal_b = normalize_postal_code(postal_b)

    # Missing postal codes are not treated as matches
    if not postal_a or not postal_b:
        return 0

    return int(postal_a == postal_b)


# Feature 2: Postal Code Availability
def postal_code_available(
    postal_a: str | None,
    postal_b: str | None
) -> int:

    postal_a = normalize_postal_code(postal_a)
    postal_b = normalize_postal_code(postal_b)

    return int(bool(postal_a) and bool(postal_b))


# Normalize country names
def normalize_country(country: str | None) -> str:

    if not country:
        return ""

    return " ".join(country.casefold().split())


# Feature 3: Country Agreement
def country_match(
    country_a: str | None,
    country_b: str | None
) -> int:

    country_a = normalize_country(country_a)
    country_b = normalize_country(country_b)

    if not country_a or not country_b:
        return 0

    return int(country_a == country_b)


# Feature 4: Country Availability
def country_available(
    country_a: str | None,
    country_b: str | None
) -> int:

    country_a = normalize_country(country_a)
    country_b = normalize_country(country_b)

    return int(bool(country_a) and bool(country_b))


# Test our postal features
if __name__ == "__main__":

    print(
        "Same Postal Code:",
        postal_code_match("641001", "641 001")
    )

    print(
        "Different Postal Code:",
        postal_code_match("641001", "641002")
    )

    print(
        "Missing Postal Code:",
        postal_code_match("641001", None)
    )

    print(
        "Both Postal Codes Available:",
        postal_code_available("641001", "641 001")
    )

    print(
        "One Postal Code Missing:",
        postal_code_available("641001", None)
    )


    print(
        "Same Country:",
        country_match("India", "india")
    )

    print(
        "Different Country:",
        country_match("India", "France")
    )

    print(
        "Unseen Country:",
        country_match("France", "FRANCE")
    )

    print(
        "Missing Country:",
        country_match("India", None)
    )

    print(
        "Both Countries Available:",
        country_available("India", "France")
    )

    print(
        "One Country Missing:",
        country_available("India", None)
    )
