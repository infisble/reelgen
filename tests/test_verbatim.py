from reelgen.verbatim import compare, detect_lang, extract_lines, normalize_words


def test_extracts_all_quote_styles_in_order():
    idea = 'A says "Hello there." B: «Привіт!» C: “Second” D: „Третій“'
    assert extract_lines(idea) == ["Hello there.", "Привіт!", "Second", "Третій"]


def test_no_quotes_returns_empty():
    assert extract_lines("A cat walks into a bar.") == []


def test_apostrophes_are_not_quotes():
    assert extract_lines("Кав’ярня, м'ята і don't.") == []


def test_compare_ignores_case_punctuation_and_apostrophe_style():
    assert compare("Я повернулася, бо кав’ярня — тут!", "я повернулася бо кавярня тут").ok


def test_compare_catches_paraphrase():
    r = compare("I came back because nobody asks here", "I returned because nobody asks here")
    assert not r.ok and 0 < r.wer < 0.3


def test_compare_catches_missing_word():
    assert not compare("one two three", "one three").ok


def test_detect_lang():
    assert detect_lang("Привіт, як справи?") == "uk"
    assert detect_lang("Hello there") == "en"


def test_normalize_words_hyphen_splits():
    assert normalize_words("Hi-fi, OK?") == ["hi", "fi", "ok"]
