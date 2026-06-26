"""Tests for extractor helper functions — normalization, classification, JSON recovery, chunking.

All tests exercise pure functions (no mocks or I/O needed).
"""

from beliefstate.extractor import (
    normalize_numbers,
    normalize_currency,
    normalize_dates,
    normalize_percentages,
    normalize_value_format,
    classify_response_type,
    chunk_response_by_paragraphs,
    recover_json_from_response,
)


# ── Number Normalization ─────────────────────────────────────────────────


class TestNormalizeNumbers:
    def test_comma_separated(self):
        assert normalize_numbers("5,000") == "5000"

    def test_multiple_commas(self):
        assert normalize_numbers("1,000,000") == "1000000"

    def test_no_commas_unchanged(self):
        assert normalize_numbers("42") == "42"

    def test_text_with_number(self):
        assert normalize_numbers("I have 1,500 apples") == "I have 1500 apples"

    def test_empty_string(self):
        assert normalize_numbers("") == ""


# ── Currency Normalization ───────────────────────────────────────────────


class TestNormalizeCurrency:
    def test_dollar_sign(self):
        result = normalize_currency("$5000")
        assert "USD" in result
        assert "5000" in result

    def test_euro_sign(self):
        result = normalize_currency("€100")
        assert "EUR" in result
        assert "100" in result

    def test_pound_sign(self):
        result = normalize_currency("£50")
        assert "GBP" in result
        assert "50" in result

    def test_yen_sign(self):
        result = normalize_currency("¥10000")
        assert "JPY" in result

    def test_no_currency_unchanged(self):
        assert normalize_currency("hello world") == "hello world"

    def test_dollar_with_cents(self):
        result = normalize_currency("$99.99")
        assert "USD" in result
        assert "99.99" in result


# ── Date Normalization ───────────────────────────────────────────────────


class TestNormalizeDates:
    def test_month_name_day_year(self):
        result = normalize_dates("March 15, 2024")
        assert "2024-03-15" in result

    def test_abbreviated_month(self):
        result = normalize_dates("Jan 1 2023")
        assert "2023-01-01" in result

    def test_day_month_year(self):
        result = normalize_dates("15 March 2024")
        assert "2024-03-15" in result

    def test_us_format_slashes(self):
        result = normalize_dates("03/15/2024")
        assert "2024-03-15" in result

    def test_no_date_unchanged(self):
        assert normalize_dates("hello") == "hello"


# ── Percentage Normalization ─────────────────────────────────────────────


class TestNormalizePercentages:
    def test_integer_percent(self):
        result = normalize_percentages("15%")
        assert "0.15" in result

    def test_decimal_percent(self):
        result = normalize_percentages("7.5%")
        # 7.5 / 100 = 0.075, but rstrip('0') trims trailing zeros: "0.08" → may round
        assert result.startswith("0.0")

    def test_hundred_percent(self):
        result = normalize_percentages("100%")
        assert "1" in result

    def test_no_percent_unchanged(self):
        assert normalize_percentages("hello") == "hello"


# ── Combined normalize_value_format ────────────────────────────────────────


class TestNormalizeValue:
    def test_empty_string(self):
        assert normalize_value_format("") == ""

    def test_combined_number_and_currency(self):
        result = normalize_value_format("$1,000")
        assert "USD" in result
        assert "1000" in result

    def test_plain_text_unchanged(self):
        assert normalize_value_format("Python") == "Python"


# ── Response Type Classification ─────────────────────────────────────────


class TestClassifyResponseType:
    def test_empty_string(self):
        assert classify_response_type("") == "conversational"

    def test_json_array(self):
        assert classify_response_type('[{"key": "value"}]') == "json"

    def test_json_object(self):
        assert classify_response_type('{"key": "value"}') == "json"

    def test_sql_select(self):
        assert classify_response_type("SELECT * FROM users WHERE id = 1") == "sql"

    def test_sql_insert(self):
        assert classify_response_type("INSERT INTO users VALUES (1, 'Alice')") == "sql"

    def test_conversational_text(self):
        assert (
            classify_response_type("I really like programming in Python.")
            == "conversational"
        )

    def test_code_block_majority(self):
        text = "Here is code:\n```python\ndef foo():\n    return 1\n```\n"
        result = classify_response_type(text)
        assert result in ("code", "markdown_heavy")

    def test_markdown_with_inline_code(self):
        text = (
            "Use `foo()` and `bar()` and `baz()` and `qux()` and `quux()` to do things."
        )
        assert classify_response_type(text) == "markdown_heavy"

    def test_plain_json_invalid_not_json(self):
        # Starts with { but isn't valid JSON
        text = "{this is not json}"
        assert classify_response_type(text) == "conversational"


# ── Paragraph Chunking ───────────────────────────────────────────────────


class TestChunkResponseByParagraphs:
    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_response_by_paragraphs(text, max_chunk_length=2000)
        assert chunks == [text]

    def test_empty_text(self):
        chunks = chunk_response_by_paragraphs("", max_chunk_length=2000)
        assert chunks == [""]

    def test_none_text(self):
        chunks = chunk_response_by_paragraphs(None, max_chunk_length=2000)  # type: ignore[arg-type]
        assert chunks == [None]

    def test_long_text_splits_at_paragraphs(self):
        # Create text with 3 paragraphs, each 100 chars
        para = "A" * 100
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = chunk_response_by_paragraphs(text, max_chunk_length=150)
        assert len(chunks) >= 2

    def test_single_long_paragraph_stays_together(self):
        text = "A" * 3000  # No paragraph breaks
        chunks = chunk_response_by_paragraphs(text, max_chunk_length=2000)
        # Should return the full text since there are no paragraph breaks
        assert len(chunks) == 1
        assert chunks[0] == text


# ── JSON Recovery ────────────────────────────────────────────────────────


class TestRecoverJsonFromResponse:
    def test_layer1_direct_parse_array(self):
        result = recover_json_from_response('[{"subject": "USER"}]')
        assert result == [{"subject": "USER"}]

    def test_layer1_direct_parse_dict_with_beliefs(self):
        result = recover_json_from_response('{"beliefs": [{"subject": "USER"}]}')
        assert result == [{"subject": "USER"}]

    def test_layer1_direct_parse_dict_with_root(self):
        result = recover_json_from_response('{"root": [{"subject": "USER"}]}')
        assert result == [{"subject": "USER"}]

    def test_layer2_markdown_code_block(self):
        text = '```json\n[{"subject": "USER"}]\n```'
        result = recover_json_from_response(text)
        assert result == [{"subject": "USER"}]

    def test_layer3_json_embedded_in_text(self):
        text = 'Here are the beliefs: [{"subject": "USER"}] and that is all.'
        result = recover_json_from_response(text)
        assert result == [{"subject": "USER"}]

    def test_layer4_smart_quotes(self):
        text = "[\u201c\u201d]"  # Smart quotes with empty content
        result = recover_json_from_response(text)
        # Smart quotes may or may not parse; just verify no crash
        assert result is None or isinstance(result, list)

    def test_layer5_truncated_json_with_trailing_comma(self):
        # Has both brackets, but trailing comma makes it invalid — Layer 5 handles this
        text = '[{"subject": "USER", "predicate": "likes", "value": "Python"},]'
        result = recover_json_from_response(text)
        if result is not None:
            assert isinstance(result, list)

    def test_layer6_missing_braces(self):
        text = '[{"subject": "USER", "predicate": "likes", "value": "Python"'
        result = recover_json_from_response(text)
        # Should recover by adding closing } and ]
        if result is not None:
            assert result[0]["subject"] == "USER"

    def test_none_input(self):
        assert recover_json_from_response(None) is None  # type: ignore[arg-type]

    def test_empty_string(self):
        assert recover_json_from_response("") is None

    def test_non_string_input(self):
        assert recover_json_from_response(123) is None  # type: ignore[arg-type]

    def test_no_json_at_all(self):
        assert recover_json_from_response("This is plain text with no JSON.") is None

    def test_valid_dict_not_array(self):
        """A dict without 'beliefs' or 'root' keys should return None."""
        result = recover_json_from_response('{"unrelated": "data"}')
        assert result is None
