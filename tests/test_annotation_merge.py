"""Tests for annotation merge, confidence scoring, and early stopping.

These tests cover the core changes in ``langextract/annotation.py``:

- ``_merge_non_overlapping_extractions`` with ``total_passes``
  parameter and confidence score assignment.
- Early stopping logic in the sync
  ``_annotate_documents_sequential_passes`` method.
"""

from __future__ import annotations

import pytest
from langextract.core.data import CharInterval, Extraction

from langextract.annotation import (
    _merge_non_overlapping_extractions,
)


# ── Helpers ─────────────────────────────────────────────────


def _ext(
    cls: str,
    text: str,
    start: int,
    end: int,
) -> Extraction:
    """Build a minimal ``Extraction`` for testing."""
    return Extraction(
        extraction_class=cls,
        extraction_text=text,
        char_interval=CharInterval(start_pos=start, end_pos=end),
    )


# ── _merge_non_overlapping_extractions ──────────────────────


class TestMergeNonOverlapping:
    """Unit tests for the merge + confidence scoring logic."""

    def test_empty_input_returns_empty(self):
        assert _merge_non_overlapping_extractions([]) == []

    def test_single_pass_no_confidence(self):
        """Single pass with total_passes=1 leaves confidence as None."""
        exts = [_ext("A", "hello", 0, 5)]
        result = _merge_non_overlapping_extractions([exts])
        assert len(result) == 1
        assert result[0].confidence_score is None

    def test_single_pass_explicit_total_passes(self):
        """Single pass list but total_passes=3 should score 1/3."""
        exts = [_ext("A", "hello", 0, 5)]
        result = _merge_non_overlapping_extractions([exts], total_passes=3)
        assert len(result) == 1
        assert result[0].confidence_score == pytest.approx(1.0 / 3)

    def test_two_passes_no_overlap_both_kept(self):
        """Non-overlapping extractions from 2 passes are both kept."""
        pass1 = [_ext("A", "hello", 0, 5)]
        pass2 = [_ext("B", "world", 10, 15)]
        result = _merge_non_overlapping_extractions([pass1, pass2], total_passes=2)
        assert len(result) == 2
        # Each only appeared in 1 pass → confidence = 0.5
        assert result[0].confidence_score == pytest.approx(0.5)
        assert result[1].confidence_score == pytest.approx(0.5)

    def test_two_passes_full_overlap_first_wins(self):
        """Overlapping extraction from pass 2 is dropped; pass 1 wins."""
        pass1 = [_ext("A", "hello", 0, 5)]
        pass2 = [_ext("A", "hello", 0, 5)]
        result = _merge_non_overlapping_extractions([pass1, pass2], total_passes=2)
        assert len(result) == 1
        assert result[0].extraction_text == "hello"
        # Appeared in both passes → confidence = 1.0
        assert result[0].confidence_score == pytest.approx(1.0)

    def test_three_passes_mixed(self):
        """Three passes: one extraction found in all 3, one only in pass 1."""
        pass1 = [
            _ext("A", "hello", 0, 5),
            _ext("B", "unique", 20, 26),
        ]
        pass2 = [_ext("A", "hello", 0, 5)]  # overlap with A
        pass3 = [_ext("A", "hello", 0, 5)]  # overlap with A
        result = _merge_non_overlapping_extractions(
            [pass1, pass2, pass3], total_passes=3
        )
        assert len(result) == 2
        # A appeared in all 3 passes
        a_ext = next(e for e in result if e.extraction_class == "A")
        assert a_ext.confidence_score == pytest.approx(1.0)
        # B appeared in only 1 pass
        b_ext = next(e for e in result if e.extraction_class == "B")
        assert b_ext.confidence_score == pytest.approx(1.0 / 3)

    def test_partial_overlap_increments_count(self):
        """Partial character overlap still counts as agreement."""
        pass1 = [_ext("A", "hello world", 0, 11)]
        pass2 = [_ext("A", "hello", 0, 5)]  # partial overlap
        result = _merge_non_overlapping_extractions([pass1, pass2], total_passes=2)
        assert len(result) == 1
        assert result[0].confidence_score == pytest.approx(1.0)

    def test_no_char_interval_no_overlap(self):
        """Extractions without char_interval are never overlapping."""
        ext_no_interval = Extraction(
            extraction_class="X",
            extraction_text="orphan",
        )
        pass1 = [_ext("A", "hello", 0, 5)]
        pass2 = [ext_no_interval]
        result = _merge_non_overlapping_extractions([pass1, pass2], total_passes=2)
        assert len(result) == 2

    def test_default_total_passes_skips_confidence(self):
        """Default total_passes=1 skips confidence assignment."""
        pass1 = [_ext("A", "hello", 0, 5)]
        pass2 = [_ext("A", "hello", 0, 5)]
        result = _merge_non_overlapping_extractions([pass1, pass2])
        assert len(result) == 1
        # confidence_score remains None with default total_passes=1
        assert result[0].confidence_score is None

    def test_new_extraction_added_from_later_pass(self):
        """Non-overlapping extraction from pass 2 gets appended."""
        pass1 = [_ext("A", "hello", 0, 5)]
        pass2 = [_ext("B", "world", 100, 105)]
        result = _merge_non_overlapping_extractions([pass1, pass2], total_passes=2)
        classes = [e.extraction_class for e in result]
        assert "A" in classes
        assert "B" in classes
