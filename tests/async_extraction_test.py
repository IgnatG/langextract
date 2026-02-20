# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for async extraction path (async_infer, async_annotate, async_extract).

Verifies that:
- BaseLanguageModel.async_infer delegates to sync infer by default.
- Annotator.async_annotate_text produces the same results as annotate_text.
- Annotator.async_annotate_documents produces the same results as
  annotate_documents.
- async_extract mirrors extract output.
- Multi-pass early stopping works in the async path.
"""

from __future__ import annotations

import asyncio
import textwrap
from collections.abc import Iterator, Sequence
from unittest import mock

import pytest

from langextract import annotation, extraction, prompting, resolver as resolver_lib
from langextract.core import base_model, data, types


# ── Helpers ─────────────────────────────────────────────────────────────────


class FakeLanguageModel(base_model.BaseLanguageModel):
    """Minimal in-memory language model for testing."""

    def __init__(self, responses: list[str] | None = None):
        super().__init__()
        self.responses = responses or []
        self._call_index = 0
        self.infer_call_count = 0
        self.async_infer_call_count = 0

    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[types.ScoredOutput]]:
        self.infer_call_count += 1
        for _ in batch_prompts:
            resp = self.responses[self._call_index % len(self.responses)]
            self._call_index += 1
            yield [types.ScoredOutput(score=1.0, output=resp)]


class FakeAsyncLanguageModel(FakeLanguageModel):
    """Fake model that overrides async_infer with native async."""

    async def async_infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> list[Sequence[types.ScoredOutput]]:
        self.async_infer_call_count += 1
        results = []
        for _ in batch_prompts:
            resp = self.responses[self._call_index % len(self.responses)]
            self._call_index += 1
            results.append([types.ScoredOutput(score=1.0, output=resp)])
        return results


SAMPLE_TEXT = (
    "Patient Jane Doe, ID 67890, received 10mg of Lisinopril daily for"
    " hypertension diagnosed on 2023-03-15."
)

SAMPLE_YAML_RESPONSE = textwrap.dedent(
    f"""\
    ```yaml
    {data.EXTRACTIONS_KEY}:
    - patient: "Jane Doe"
      patient_index: 1
      medication: "Lisinopril"
      medication_index: 8
    ```"""
)


def _make_annotator(model):
    return annotation.Annotator(
        language_model=model,
        prompt_template=prompting.PromptTemplateStructured(description=""),
    )


# ── Tests ───────────────────────────────────────────────────────────────────


class TestBaseModelAsyncInferFallback:
    """Verify the default async_infer delegates to sync infer."""

    @pytest.mark.asyncio
    async def test_default_async_infer_delegates_to_sync(self):
        model = FakeLanguageModel(responses=[SAMPLE_YAML_RESPONSE])
        result = await model.async_infer(["prompt1"])
        assert len(result) == 1
        assert result[0][0].output == SAMPLE_YAML_RESPONSE
        assert model.infer_call_count == 1

    @pytest.mark.asyncio
    async def test_native_async_infer_does_not_call_sync(self):
        model = FakeAsyncLanguageModel(responses=[SAMPLE_YAML_RESPONSE])
        result = await model.async_infer(["prompt1"])
        assert len(result) == 1
        assert result[0][0].output == SAMPLE_YAML_RESPONSE
        assert model.async_infer_call_count == 1
        assert model.infer_call_count == 0  # sync path NOT used


class TestAsyncAnnotateText:
    """Verify async_annotate_text produces equivalent results."""

    @pytest.mark.asyncio
    async def test_single_chunk_extraction(self):
        model = FakeLanguageModel(responses=[SAMPLE_YAML_RESPONSE])
        annotator = _make_annotator(model)
        resolver = resolver_lib.Resolver(
            format_type=data.FormatType.YAML,
            extraction_index_suffix=resolver_lib.DEFAULT_INDEX_SUFFIX,
        )

        result = await annotator.async_annotate_text(
            text=SAMPLE_TEXT,
            resolver=resolver,
        )

        assert result.text == SAMPLE_TEXT
        assert result.extractions is not None
        classes = {e.extraction_class for e in result.extractions}
        assert "patient" in classes
        assert "medication" in classes

    @pytest.mark.asyncio
    async def test_async_matches_sync(self):
        """Async and sync paths should produce identical extractions."""
        responses = [SAMPLE_YAML_RESPONSE] * 2

        sync_model = FakeLanguageModel(responses=responses)
        sync_annotator = _make_annotator(sync_model)
        resolver = resolver_lib.Resolver(
            format_type=data.FormatType.YAML,
            extraction_index_suffix=resolver_lib.DEFAULT_INDEX_SUFFIX,
        )
        sync_result = sync_annotator.annotate_text(text=SAMPLE_TEXT, resolver=resolver)

        async_model = FakeLanguageModel(responses=responses)
        async_annotator = _make_annotator(async_model)
        async_result = await async_annotator.async_annotate_text(
            text=SAMPLE_TEXT, resolver=resolver
        )

        assert len(sync_result.extractions) == len(async_result.extractions)
        for s, a in zip(sync_result.extractions, async_result.extractions):
            assert s.extraction_class == a.extraction_class
            assert s.extraction_text == a.extraction_text


class TestAsyncAnnotateDocuments:
    """Verify async_annotate_documents for multi-document input."""

    @pytest.mark.asyncio
    async def test_multiple_documents(self):
        model = FakeLanguageModel(responses=[SAMPLE_YAML_RESPONSE] * 4)
        annotator = _make_annotator(model)
        resolver = resolver_lib.Resolver(
            format_type=data.FormatType.YAML,
            extraction_index_suffix=resolver_lib.DEFAULT_INDEX_SUFFIX,
        )

        docs = [
            data.Document(text=SAMPLE_TEXT, document_id="doc1"),
            data.Document(text=SAMPLE_TEXT, document_id="doc2"),
        ]

        results = await annotator.async_annotate_documents(
            documents=docs, resolver=resolver
        )

        assert len(results) == 2
        assert results[0].document_id == "doc1"
        assert results[1].document_id == "doc2"
        for r in results:
            assert r.extractions is not None
            assert len(r.extractions) > 0


class TestAsyncMultiPassEarlyStopping:
    """Verify async multi-pass stops early when convergent."""

    @pytest.mark.asyncio
    async def test_early_stop_when_no_new_extractions(self):
        """Pass 2 yields identical extractions → early stop before pass 3."""
        model = FakeLanguageModel(responses=[SAMPLE_YAML_RESPONSE] * 10)
        annotator = _make_annotator(model)
        resolver = resolver_lib.Resolver(
            format_type=data.FormatType.YAML,
            extraction_index_suffix=resolver_lib.DEFAULT_INDEX_SUFFIX,
        )

        results = await annotator.async_annotate_documents(
            documents=[data.Document(text=SAMPLE_TEXT, document_id="doc1")],
            resolver=resolver,
            extraction_passes=3,
        )

        assert len(results) == 1
        # The model should have been called only twice (pass 1 + pass 2),
        # not three times, because pass 2 adds nothing new.
        # With single-chunk text, each pass = 1 infer call.
        assert model.infer_call_count <= 2


class TestAsyncExtractFunction:
    """Verify the top-level async_extract function."""

    @pytest.mark.asyncio
    async def test_async_extract_string_input(self):
        model = FakeLanguageModel(responses=[SAMPLE_YAML_RESPONSE])
        examples = [
            data.ExampleData(
                text=SAMPLE_TEXT,
                extractions=[
                    data.Extraction(
                        extraction_class="patient",
                        extraction_text="Jane Doe",
                    ),
                ],
            )
        ]

        result = await extraction.async_extract(
            text_or_documents=SAMPLE_TEXT,
            prompt_description="Extract patient info",
            examples=examples,
            model=model,
            show_progress=False,
            prompt_validation_level="OFF",
        )

        assert isinstance(result, data.AnnotatedDocument)
        assert result.extractions is not None
