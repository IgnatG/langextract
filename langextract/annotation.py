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

"""Provides functionality for annotating medical text using a language model.

The annotation process involves tokenizing the input text, generating prompts
for the language model, and resolving the language model's output into
structured annotations.

Usage example:
    annotator = Annotator(language_model, prompt_template)
    annotated_documents = annotator.annotate_documents(documents, resolver)
"""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Iterable, Iterator
import time
from typing import DefaultDict

from absl import logging
from intervaltree import IntervalTree

from langextract import chunking
from langextract import progress
from langextract import prompting
from langextract import resolver as resolver_lib
from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import format_handler as fh
from langextract.core import tokenizer as tokenizer_lib

# Threshold for switching from flat O(n²) overlap check to an interval tree.
# For small extraction counts the flat loop is faster due to lower constant
# factors; the tree pays off when the merged set grows large.
_INTERVAL_TREE_THRESHOLD = 200


def _merge_non_overlapping_extractions(
    all_extractions: list[Iterable[data.Extraction]],
    total_passes: int = 1,
) -> list[data.Extraction]:
    """Merges extractions from multiple extraction passes.

    When extractions from different passes overlap in their character positions,
    the extraction from the earlier pass is kept (first-pass wins strategy).
    Only non-overlapping extractions from later passes are added to the result.

    When ``total_passes > 1``, a ``confidence_score`` is computed for
    each merged extraction as ``appearances / total_passes``, where
    ``appearances`` is the number of passes that produced an overlapping
    extraction in the same character region.

    For large extraction counts (≥ ``_INTERVAL_TREE_THRESHOLD``), an
    ``IntervalTree`` is used for O(log n) overlap queries instead of the naïve
    O(n²) pairwise scan.

    Args:
      all_extractions: List of extraction iterables from different sequential
        extraction passes, ordered by pass number.
      total_passes: The number of extraction passes that were actually
        executed (accounting for early stopping).  Defaults to ``1``
        which skips confidence scoring.

    Returns:
      List of merged extractions with overlaps resolved in favor of earlier
      passes. When ``total_passes > 1``, each extraction's
      ``confidence_score`` is set to ``appearances / total_passes``.
    """
    if not all_extractions:
        return []

    if len(all_extractions) == 1:
        single = list(all_extractions[0])
        if total_passes > 1:
            for ext in single:
                ext.confidence_score = 1.0 / total_passes
        return single

    merged_extractions = list(all_extractions[0])
    # Track how many passes produced each merged extraction.
    appearance_counts: list[int] = [1] * len(merged_extractions)

    # Seed the interval tree with first-pass extractions.
    tree: IntervalTree | None = None

    def _build_tree() -> IntervalTree:
        """Build the interval tree from ``merged_extractions``.

        Each interval stores the index into ``merged_extractions`` as its
        ``data`` field for efficient appearance-count updates.
        """
        t = IntervalTree()
        for idx, ext in enumerate(merged_extractions):
            if ext.char_interval is not None:
                s, e = ext.char_interval.start_pos, ext.char_interval.end_pos
                if s is not None and e is not None and s < e:
                    t.addi(s, e, idx)
        return t

    for pass_extractions in all_extractions[1:]:
        use_tree = len(merged_extractions) >= _INTERVAL_TREE_THRESHOLD

        if use_tree and tree is None:
            tree = _build_tree()

        for extraction in pass_extractions:
            overlaps = False
            overlapping_idx: int | None = None
            if extraction.char_interval is not None:
                s = extraction.char_interval.start_pos
                e = extraction.char_interval.end_pos
                if s is not None and e is not None:
                    if use_tree and tree is not None:
                        # O(log n + k) overlap query via interval tree.
                        hits = tree.overlap(s, e)
                        if hits:
                            overlaps = True
                            # Attribute the match to the earliest merged
                            # extraction (first-pass-wins).
                            overlapping_idx = min(iv.data for iv in hits)
                    else:
                        # Flat O(n) scan — faster for small merged sets.
                        for idx, existing_extraction in enumerate(merged_extractions):
                            if existing_extraction.char_interval is not None:
                                if _extractions_overlap(
                                    extraction, existing_extraction
                                ):
                                    overlaps = True
                                    overlapping_idx = idx
                                    break

            if overlaps:
                # Increment the appearance count for the matched extraction.
                if overlapping_idx is not None:
                    appearance_counts[overlapping_idx] += 1
            else:
                new_idx = len(merged_extractions)
                merged_extractions.append(extraction)
                appearance_counts.append(1)
                # Keep the tree in sync when we're using it.
                if (
                    use_tree
                    and tree is not None
                    and extraction.char_interval is not None
                ):
                    s = extraction.char_interval.start_pos
                    e = extraction.char_interval.end_pos
                    if s is not None and e is not None and s < e:
                        tree.addi(s, e, new_idx)
                # If we just crossed the threshold, build the tree for
                # the next iteration.
                if not use_tree and len(merged_extractions) >= _INTERVAL_TREE_THRESHOLD:
                    tree = _build_tree()

    # ── Assign confidence scores ──
    if total_passes > 1:
        for ext, count in zip(merged_extractions, appearance_counts):
            ext.confidence_score = count / total_passes

    return merged_extractions


def _extractions_overlap(
    extraction1: data.Extraction, extraction2: data.Extraction
) -> bool:
    """Checks if two extractions overlap based on their character intervals.

    Args:
      extraction1: First extraction to compare.
      extraction2: Second extraction to compare.

    Returns:
      True if the extractions overlap, False otherwise.
    """
    if extraction1.char_interval is None or extraction2.char_interval is None:
        return False

    start1, end1 = (
        extraction1.char_interval.start_pos,
        extraction1.char_interval.end_pos,
    )
    start2, end2 = (
        extraction2.char_interval.start_pos,
        extraction2.char_interval.end_pos,
    )

    if start1 is None or end1 is None or start2 is None or end2 is None:
        return False

    # Two intervals overlap if one starts before the other ends
    return start1 < end2 and start2 < end1


def _document_chunk_iterator(
    documents: Iterable[data.Document],
    max_char_buffer: int,
    restrict_repeats: bool = True,
    tokenizer: tokenizer_lib.Tokenizer | None = None,
) -> Iterator[chunking.TextChunk]:
    """Iterates over documents to yield text chunks along with the document ID.

    Args:
      documents: A sequence of Document objects.
      max_char_buffer: The maximum character buffer size for the ChunkIterator.
      restrict_repeats: Whether to restrict the same document id from being
        visited more than once.
      tokenizer: Optional tokenizer instance.

    Yields:
      TextChunk containing document ID for a corresponding document.

    Raises:
      InvalidDocumentError: If restrict_repeats is True and the same document ID
        is visited more than once. Valid documents prior to the error will be
        returned.
    """
    visited_ids = set()
    for document in documents:
        if tokenizer:
            tokenized_text = tokenizer.tokenize(document.text or "")
        else:
            tokenized_text = document.tokenized_text
        document_id = document.document_id
        if restrict_repeats and document_id in visited_ids:
            raise exceptions.InvalidDocumentError(
                f"Document id {document_id} is already visited."
            )
        chunk_iter = chunking.ChunkIterator(
            text=tokenized_text,
            max_char_buffer=max_char_buffer,
            document=document,
            tokenizer_impl=tokenizer or tokenizer_lib.RegexTokenizer(),
        )
        visited_ids.add(document_id)

        yield from chunk_iter


class Annotator:
    """Annotates documents with extractions using a language model."""

    def __init__(
        self,
        language_model: base_model.BaseLanguageModel,
        prompt_template: prompting.PromptTemplateStructured,
        format_type: data.FormatType = data.FormatType.YAML,
        attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
        fence_output: bool = False,
        format_handler: fh.FormatHandler | None = None,
    ):
        """Initializes Annotator.

        Args:
          language_model: Model which performs language model inference.
          prompt_template: Structured prompt template where the answer is expected
            to be formatted text (YAML or JSON).
          format_type: The format type for the output (YAML or JSON).
          attribute_suffix: Suffix to append to attribute keys in the output.
          fence_output: Whether to expect/generate fenced output (```json or
            ```yaml). When True, the model is prompted to generate fenced output and
            the resolver expects it. When False, raw JSON/YAML is expected.
            Defaults to False. If format_handler is provided, it takes precedence.
          format_handler: Optional FormatHandler for managing format-specific logic.
        """
        self._language_model = language_model

        if format_handler is None:
            format_handler = fh.FormatHandler(
                format_type=format_type,
                use_wrapper=True,
                wrapper_key=data.EXTRACTIONS_KEY,
                use_fences=fence_output,
                attribute_suffix=attribute_suffix,
            )

        self._prompt_generator = prompting.QAPromptGenerator(
            template=prompt_template,
            format_handler=format_handler,
        )

        logging.debug("Annotator initialized with format_handler: %s", format_handler)

    def annotate_documents(
        self,
        documents: Iterable[data.Document],
        resolver: resolver_lib.AbstractResolver | None = None,
        max_char_buffer: int = 200,
        batch_length: int = 1,
        debug: bool = True,
        extraction_passes: int = 1,
        context_window_chars: int | None = None,
        show_progress: bool = True,
        tokenizer: tokenizer_lib.Tokenizer | None = None,
        **kwargs,
    ) -> Iterator[data.AnnotatedDocument]:
        """Annotates a sequence of documents with NLP extractions.

          Breaks documents into chunks, processes them into prompts and performs
          batched inference, mapping annotated extractions back to the original
          document. Batch processing is determined by batch_length, and can operate
          across documents for optimized throughput.

        Args:
          documents: Documents to annotate. Each document is expected to have a
            unique document_id.
          resolver: Resolver to use for extracting information from text.
          max_char_buffer: Max number of characters that we can run inference on.
            The text will be broken into chunks up to this length.
          batch_length: Number of chunks to process in a single batch.
          debug: Whether to populate debug fields.
          extraction_passes: Number of sequential extraction attempts to improve
            recall by finding additional entities. Defaults to 1, which performs
            standard single extraction.
            Values > 1 reprocess tokens multiple times, potentially increasing
            costs with the potential for a more thorough extraction.
          context_window_chars: Number of characters from the previous chunk to
            include as context for the current chunk. Helps with coreference
            resolution across chunk boundaries. Defaults to None (disabled).
          show_progress: Whether to show progress bar. Defaults to True.
          tokenizer: Optional tokenizer to use. If None, uses default tokenizer.
          **kwargs: Additional arguments passed to LanguageModel.infer and Resolver.

        Yields:
          Resolved annotations from input documents.

        Raises:
          ValueError: If there are no scored outputs during inference.
        """
        if resolver is None:
            resolver = resolver_lib.Resolver(format_type=data.FormatType.YAML)

        if extraction_passes == 1:
            yield from self._annotate_documents_single_pass(
                documents,
                resolver,
                max_char_buffer,
                batch_length,
                debug,
                show_progress,
                context_window_chars=context_window_chars,
                tokenizer=tokenizer,
                **kwargs,
            )
        else:
            yield from self._annotate_documents_sequential_passes(
                documents,
                resolver,
                max_char_buffer,
                batch_length,
                debug,
                extraction_passes,
                show_progress,
                context_window_chars=context_window_chars,
                tokenizer=tokenizer,
                **kwargs,
            )

    def _annotate_documents_single_pass(
        self,
        documents: Iterable[data.Document],
        resolver: resolver_lib.AbstractResolver,
        max_char_buffer: int,
        batch_length: int,
        debug: bool,
        show_progress: bool = True,
        context_window_chars: int | None = None,
        tokenizer: tokenizer_lib.Tokenizer | None = None,
        **kwargs,
    ) -> Iterator[data.AnnotatedDocument]:
        """Single-pass annotation with stable ordering and streaming emission.

        Streams input without full materialization, maintains correct attribution
        across batches, and emits completed documents immediately to minimize
        peak memory usage. Handles generators from both infer() and align().

        When context_window_chars is set, includes text from the previous chunk as
        context for coreference resolution across chunk boundaries.
        """
        doc_order: list[str] = []
        doc_text_by_id: dict[str, str] = {}
        per_doc: DefaultDict[str, list[data.Extraction]] = collections.defaultdict(list)
        next_emit_idx = 0

        def _capture_docs(src: Iterable[data.Document]) -> Iterator[data.Document]:
            """Captures document order and text lazily as chunks are produced."""
            for document in src:
                document_id = document.document_id
                if document_id in doc_text_by_id:
                    raise exceptions.InvalidDocumentError(
                        f"Duplicate document_id: {document_id}"
                    )
                doc_order.append(document_id)
                doc_text_by_id[document_id] = document.text or ""
                yield document

        def _emit_docs_iter(
            keep_last_doc: bool,
        ) -> Iterator[data.AnnotatedDocument]:
            """Yields documents that are guaranteed complete.

            Args:
              keep_last_doc: If True, retains the most recently started document
                for additional extractions. If False, emits all remaining documents.
            """
            nonlocal next_emit_idx
            limit = max(0, len(doc_order) - 1) if keep_last_doc else len(doc_order)
            while next_emit_idx < limit:
                document_id = doc_order[next_emit_idx]
                yield data.AnnotatedDocument(
                    document_id=document_id,
                    extractions=per_doc.get(document_id, []),
                    text=doc_text_by_id.get(document_id, ""),
                )
                per_doc.pop(document_id, None)
                doc_text_by_id.pop(document_id, None)
                next_emit_idx += 1

        chunk_iter = _document_chunk_iterator(
            _capture_docs(documents), max_char_buffer, tokenizer=tokenizer
        )
        batches = chunking.make_batches_of_textchunk(chunk_iter, batch_length)

        model_info = progress.get_model_info(self._language_model)
        batch_iter = progress.create_extraction_progress_bar(
            batches, model_info=model_info, disable=not show_progress
        )

        chars_processed = 0

        prompt_builder = prompting.ContextAwarePromptBuilder(
            generator=self._prompt_generator,
            context_window_chars=context_window_chars,
        )

        try:
            for batch in batch_iter:
                if not batch:
                    continue

                prompts = [
                    prompt_builder.build_prompt(
                        chunk.chunk_text, chunk.document_id, chunk.additional_context
                    )
                    for chunk in batch
                ]

                if show_progress:
                    current_chars = sum(
                        len(text_chunk.chunk_text) for text_chunk in batch
                    )
                    try:
                        batch_iter.set_description(
                            progress.format_extraction_progress(
                                model_info,
                                current_chars=current_chars,
                                processed_chars=chars_processed,
                            )
                        )
                    except AttributeError:
                        pass

                outputs = self._language_model.infer(batch_prompts=prompts, **kwargs)
                if not isinstance(outputs, list):
                    outputs = list(outputs)

                for text_chunk, scored_outputs in zip(batch, outputs):
                    if not isinstance(scored_outputs, list):
                        scored_outputs = list(scored_outputs)
                    if not scored_outputs:
                        raise exceptions.InferenceOutputError(
                            "No scored outputs from language model."
                        )

                    resolved_extractions = resolver.resolve(
                        scored_outputs[0].output, debug=debug, **kwargs
                    )

                    token_offset = (
                        text_chunk.token_interval.start_index
                        if text_chunk.token_interval
                        else 0
                    )
                    char_offset = (
                        text_chunk.char_interval.start_pos
                        if text_chunk.char_interval
                        else 0
                    )

                    aligned_extractions = resolver.align(
                        resolved_extractions,
                        text_chunk.chunk_text,
                        token_offset,
                        char_offset,
                        tokenizer_inst=tokenizer,
                        **kwargs,
                    )

                    for extraction in aligned_extractions:
                        per_doc[text_chunk.document_id].append(extraction)

                    if show_progress and text_chunk.char_interval is not None:
                        chars_processed += (
                            text_chunk.char_interval.end_pos
                            - text_chunk.char_interval.start_pos
                        )

                yield from _emit_docs_iter(keep_last_doc=True)

        finally:
            batch_iter.close()

        yield from _emit_docs_iter(keep_last_doc=False)

    def _annotate_documents_sequential_passes(
        self,
        documents: Iterable[data.Document],
        resolver: resolver_lib.AbstractResolver,
        max_char_buffer: int,
        batch_length: int,
        debug: bool,
        extraction_passes: int,
        show_progress: bool = True,
        context_window_chars: int | None = None,
        tokenizer: tokenizer_lib.Tokenizer | None = None,
        **kwargs,
    ) -> Iterator[data.AnnotatedDocument]:
        """Sequential extraction passes with early stopping.

        Stops early when a pass adds zero new non-overlapping
        extractions, saving unnecessary LLM API calls.
        """

        logging.info(
            "Starting sequential extraction passes for improved recall with %d"
            " passes.",
            extraction_passes,
        )

        document_list = list(documents)

        document_extractions_by_pass: dict[str, list[list[data.Extraction]]] = {}
        document_texts: dict[str, str] = {}
        # Preserve text up-front so we can emit documents even if later passes
        # produce no extractions.
        for _doc in document_list:
            document_texts[_doc.document_id] = _doc.text or ""

        prev_merged_count = 0

        for pass_num in range(extraction_passes):
            logging.info(
                "Starting extraction pass %d of %d", pass_num + 1, extraction_passes
            )

            for annotated_doc in self._annotate_documents_single_pass(
                document_list,
                resolver,
                max_char_buffer,
                batch_length,
                debug=(debug and pass_num == 0),
                show_progress=show_progress if pass_num == 0 else False,
                context_window_chars=context_window_chars,
                tokenizer=tokenizer,
                **kwargs,
            ):
                doc_id = annotated_doc.document_id

                if doc_id not in document_extractions_by_pass:
                    document_extractions_by_pass[doc_id] = []

                document_extractions_by_pass[doc_id].append(
                    annotated_doc.extractions or []
                )

            # ── Early stopping: merge once, cache count ──
            # We merge here and stash the per-doc merged lists so the
            # final emit phase can reuse them (with confidence scoring)
            # instead of re-merging from scratch.
            merged_cache: dict[str, list[data.Extraction]] = {}
            current_merged_count = 0
            for doc_id, all_pass_exts in document_extractions_by_pass.items():
                merged = _merge_non_overlapping_extractions(all_pass_exts)
                merged_cache[doc_id] = merged
                current_merged_count += len(merged)

            if pass_num > 0 and current_merged_count == prev_merged_count:
                logging.info(
                    "Early stop: pass %d added 0 new extractions.",
                    pass_num + 1,
                )
                break
            prev_merged_count = current_merged_count

        # Emit results strictly in original input order.
        # Re-merge with ``total_passes`` to assign confidence scores.
        for doc in document_list:
            doc_id = doc.document_id
            all_pass_extractions = document_extractions_by_pass.get(doc_id, [])
            actual_passes = len(all_pass_extractions)
            merged_extractions = _merge_non_overlapping_extractions(
                all_pass_extractions,
                total_passes=actual_passes,
            )

            if debug:
                total_extractions = sum(
                    len(extractions) for extractions in all_pass_extractions
                )
                logging.info(
                    "Document %s: Merged %d extractions from %d passes into "
                    "%d non-overlapping extractions.",
                    doc_id,
                    total_extractions,
                    actual_passes,
                    len(merged_extractions),
                )

            yield data.AnnotatedDocument(
                document_id=doc_id,
                extractions=merged_extractions,
                text=document_texts.get(doc_id, doc.text or ""),
            )

        logging.info("Sequential extraction passes completed.")

    def annotate_text(
        self,
        text: str,
        resolver: resolver_lib.AbstractResolver | None = None,
        max_char_buffer: int = 200,
        batch_length: int = 1,
        additional_context: str | None = None,
        debug: bool = True,
        extraction_passes: int = 1,
        context_window_chars: int | None = None,
        show_progress: bool = True,
        tokenizer: tokenizer_lib.Tokenizer | None = None,
        **kwargs,
    ) -> data.AnnotatedDocument:
        """Annotates text with NLP extractions for text input.

        Args:
          text: Source text to annotate.
          resolver: Resolver to use for extracting information from text.
          max_char_buffer: Max number of characters that we can run inference on.
            The text will be broken into chunks up to this length.
          batch_length: Number of chunks to process in a single batch.
          additional_context: Additional context to supplement prompt instructions.
          debug: Whether to populate debug fields.
          extraction_passes: Number of sequential extraction passes to improve
            recall by finding additional entities. Defaults to 1, which performs
            standard single extraction. Values > 1 reprocess tokens multiple times,
            potentially increasing costs.
          context_window_chars: Number of characters from the previous chunk to
            include as context for coreference resolution. Defaults to None
            (disabled).
          show_progress: Whether to show progress bar. Defaults to True.
          tokenizer: Optional tokenizer instance.
          **kwargs: Additional arguments for inference and resolver_lib.

        Returns:
          Resolved annotations from text for document.
        """
        if resolver is None:
            resolver = resolver_lib.Resolver(
                format_type=data.FormatType.YAML,
            )

        start_time = time.time() if debug else None

        documents = [
            data.Document(
                text=text,
                document_id=None,
                additional_context=additional_context,
            )
        ]

        annotations = list(
            self.annotate_documents(
                documents=documents,
                resolver=resolver,
                max_char_buffer=max_char_buffer,
                batch_length=batch_length,
                debug=debug,
                extraction_passes=extraction_passes,
                context_window_chars=context_window_chars,
                show_progress=show_progress,
                tokenizer=tokenizer,
                **kwargs,
            )
        )
        assert (
            len(annotations) == 1
        ), f"Expected 1 annotation but got {len(annotations)} annotations."

        if debug and annotations[0].extractions:
            elapsed_time = time.time() - start_time if start_time else None
            num_extractions = len(annotations[0].extractions)
            unique_classes = len(
                set(e.extraction_class for e in annotations[0].extractions)
            )
            num_chunks = len(text) // max_char_buffer + (
                1 if len(text) % max_char_buffer else 0
            )

            progress.print_extraction_summary(
                num_extractions,
                unique_classes,
                elapsed_time=elapsed_time,
                chars_processed=len(text),
                num_chunks=num_chunks,
            )

        return data.AnnotatedDocument(
            document_id=annotations[0].document_id,
            extractions=annotations[0].extractions,
            text=annotations[0].text,
        )

    # ── Async API ──────────────────────────────────────────────────────

    async def async_annotate_documents(
        self,
        documents: Iterable[data.Document],
        resolver: resolver_lib.AbstractResolver | None = None,
        max_char_buffer: int = 200,
        batch_length: int = 1,
        debug: bool = True,
        extraction_passes: int = 1,
        context_window_chars: int | None = None,
        show_progress: bool = True,
        tokenizer: tokenizer_lib.Tokenizer | None = None,
        **kwargs,
    ) -> list[data.AnnotatedDocument]:
        """Async version of ``annotate_documents``.

        Uses ``BaseLanguageModel.async_infer`` for non-blocking LLM calls
        and pipelines inference with alignment so that batch N+1 inference
        overlaps with batch N alignment.

        Args:
          documents: Documents to annotate.
          resolver: Resolver to use for extracting information from text.
          max_char_buffer: Max characters per inference chunk.
          batch_length: Number of chunks processed per batch.
          debug: Whether to populate debug fields.
          extraction_passes: Number of sequential extraction passes.
          context_window_chars: Context overlap between chunks.
          show_progress: Whether to show progress bar.
          tokenizer: Optional tokenizer instance.
          **kwargs: Additional arguments passed to the model and resolver.

        Returns:
          List of AnnotatedDocument results.
        """
        if resolver is None:
            resolver = resolver_lib.Resolver(format_type=data.FormatType.YAML)

        if extraction_passes == 1:
            return await self._async_annotate_documents_single_pass(
                documents,
                resolver,
                max_char_buffer,
                batch_length,
                debug,
                show_progress,
                context_window_chars=context_window_chars,
                tokenizer=tokenizer,
                **kwargs,
            )
        else:
            return await self._async_annotate_documents_sequential_passes(
                documents,
                resolver,
                max_char_buffer,
                batch_length,
                debug,
                extraction_passes,
                show_progress,
                context_window_chars=context_window_chars,
                tokenizer=tokenizer,
                **kwargs,
            )

    async def _async_annotate_documents_single_pass(
        self,
        documents: Iterable[data.Document],
        resolver: resolver_lib.AbstractResolver,
        max_char_buffer: int,
        batch_length: int,
        debug: bool,
        show_progress: bool = True,
        context_window_chars: int | None = None,
        tokenizer: tokenizer_lib.Tokenizer | None = None,
        **kwargs,
    ) -> list[data.AnnotatedDocument]:
        """Async single-pass annotation with I/O-CPU pipelining.

        Fires ``async_infer`` for the current batch while running alignment
        for the previous batch via ``asyncio.to_thread``, achieving overlap
        between I/O-bound and CPU-bound work.
        """
        doc_order: list[str] = []
        doc_text_by_id: dict[str, str] = {}
        per_doc: DefaultDict[str, list[data.Extraction]] = collections.defaultdict(list)

        def _capture_docs(
            src: Iterable[data.Document],
        ) -> Iterator[data.Document]:
            for document in src:
                document_id = document.document_id
                if document_id in doc_text_by_id:
                    raise exceptions.InvalidDocumentError(
                        f"Duplicate document_id: {document_id}"
                    )
                doc_order.append(document_id)
                doc_text_by_id[document_id] = document.text or ""
                yield document

        chunk_iter = _document_chunk_iterator(
            _capture_docs(documents), max_char_buffer, tokenizer=tokenizer
        )
        batches = list(chunking.make_batches_of_textchunk(chunk_iter, batch_length))

        prompt_builder = prompting.ContextAwarePromptBuilder(
            generator=self._prompt_generator,
            context_window_chars=context_window_chars,
        )

        # Pipeline: run alignment for batch N while inference runs for batch N+1
        prev_align_task: asyncio.Task | None = None

        def _align_batch(
            batch: list,
            outputs: list,
        ) -> None:
            """CPU-bound: resolve + align a batch (runs in thread)."""
            for text_chunk, scored_outputs in zip(batch, outputs):
                if not isinstance(scored_outputs, list):
                    scored_outputs = list(scored_outputs)
                if not scored_outputs:
                    raise exceptions.InferenceOutputError(
                        "No scored outputs from language model."
                    )

                resolved_extractions = resolver.resolve(
                    scored_outputs[0].output, debug=debug, **kwargs
                )

                token_offset = (
                    text_chunk.token_interval.start_index
                    if text_chunk.token_interval
                    else 0
                )
                char_offset = (
                    text_chunk.char_interval.start_pos
                    if text_chunk.char_interval
                    else 0
                )

                aligned_extractions = resolver.align(
                    resolved_extractions,
                    text_chunk.chunk_text,
                    token_offset,
                    char_offset,
                    tokenizer_inst=tokenizer,
                    **kwargs,
                )

                for extraction in aligned_extractions:
                    per_doc[text_chunk.document_id].append(extraction)

        for batch in batches:
            if not batch:
                continue

            prompts = [
                prompt_builder.build_prompt(
                    chunk.chunk_text, chunk.document_id, chunk.additional_context
                )
                for chunk in batch
            ]

            # Fire async inference
            outputs = await self._language_model.async_infer(
                batch_prompts=prompts, **kwargs
            )
            if not isinstance(outputs, list):
                outputs = list(outputs)

            # Wait for previous alignment to finish before starting new one
            if prev_align_task is not None:
                await prev_align_task

            # Schedule alignment in a thread (CPU-bound work)
            prev_align_task = asyncio.ensure_future(
                asyncio.to_thread(_align_batch, batch, outputs)
            )

        # Wait for the final alignment
        if prev_align_task is not None:
            await prev_align_task

        # Build results in document order
        results: list[data.AnnotatedDocument] = []
        for document_id in doc_order:
            results.append(
                data.AnnotatedDocument(
                    document_id=document_id,
                    extractions=per_doc.get(document_id, []),
                    text=doc_text_by_id.get(document_id, ""),
                )
            )
        return results

    async def _async_annotate_documents_sequential_passes(
        self,
        documents: Iterable[data.Document],
        resolver: resolver_lib.AbstractResolver,
        max_char_buffer: int,
        batch_length: int,
        debug: bool,
        extraction_passes: int,
        show_progress: bool = True,
        context_window_chars: int | None = None,
        tokenizer: tokenizer_lib.Tokenizer | None = None,
        **kwargs,
    ) -> list[data.AnnotatedDocument]:
        """Async sequential extraction passes with early stopping.

        Stops early when a pass adds zero new non-overlapping extractions,
        saving unnecessary LLM API calls.
        """
        logging.info(
            "Starting async sequential extraction passes (%d passes).",
            extraction_passes,
        )

        document_list = list(documents)
        document_extractions_by_pass: dict[str, list[list[data.Extraction]]] = {}
        document_texts: dict[str, str] = {}
        for _doc in document_list:
            document_texts[_doc.document_id] = _doc.text or ""

        prev_merged_count = 0

        for pass_num in range(extraction_passes):
            logging.info(
                "Starting extraction pass %d of %d",
                pass_num + 1,
                extraction_passes,
            )

            pass_results = await self._async_annotate_documents_single_pass(
                document_list,
                resolver,
                max_char_buffer,
                batch_length,
                debug=(debug and pass_num == 0),
                show_progress=show_progress if pass_num == 0 else False,
                context_window_chars=context_window_chars,
                tokenizer=tokenizer,
                **kwargs,
            )

            for annotated_doc in pass_results:
                doc_id = annotated_doc.document_id
                if doc_id not in document_extractions_by_pass:
                    document_extractions_by_pass[doc_id] = []
                document_extractions_by_pass[doc_id].append(
                    annotated_doc.extractions or []
                )

            # ── Early stopping: merge once, cache count ──
            merged_cache: dict[str, list[data.Extraction]] = {}
            current_merged_count = 0
            for doc_id, all_pass_exts in document_extractions_by_pass.items():
                merged = _merge_non_overlapping_extractions(all_pass_exts)
                merged_cache[doc_id] = merged
                current_merged_count += len(merged)

            if pass_num > 0 and current_merged_count == prev_merged_count:
                logging.info(
                    "Early stop: pass %d added 0 new extractions.",
                    pass_num + 1,
                )
                break
            prev_merged_count = current_merged_count

        # Emit results in original order.
        # Re-merge with ``total_passes`` to assign confidence scores.
        results: list[data.AnnotatedDocument] = []
        for doc in document_list:
            doc_id = doc.document_id
            all_pass_extractions = document_extractions_by_pass.get(doc_id, [])
            actual_passes = len(all_pass_extractions)
            merged_extractions = _merge_non_overlapping_extractions(
                all_pass_extractions,
                total_passes=actual_passes,
            )

            if debug:
                total_extractions = sum(len(exts) for exts in all_pass_extractions)
                logging.info(
                    "Document %s: Merged %d extractions from %d passes into "
                    "%d non-overlapping extractions.",
                    doc_id,
                    total_extractions,
                    actual_passes,
                    len(merged_extractions),
                )

            results.append(
                data.AnnotatedDocument(
                    document_id=doc_id,
                    extractions=merged_extractions,
                    text=document_texts.get(doc_id, doc.text or ""),
                )
            )

        logging.info("Async sequential extraction passes completed.")
        return results

    async def async_annotate_text(
        self,
        text: str,
        resolver: resolver_lib.AbstractResolver | None = None,
        max_char_buffer: int = 200,
        batch_length: int = 1,
        additional_context: str | None = None,
        debug: bool = True,
        extraction_passes: int = 1,
        context_window_chars: int | None = None,
        show_progress: bool = True,
        tokenizer: tokenizer_lib.Tokenizer | None = None,
        **kwargs,
    ) -> data.AnnotatedDocument:
        """Async version of ``annotate_text``.

        Args:
          text: Source text to annotate.
          resolver: Resolver to use for extracting information from text.
          max_char_buffer: Max characters per inference chunk.
          batch_length: Number of chunks per batch.
          additional_context: Additional context for the prompt.
          debug: Whether to populate debug fields.
          extraction_passes: Number of sequential extraction passes.
          context_window_chars: Context overlap between chunks.
          show_progress: Whether to show progress bar.
          tokenizer: Optional tokenizer instance.
          **kwargs: Additional arguments for inference and resolver.

        Returns:
          An AnnotatedDocument with the extracted information.
        """
        if resolver is None:
            resolver = resolver_lib.Resolver(format_type=data.FormatType.YAML)

        start_time = time.time() if debug else None

        documents = [
            data.Document(
                text=text,
                document_id=None,
                additional_context=additional_context,
            )
        ]

        annotations = await self.async_annotate_documents(
            documents=documents,
            resolver=resolver,
            max_char_buffer=max_char_buffer,
            batch_length=batch_length,
            debug=debug,
            extraction_passes=extraction_passes,
            context_window_chars=context_window_chars,
            show_progress=show_progress,
            tokenizer=tokenizer,
            **kwargs,
        )
        assert (
            len(annotations) == 1
        ), f"Expected 1 annotation but got {len(annotations)} annotations."

        if debug and annotations[0].extractions:
            elapsed_time = time.time() - start_time if start_time else None
            num_extractions = len(annotations[0].extractions)
            unique_classes = len(
                set(e.extraction_class for e in annotations[0].extractions)
            )
            num_chunks = len(text) // max_char_buffer + (
                1 if len(text) % max_char_buffer else 0
            )

            progress.print_extraction_summary(
                num_extractions,
                unique_classes,
                elapsed_time=elapsed_time,
                chars_processed=len(text),
                num_chunks=num_chunks,
            )

        return data.AnnotatedDocument(
            document_id=annotations[0].document_id,
            extractions=annotations[0].extractions,
            text=annotations[0].text,
        )
