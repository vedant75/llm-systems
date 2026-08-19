from __future__ import annotations

from collections.abc import Iterable, Iterator

from .bpe_optimized_helper import (
    PAT,
    pretokenize_text,
    merge_word,
    split_on_special_tokens,
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:

        self.vocab = vocab
        self.merges = merges

        self.special_tokens = (
            special_tokens
            if special_tokens is not None
            else []
        )

        # bytes token -> token id
        self.inverse_vocab = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
        }

        # Lower rank = merge was learned earlier.
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

    def _apply_bpe(
        self,
        word: tuple[bytes, ...],
    ) -> tuple[bytes, ...]:

        while len(word) > 1:

            best_pair = None
            best_rank = None

            for pair in zip(word, word[1:]):

                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            word = merge_word(
                word,
                best_pair,
            )

        return word

    def _encode_text_iter(
        self,
        text: str,
    ) -> Iterator[int]:

        segments = split_on_special_tokens(
            text,
            self.special_tokens,
        )

        for segment, is_special in segments:

            if is_special:
                special_bytes = segment.encode("utf-8")
                yield self.inverse_vocab[special_bytes]

                continue


            pretokens = pretokenize_text(segment)

            for word in pretokens:

                word = self._apply_bpe(word)

                for symbol in word:
                    yield self.inverse_vocab[symbol]

    def encode(
        self,
        text: str,
    ) -> list[int]:

        return list(
            self._encode_text_iter(text)
        )

    def _safe_prefix_end(
        self,
        buffer: str,
    ) -> int:

        if not buffer:
            return 0

        segments = split_on_special_tokens(
            buffer,
            self.special_tokens,
        )

        ordinary_hold_start = len(buffer)

        if segments:
            last_segment, is_special = segments[-1]

            if not is_special:

                last_segment_start = (
                    len(buffer) - len(last_segment)
                )

                last_match = None

                for match in PAT.finditer(last_segment):
                    last_match = match

                if (
                    last_match is not None
                    and last_match.end() == len(last_segment)
                ):
                    ordinary_hold_start = (
                        last_segment_start
                        + last_match.start()
                    )

        special_hold_start = len(buffer)

        if self.special_tokens:

            max_special_length = max(
                len(token)
                for token in self.special_tokens
            )

            max_suffix_length = min(
                len(buffer),
                max_special_length - 1,
            )

            for suffix_length in range(
                max_suffix_length,
                0,
                -1,
            ):
                suffix = buffer[-suffix_length:]

                if any(
                    len(token) > suffix_length
                    and token.startswith(suffix)
                    for token in self.special_tokens
                ):
                    special_hold_start = (
                        len(buffer) - suffix_length
                    )
                    break
        return min(
            ordinary_hold_start,
            special_hold_start,
        )

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:

        buffer = ""

        for chunk in iterable:

            if not chunk:
                continue

            buffer += chunk

            safe_end = self._safe_prefix_end(
                buffer
            )

            if safe_end == 0:
                continue

            safe_text = buffer[:safe_end]

            yield from self._encode_text_iter(
                safe_text
            )

            buffer = buffer[safe_end:]

        if buffer:
            yield from self._encode_text_iter(
                buffer
            )

    def decode(
        self,
        ids: list[int],
    ) -> str:

        result = []

        for token_id in ids:
            result.append(
                self.vocab[token_id]
            )

        return b"".join(result).decode(
            "utf-8",
            errors="replace",
        )