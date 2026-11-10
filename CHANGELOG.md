# Changelog

All notable changes to nbxgrammar are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - unreleased (on main; published with the next version that changes what a user sees)

### Changed
- **Nothing a caller sees**: the same masks, the same C ABI (the ten exported symbols, unchanged), the same speed
  (about 92 µs a step on Qwen3's 151 936 tokens over the oracle's walks, before and after). Inside, the brick follows
  NeuroBrix's bricks rule: the matcher's frames are its own (the vocabulary and the C ABI ask it `in_plain_string`,
  `finish`, and keep an opaque `State` that `Matcher::resume` refuses against another grammar, by name), and no
  function is longer than 53 code lines — `Matcher::feed` was 173, now one method per frame kind; the schema
  compiler's keyword check, literals, choices and repetitions are named functions.

### Added
- `tests/golden_masks.rs`: the masks of fixed walks over ten grammars, hashed and recorded on 0.1.0 before the change
  above, reproduced by it with the file untouched; `tests/contract.rs`: the matcher's contract from outside the crate.
- CI denies `clippy::too_many_lines` at the threshold of `clippy.toml`.

## [0.1.0] - unreleased

### Added
- **The grammar engine**: a JSON Schema subset and grammar descriptions (literals, schemas, sequences, choices,
  repetitions, free text up to a terminator or a special token, token ids) compiled into a matcher; per-step token
  masks over a tokenizer's byte vocabulary, string-safe and free-text tokens cached once per vocabulary; any constraint
  it cannot enforce refused by name.
- **The Python package**: `Grammar.json_schema`, `Grammar.description`, `Vocab`, `Matcher.mask / advance / accepting`,
  through the library's C ABI with `ctypes`; one wheel per platform (`py3-none-<platform>`) for Linux x86_64 and
  aarch64, macOS arm64 and x86_64, Windows x86_64.
