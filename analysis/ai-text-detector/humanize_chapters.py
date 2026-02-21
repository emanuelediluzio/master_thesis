import json
import random
import re
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
LATEX_PATH = ROOT / "tesi_svg_captioning.tex"
DATA_DIR = ROOT / "analysis" / "AI-Text-Humanizer"

CHAPTERS = [
    "Introduction",
    "State of the Art",
    "System Architecture",
    "Fine-tuning Techniques",
    "Multimodal Extensions",
    "Experimental Methodology",
    "Results and Analysis",
    "Applications and Impact",
    "Conclusions and Future Developments",
]

WORD_PATTERN = re.compile(r"\\b([A-Za-z][A-Za-z\-']*)\\b")

SKIP_TOKENS = {"svg", "svgs"}
SKIP_PARAGRAPH_TOKENS = ["\\texttt", "\\url", "\\allowbreak", "\\path", "\\includegraphics"]
PROTECTED_COMMAND_PREFIXES = ("\\",)

REPLACE_PROBABILITY = 0.25
RANDOM_SEED = 42


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_resources() -> tuple[Dict[str, List[str]], Set[str], Set[str]]:
    vocab = load_json(DATA_DIR / "eng_synonyms.json")
    stop_words_list = load_json(DATA_DIR / "stop_words.json")
    fixed_terms_list = load_json(DATA_DIR / "fixedterms.json")
    stop_words = {w.lower() for w in stop_words_list}
    fixed_terms = {w.lower() for w in fixed_terms_list}
    return vocab, stop_words, fixed_terms


VOCAB, STOP_WORDS, FIXED_TERMS = load_resources()


def should_skip_paragraph(paragraph: str) -> bool:
    norm = paragraph.strip()
    if not norm:
        return True
    lower = norm.lower()
    if any(token in paragraph for token in SKIP_PARAGRAPH_TOKENS):
        return True
    if "\\begin{" in paragraph or "\\end{" in paragraph:
        return True
    if norm.startswith("\\"):
        return False
    return False


def is_command(match_start: int, text: str) -> bool:
    if match_start == 0:
        return False
    prev_char = text[match_start - 1]
    if prev_char == "\\":
        return True
    if prev_char == '{' and match_start >= 2 and text[match_start - 2] == '\\':
        return True
    return False


def is_proper_noun(word: str) -> bool:
    if len(word) <= 1:
        return False
    return word[0].isupper() and word[1:].islower()


def normalise_synonym(word: str, template: str) -> str:
    if template.isupper():
        return word.upper()
    if template[0].isupper():
        return word.capitalize()
    return word


def pick_synonym(word: str, idx: int) -> str | None:
    lower = word.lower()
    if lower not in VOCAB:
        return None
    candidates = []
    for synonym in VOCAB[lower]:
        if not synonym:
            continue
        syn = synonym.strip()
        if syn.lower() == lower:
            continue
        if any(ch.isdigit() for ch in syn):
            continue
        if " " in syn:
            continue
        if syn.startswith("-") or syn.endswith("-"):
            continue
        if syn.lower() in STOP_WORDS or syn.lower() in FIXED_TERMS:
            continue
        if not re.fullmatch(r"[A-Za-z\-']+", syn):
            continue
        candidates.append(syn)
    if not candidates:
        return None
    # deterministic choice based on hash
    index = (hash((lower, idx)) & 0xFFFFFFFF) % len(candidates)
    return normalise_synonym(candidates[index], word)


def should_replace(word: str, idx: int) -> bool:
    lower = word.lower()
    if lower in STOP_WORDS:
        return False
    if lower in FIXED_TERMS:
        return False
    if lower in SKIP_TOKENS:
        return False
    if any(ch.isdigit() for ch in word):
        return False
    if word.isupper():
        return False
    if is_proper_noun(word):
        return False
    rng_value = (hash((lower, idx, RANDOM_SEED)) & 0xFFFFFFFF) / 0xFFFFFFFF
    return rng_value < REPLACE_PROBABILITY


def humanize_paragraph(paragraph: str, seed_offset: int) -> str:
    idx_counter = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx_counter
        word = match.group(0)
        start = match.start()
        if is_command(start, paragraph):
            return word
        if should_replace(word, idx_counter + seed_offset):
            new_word = pick_synonym(word, idx_counter + seed_offset)
            idx_counter += 1
            return new_word or word
        idx_counter += 1
        return word

    return WORD_PATTERN.sub(repl, paragraph)


def rewrite_body(body: str, chapter_index: int) -> str:
    parts = re.split(r"(\n\s*\n)", body)
    new_parts: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            new_parts.append(part)
            continue
        if should_skip_paragraph(part):
            new_parts.append(part)
            continue
        new_parts.append(humanize_paragraph(part, chapter_index * 1000 + i))
    return ''.join(new_parts)


def process_chapter(text: str, chapter_name: str, chapter_index: int) -> str:
    pattern = re.compile(
        rf"(\\chapter\{{{re.escape(chapter_name)}\}})(.*?)(?=\\chapter\{{|\\appendix|\\end\{{document\}})",
        re.S,
    )

    def repl(match: re.Match[str]) -> str:
        header = match.group(1)
        body = match.group(2)
        new_body = rewrite_body(body, chapter_index)
        return header + new_body

    text, count = pattern.subn(repl, text, count=1)
    if count == 0:
        print(f"[WARN] Chapter '{chapter_name}' not found")
    return text


def main():
    text = LATEX_PATH.read_text(encoding="utf-8")
    for idx, chapter in enumerate(CHAPTERS):
        text = process_chapter(text, chapter, idx)
    LATEX_PATH.write_text(text, encoding="utf-8")
    print("Humanization pass completed.")


if __name__ == "__main__":
    main()
