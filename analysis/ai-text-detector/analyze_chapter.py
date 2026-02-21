import argparse
import re
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "desklib/ai-text-detector-v1.01"
LATEX_FILE = Path.home() / "Desktop/universita/tesi_magistrale/tesi_svg_captioning.tex"
MAX_LEN = 768
THRESHOLD = 0.5


def locate_chapter(text: str, chapter_name: str) -> str:
    pattern = rf"\\chapter\{{{re.escape(chapter_name)}\}}(.*?)(\\chapter\\*?\{{|\\end\{{document\}})"
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError(f"Capitolo '{chapter_name}' non trovato")
    content = match.group(1)
    return content


SKIP_ENV_PATTERNS = (
    "\\begin{table}",
    "\\begin{tabular}",
    "\\begin{figure}",
    "\\begin{algorithm}",
    "\\begin{lstlisting}",
)


def split_paragraphs(block: str) -> List[str]:
    # split on blank lines but keep paragraphs with actual prose
    fragments = [frag.strip() for frag in block.split("\n\n")]
    paragraphs: List[str] = []
    for frag in fragments:
        if not frag:
            continue
        if any(pattern in frag for pattern in SKIP_ENV_PATTERNS):
            continue
        if frag.lstrip().startswith('\\chapter'):
            continue
        paragraphs.append(frag)
    return paragraphs


def clean_latex(paragraph: str) -> str:
    cleaned = re.sub(r"%.*", "", paragraph)
    cleaned = re.sub(r"\\cite\{[^}]+\}", "", cleaned)
    cleaned = re.sub(r"\\ref\{[^}]+\}", "", cleaned)
    cleaned = re.sub(r"\$[^$]*\$", "", cleaned)
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, trust_remote_code=True, num_labels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def predict(paragraph: str, tokenizer, model, device) -> float:
    encoded = tokenizer(
        paragraph,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
    prob = torch.sigmoid(logits)[0, 0].item()
    return prob


def analyze_chapter(name: str, tokenizer, model, device) -> List[Tuple[int, float, str]]:
    latex = LATEX_FILE.read_text(encoding="utf-8", errors="ignore")
    content = locate_chapter(latex, name)
    raw_paragraphs = split_paragraphs(content)

    results = []
    for idx, raw in enumerate(raw_paragraphs, start=1):
        cleaned = clean_latex(raw)
        alpha_count = sum(ch.isalpha() for ch in cleaned)
        if not cleaned or alpha_count < 30:
            continue
        prob = predict(cleaned, tokenizer, model, device)
        results.append((idx, prob, cleaned))
    return results


def main():
    parser = argparse.ArgumentParser(description="Analizza i paragrafi di un capitolo con Desklib AI Detector")
    parser.add_argument("chapter", help="Nome esatto del capitolo (ad es. 'Introduction')")
    parser.add_argument("--out", default=None, help="File di output (default stdout)")
    args = parser.parse_args()

    tokenizer, model, device = load_model()

    try:
        results = analyze_chapter(args.chapter, tokenizer, model, device)
    except ValueError as exc:
        print(exc)
        return

    lines = []
    header = f"Risultati Capitolo: {args.chapter}\nTrovati {len(results)} paragrafi\n"
    lines.append(header)
    for idx, prob, text in results:
        label = "AI-generated" if prob >= THRESHOLD else "Human-like"
        lines.append(f"Paragrafo {idx:02d} | Probabilita AI: {prob:.2%} | Etichetta: {label}")
        lines.append(text)
        lines.append("-" * 100)

    output = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
