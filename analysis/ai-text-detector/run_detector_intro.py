import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "desklib/ai-text-detector-v1.01"
LATEX_PATH = Path.home() / "Desktop/universita/tesi_magistrale/tesi_svg_captioning.tex"
MAX_LEN = 768
THRESHOLD = 0.5


def clean_latex(latex_block: str) -> str:
    block = re.sub(r"%.*", "", latex_block)
    block = re.sub(r"\\cite\{[^}]+\}", "", block)
    block = re.sub(r"\\ref\{[^}]+\}", "", block)
    block = re.sub(r"\$[^$]*\$", "", block)
    block = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", block)
    block = block.replace("{", " ").replace("}", " ")
    block = re.sub(r"\s+", " ", block)
    return block.strip()


def predict(text: str, model, tokenizer, device) -> float:
    encoded = tokenizer(
        text,
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


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    latex = LATEX_PATH.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"\\chapter\{Introduction\}(.*?)\\chapter\{State of the Art\}", latex, re.S)
    if not match:
        raise SystemExit("Capitolo 1 non trovato nel file LaTeX.")

    chapter_text = match.group(1)
    raw_paragraphs = [frag.strip() for frag in chapter_text.split("\n\n") if frag.strip()]

    print(f"Trovati {len(raw_paragraphs)} paragrafi nel Capitolo 1\n")

    for idx, raw in enumerate(raw_paragraphs, 1):
        paragraph = clean_latex(raw)
        if not paragraph:
            continue
        prob = predict(paragraph, model, tokenizer, device)
        label = "AI-generated" if prob >= THRESHOLD else "Human-like"
        print(f"Paragrafo {idx:02d} | Probabilita AI: {prob:.2%} | Etichetta: {label}")
        print(paragraph)
        print("-" * 100)


if __name__ == "__main__":
    main()
