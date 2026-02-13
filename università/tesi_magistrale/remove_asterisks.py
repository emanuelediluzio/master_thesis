import re
import os

target_files = [
    "/Users/emanuelediluzio/Desktop/università/tesi_magistrale/gallery.tex"
]

def apply_substitutions(content):
    # Pattern: asterisk, optional whitespace, \textbf
    # We want to remove the asterisk and the whitespace before \textbf
    # But checking if it's start of line or preceded by space might be good to avoid messing up math?
    # Actually, looking at the grep output: " * \textbf{Polygon 1}:"
    # It acts as a separator.
    # Replacement: just "\textbf" (plus ensure a preceding space if it wasn't start of line?)
    # If I replace " * \textbf" with " \textbf", it keeps the separator nature (space).
    # If I replace "* \textbf" with "\textbf", I might merge words.
    # E.g. "Text.* \textbf{Head}" -> "Text.\textbf{Head}". 
    # Usually we want a space.
    # If the original was "* \textbf", replacing with "\textbf" leaves just the space before *.
    # Example: "scheme. * \textbf{Polygon 1}" -> "scheme. \textbf{Polygon 1}".
    # Example starting line: "        * \textbf{Base:}" -> "        \textbf{Base:}"
    
    # So replacing `* \textbf` with `\textbf` seems correct, preserving surrounding spaces?
    # No, `* ` is 2 chars. `\textbf` is 7 chars.
    # If I have `... * \textbf{...}`, I want `... \textbf{...}`.
    # So I replace `\* \textbf` with `\textbf`.
    
    # Regex: `\*\s*\\textbf`
    # Replace with `\\textbf`
    
    # Let's be safe and check for the context.
    
    new_content = re.sub(r'\*\s*\\textbf', r'\\textbf', content)
    
    return new_content

def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (not found)")
        return

    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = apply_substitutions(content)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path}")
    else:
        print(f"No changes for {file_path}")

for fp in target_files:
    process_file(fp)
