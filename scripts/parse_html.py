from bs4 import BeautifulSoup
import re

HTML_FILE = "cherry30.html"
OUTPUT_TEX = "gallery.tex"

def parse_and_generate():
    with open(HTML_FILE, "r") as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    samples = soup.find_all('div', class_='sample-container')
    
    latex_output = "\\subsection{Qualitative Gallery}\nThe following pages present a selection of generated captions compared to the ground truth.\n\n"
    
    # We only want samples 0 to 15 (5.3 to 5.18)
    # The user said "remove > 5.19", which is Sample 16+.
    
    for i in range(16):
        # Find sample container for Sample i
        # The HTML sample headers are "Sample 0", "Sample 1" etc.
        # We need to find the correct div.
        
        target_sample = None
        for s in samples:
            header = s.find('h2')
            if header and header.text.strip() == f"Sample {i}":
                target_sample = s
                break
        
        if not target_sample:
            print(f"Warning: Sample {i} not found in HTML.")
            continue
            
        # Extract Filename/Class hint?
        # Filename is in <div class='filename'>
        filename_div = target_sample.find('div', class_='filename')
        filename = filename_div.text.strip() if filename_div else "Unknown"
        
        # Ground Truth
        gt_div = target_sample.find('div', class_='caption-text') # The first one after GT label?
        # Structure: <div class='captions'><div class='ground-truth'>Ground truth:</div><div class='caption-text'>THE TEXT</div>
        captions_div = target_sample.find('div', class_='captions')
        gt_text = ""
        if captions_div:
            # The GT text is the first caption-text sibling of ground-truth?
            # Actually finding 'caption-text' inside captions_div, assuming first is GT.
            gt_candidates = captions_div.find_all('div', class_='caption-text')
            if gt_candidates:
                gt_text = gt_candidates[0].text.strip()
        
        # Generated Caption for SPE+Qwen2-7b v2
        # It's in a table.
        gen_text = "N/A"
        rows = target_sample.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                model_name = cols[0].text.strip()
                if "SPE+Qwen2-7b" in model_name: # Matches v2 or other variants
                    gen_text = cols[1].text.strip()
                    break
        
        # Escape LaTeX special chars
        def escape_tex(text):
            chars = {
                '&': '\\&',
                '%': '\\%',
                '$': '\\$',
                '#': '\\#',
                '_': '\\_',
                '{': '\\{',
                '}': '\\}',
                '~': '\\textasciitilde{}',
                '^': '\\textasciicircum{}'
            }
            return "".join(chars.get(c, c) for c in text)
        
        gt_text = escape_tex(gt_text)
        gen_text = escape_tex(gen_text)
        
        # Generate Figure LaTeX
        # Use filename as short caption title or Generic?
        # Existing gallery.tex used "Sample X: Class".
        # We don't have Class readily available unless we map filenames.
        # But user just wants the text.
        # I'll use "Sample X".
        
        latex_output += "\\clearpage\n"
        latex_output += "\\begin{figure}[p]\n"
        latex_output += "    \\centering\n"
        latex_output += f"    \\includegraphics[width=0.8\\textwidth]{{figures/sample_{i}.png}}\n"
        latex_output += f"    \\caption[Sample {i}]" + "{\\textbf{Sample " + str(i) + "}. \\textbf{Ground Truth}: " + gt_text + " \\textbf{Generated}: " + gen_text + "}\n"
        latex_output += f"    \\label{{fig:gallery_{i}}}\n"
        latex_output += "\\end{figure}\n"
        
    with open(OUTPUT_TEX, "w") as f:
        f.write(latex_output)
    print(f"Generated {OUTPUT_TEX} with 16 samples.")

if __name__ == "__main__":
    # Install bs4 if needed? No, standard environment usually has it or I can't install.
    # If bs4 missing, I'll use regex.
    try:
        parse_and_generate()
    except ImportError:
        print("BS4 not found, using regex fallback (not implemented yet).")
