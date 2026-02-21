import re

HTML_PATH = "/Users/emanuelediluzio/Desktop/multimodel_comparison.html"

def audit_samples():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # The file structure seems to contain blocks. 
    # I'll look for "File: " as a trusted anchor for each sample.
    # Then I'll look ahead for "Ground truth:"
    
    # Regex to find all occurrences of File and Ground Truth
    # This is rough but effective for indexing.
    # We want to capture the filename and the immediate ground truth text.
    
    # We split by "File: " to get chunks
    chunks = content.split("File: ")
    
    print(f"Found {len(chunks)-1} file markers.")
    
    for i, chunk in enumerate(chunks[1:]): # Skip pre-first-file chunk
        # chunk starts with "filename.svg</div>"
        filename_end = chunk.find("</div>")
        filename = chunk[:filename_end].strip()
        
        # Search full chunk for "speech bubble" and "credit card"
        chunk_lower = chunk.lower()
        if "chat" in chunk_lower or "message" in chunk_lower:
             print(f"MATCH CHAT/MESSAGE: Index {i} | File: {filename}")
        if "speech bubble" in chunk_lower:
             print(f"MATCH SPEECH BUBBLE: Index {i} | File: {filename}")
        if "credit card" in chunk_lower:
             print(f"MATCH CREDIT CARD: Index {i} | File: {filename}")
             
        # Find Ground truth
        gt_start = chunk.find("Ground truth: ")
        if gt_start != -1:
            gt_end = chunk.find("</div>", gt_start)
            gt_text = chunk[gt_start:gt_end].replace("Ground truth: ", "").strip()
            # truncate for display
            gt_preview = gt_text[:100].replace("\n", " ")
            if "speech" in gt_text.lower():
                print(f"MATCH SPEECH: {filename} | {gt_preview}")
            if "credit" in gt_text.lower():
                print(f"MATCH CREDIT: {filename} | {gt_preview}")

            
        print(f"Index {i} | File: {filename} | GT: {gt_preview}...")

audit_samples()
