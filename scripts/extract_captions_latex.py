
import re

def extract_captions(html_path, sample_ids):
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    results = {}

    for sample_id in sample_ids:
        start_marker = '<div class="sample" id="sample-{}">'.format(sample_id)
        start_idx = content.find(start_marker)
        
        if start_idx == -1:
            print(f"Sample {sample_id} not found")
            continue
            
        next_sample_match = re.search(r'<div class="sample" id="sample-', content[start_idx+len(start_marker):])
        if next_sample_match:
            end_idx = start_idx + len(start_marker) + next_sample_match.start()
        else:
            end_idx = len(content)
            
        sample_html = content[start_idx:end_idx]
        
        captions = {}
        
        # Ground Truth
        gt_match = re.search(r'<div class="ground-truth">(.*?)</div>', sample_html, re.DOTALL)
        if gt_match:
            text = gt_match.group(1).replace("Ground truth: ", "").strip()
            text = re.sub(r'<[^>]+>', '', text)
            captions['Ground Truth'] = text

        # Models from table
        rows = re.findall(r'<tr>(.*?)</tr>', sample_html, re.DOTALL)
        for row in rows:
            if '<th>' in row: continue
            
            cols = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
            if len(cols) >= 2:
                # Model Name
                model_name_raw = cols[0]
                model_name = re.sub(r'<[^>]+>', '', model_name_raw).strip()
                
                # Caption
                caption_raw = cols[1]
                caption_text = re.sub(r'<[^>]+>', '', caption_raw).strip()
                
                if model_name:
                    captions[model_name] = caption_text

        results[sample_id] = captions

    return results

def format_latex(captions):
    latex_output = ""
    
    # Priority order for display if available
    priority_order = [
         'Ground Truth', 
         'BLIP-2', 
         'Florence-2', 
         'Idefics3', 
         'BLIP-1-CPU', 
         'Qwen2-7b', 
         'Gemma-9b instruct', 
         'Llama-8b', 
         'Gemma-9b-Quantized', 
         'Llama-9b-Quantized', 
         'SPE+Qwen2-7b v2', 
         'SPE+Gemma-9b v2'
    ]
    
    # Capitalization fix map
    name_map = {
        'Qwen2-7b': 'Qwen2-7B',
        'Llama-8b': 'Llama-8B',
        'SPE+Qwen2-7b v2': 'SPE+Qwen2-7B v2',
        'SPE+Gemma-9b v2': 'SPE+Gemma-9B v2'
    }

    # Extract priority items first
    for key in priority_order:
        if key in captions:
            display_name = name_map.get(key, key)
            value = captions[key]
            # Escape LaTeX
            value = value.replace('%', '\\%').replace('$', '\\$').replace('_', '\\_').replace('#', '\\#').replace('^', '\\^')
            latex_output += f"        \\item[\\textbf{{{display_name}}}:] {value}\n"
            
    # Add any remaining keys not in priority list
    for key in captions:
        if key not in priority_order:
             # Skip empty keys if any
             if not key: continue
             display_name = name_map.get(key, key)
             value = captions[key]
             value = value.replace('%', '\\%').replace('$', '\\$').replace('_', '\\_').replace('#', '\\#').replace('^', '\\^')
             latex_output += f"        \\item[\\textbf{{{display_name}}}:] {value}\n"
        
    return latex_output

target_samples = [4, 11, 13, 16, 17, 24, 25]
data = extract_captions('/Users/emanuelediluzio/Desktop/multimodel_comparison.html', target_samples)

for sample_id in target_samples:
    print(f"--- SAMPLE {sample_id} ---")
    if sample_id in data:
        print(format_latex(data[sample_id]))
    print("-------------------")
