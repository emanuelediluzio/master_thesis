
import re

def inspect_html(html_path, target_samples):
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for sample_id in target_samples:
        print(f"--- SAMPLE {sample_id} DIAGNOSTIC ---")
        start_marker = '<div class="sample" id="sample-{}">'.format(sample_id)
        start_idx = content.find(start_marker)
        
        if start_idx == -1:
            print(f"Sample {sample_id} NOT FOUND")
            continue
            
        next_sample_match = re.search(r'<div class="sample" id="sample-', content[start_idx+len(start_marker):])
        if next_sample_match:
            end_idx = start_idx + len(start_marker) + next_sample_match.start()
        else:
            end_idx = len(content)
            
        sample_html = content[start_idx:end_idx]
        
        rows = re.findall(r'<tr>(.*?)</tr>', sample_html, re.DOTALL)
        found_models = []
        for row in rows:
            if '<th>' in row: continue
            cols = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
            if len(cols) >= 2:
                model_name = re.sub(r'<[^>]+>', '', cols[0]).strip()
                found_models.append(model_name)
        
        print(f"Found {len(found_models)} models: {found_models}")
        print("-------------------------------")

target_samples = [4, 11, 13, 16, 17, 24, 25]
inspect_html('/Users/emanuelediluzio/Desktop/multimodel_comparison.html', target_samples)
