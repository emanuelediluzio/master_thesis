
from PIL import Image, ImageDraw, ImageFont
import os

# Code content to render
code_lines = [
    '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">',
    '  <!-- Hierarchy: Grouping related elements -->',
    '  <g id="car_chassis">',
    '    <!-- Primitive: Path with Move and Line commands -->',
    '    <path d="M 10 50 L 90 50 L 90 70 Z" fill="blue"/>',
    '  </g>',
    '',
    '  <g id="wheels">',
    '    <!-- Semantic Grouping of wheels -->',
    '    <path d="M 20 70 C 20 80 30 80 30 70" fill="black"/>',
    '    <path d="M 70 70 C 70 80 80 80 80 70" fill="black"/>',
    '  </g>',
    '</svg>'
]

# Config
font_size = 24
line_height = 36
padding = 40
bg_color = (30, 30, 30) # Dark gray (VS Code-ish)
text_color = (212, 212, 212) # Default text
keyword_color = (86, 156, 214) # Blue (tags)
attr_color = (156, 220, 254) # Light Blue (attributes)
string_color = (206, 145, 120) # Orange (strings)
comment_color = (106, 153, 85) # Green (comments)

try:
    # Try to load a monospace font
    # MacOS paths
    font_path = "/System/Library/Fonts/Monaco.ttf"
    if not os.path.exists(font_path):
        font_path = "/System/Library/Fonts/Menlo.ttc"
    if not os.path.exists(font_path):
        font_path = "/Library/Fonts/Arial.ttf" # Fallback, not monospace but works
        
    font = ImageFont.truetype(font_path, font_size)
except:
    font = ImageFont.load_default()

# Canvas size
width = 900
height = padding * 2 + len(code_lines) * line_height
img = Image.new('RGB', (width, height), bg_color)
draw = ImageDraw.Draw(img)

# Window controls (fake macos buttons)
draw.ellipse((20, 20, 32, 32), fill=(255, 95, 86)) # Red
draw.ellipse((40, 20, 52, 32), fill=(255, 189, 46)) # Yellow
draw.ellipse((60, 20, 72, 32), fill=(39, 201, 63)) # Green

# Render Loop
y = padding + 20
for line in code_lines:
    x = padding
    
    # Simple syntax highlighting logic (heuristic)
    if line.strip().startswith('<!--'):
        draw.text((x, y), line, font=font, fill=comment_color)
    else:
        # Check for tags
        parts = line.split(' ')
        cursor_x = x
        for i, part in enumerate(parts):
            color = text_color
            
            # Simple heuristics
            if part.startswith('<') or part.startswith('</'):
                color = keyword_color # Tag
            elif '=' in part:
                 # split attr="value"
                 if '="' in part:
                    attr, val = part.split('="', 1)
                    val = '"' + val
                    
                    draw.text((cursor_x, y), attr, font=font, fill=attr_color)
                    w = draw.textlength(attr, font=font)
                    cursor_x += w
                    
                    draw.text((cursor_x, y), '=', font=font, fill=text_color)
                    w = draw.textlength('=', font=font)
                    cursor_x += w
                    
                    draw.text((cursor_x, y), val, font=font, fill=string_color)
                    w = draw.textlength(val, font=font)
                    cursor_x += w
                    
                    # Add space
                    if i < len(parts) - 1:
                        draw.text((cursor_x, y), ' ', font=font, fill=text_color)
                        cursor_x += draw.textlength(' ', font=font)
                    continue

            draw.text((cursor_x, y), part, font=font, fill=color)
            w = draw.textlength(part, font=font)
            cursor_x += w
            
            # Add space
            if i < len(parts) - 1:
                draw.text((cursor_x, y), ' ', font=font, fill=text_color)
                cursor_x += draw.textlength(' ', font=font)
            
    y += line_height

# Save
output_path = '/Users/emanuelediluzio/Desktop/università/tesi_magistrale/svg_code_slide4.png'
img.save(output_path)
print(f"Image saved to {output_path}")
