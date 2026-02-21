
import re

filename = '/Users/emanuelediluzio/Desktop/università/tesi_magistrale/revised_presentation_script.md'

with open(filename, 'r') as f:
    content = f.read()

# Pattern to capture speaker notes
# Assuming format: **Speaker Notes**:\n"..."
# Or just finding the quoted sections after "**Speaker Notes**"
# The file format from previous turn:
# **Speaker Notes**:
# "Text..."

# Regex to find text inside quotes that follows "Speaker Notes"
# Note: The script has notes in quotes like "Good morning..."
# We can find all text between quotes that appears after "**Speaker Notes**"

notes = []
sections = content.split('**Speaker Notes**:')
for section in sections[1:]: # Skip preamble before first note
    # The note is usually the first quoted block or paragraph
    # Let's just grab the next few lines until a separator like --- or ###
    lines = section.split('\n')
    note_text = ""
    for line in lines:
        if line.strip().startswith('---') or line.strip().startswith('###'):
            break
        note_text += line + " "
    
    # Simple cleanup to remove quotes if they wrap the whole thing
    clean_text = note_text.replace('"', '').strip()
    notes.append(clean_text)

total_words = 0
for note in notes:
    words = note.split()
    total_words += len(words)

print(f"Total words: {total_words}")
print(f"Estimated time (130 wpm - slow/clear): {total_words / 130:.1f} minutes")
print(f"Estimated time (150 wpm - normal/fast): {total_words / 150:.1f} minutes")
