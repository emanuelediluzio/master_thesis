#!/usr/bin/env bash
set -euo pipefail

# Assumes we are inside ai-text-detector repo and virtualenv is activated
OUTDIR="chapter_ai_scores"
mkdir -p "$OUTDIR"

python analyze_chapter.py "Introduction" --out "$OUTDIR/introduction.txt"
python analyze_chapter.py "State of the Art" --out "$OUTDIR/state_of_the_art.txt"
python analyze_chapter.py "System Architecture" --out "$OUTDIR/system_architecture.txt"
python analyze_chapter.py "Fine-tuning Techniques" --out "$OUTDIR/fine_tuning.txt"
python analyze_chapter.py "Multimodal Extensions" --out "$OUTDIR/multimodal_extensions.txt"
python analyze_chapter.py "Experimental Methodology" --out "$OUTDIR/experimental_methodology.txt"
python analyze_chapter.py "Results and Analysis" --out "$OUTDIR/results_and_analysis.txt"
python analyze_chapter.py "Applications and Impact" --out "$OUTDIR/applications_and_impact.txt"
python analyze_chapter.py "Conclusions and Future Developments" --out "$OUTDIR/conclusions.txt"
python analyze_chapter.py "Source Code" --out "$OUTDIR/source_code.txt"

# Appendices and acknowledgements (optional)
python analyze_chapter.py "Acknowledgements" --out "$OUTDIR/acknowledgements.txt"
