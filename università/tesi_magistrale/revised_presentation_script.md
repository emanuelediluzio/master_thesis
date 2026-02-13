# Master Thesis Presentation Script (FINAL)
## LLM-based SVG Image Captioning with Conceptual Embeddings

> **DELIVERY TIPS:**
> - Go **slow**, don't rush
> - **Vary your tone** of voice—don't be monotone
> - **Emphasize** important words
> - Wait **2 seconds** at the end of every slide
> - **Don't rush**: better slow and clear

---

### Slide 1: Title
**Visual**: 
**LLM-BASED SVG IMAGE CAPTIONING WITH CONCEPTUAL EMBEDDINGS**
Candidate: Emanuele Di Luzio
Supervisor: Prof. Lorenzo Baraldi
Co-supervisor: Dott. Leonardo Zini
Academic Year: 2024-2025
University of Modena and Reggio Emilia · Department of Engineering "Enzo Ferrari"

**Speaker Notes**:
"Good morning everyone. I am Emanuele Di Luzio and today I present my master's thesis: 'LLM-based SVG Image Captioning with Conceptual Embeddings'. [PAUSE 2 sec] In this work, I propose a method to generate textual descriptions of vector images... working directly on the SVG format, without converting it into pixels."

---

### Slide 2: The World is Built on Vectors
**Content**:
- **SVG (Scalable Vector Graphics)**: Vector Format (NOT Raster)
  - **No Fixed Resolution**: Unlike raster images, it has no defined pixel grid.
  - **Ubiquity**: Icons, logos, UI elements, technical diagrams.
  - **Dynamic**: Renders perfectly at any scale.
- **XML-Based Textual Markup**:
  - **The Primitive**: The `<path>` tag is the core building block.
  - **Mathematical Instructions**:
    - `M (x,y)`: Move to coordinates.
    - `L (x,y)`: Draw Line to.
    - `C (x1,y1, x2,y2, x,y)`: Cubic Bezier Curve.

**Speaker Notes**:
"To understand this work, we must first define our object of study. [PAUSE] Why is SVG so important? 
Icons, logos, technical diagrams... they are everywhere. Their main advantage is **Infinite Scalability**.
Since they are defined by mathematical paths, no matter how much you zoom in, they remain sharp. [PAUSE] Physically, they are **XML-based code**. They are defined by instructions like 'Move to', 'Line to', 'Curve to'. It is graphics defined as **Code**."

---

### Slide 3: Traditional VLMs FORCE Rasterization
**Content**:
- ![Traditional VLM Architecture](/Users/emanuelediluzio/.gemini/antigravity/brain/c2ec0373-bdc3-41a4-ace6-f3d20a540ff3/traditional_sota_pipeline_balanced_1770426164779.png)
- **The SOTA Pipeline** (LLaVA, BLIP, Florence-2):
  1. **Input**: They **ignore** the SVG code.
  2. **Forced Rasterization**: The vector is converted into a **Grid of Pixels**.
  3. **Vision Encoder (ViT)**: Extracts visual features from the pixels.
  4. **LLM**: Generates text based on the *image*, not the *code*.
- **Goal**: Automatically generate textual descriptions for vector graphics.
- **References**:
  - [1] **LLaVA**: Liu et al., *Visual Instruction Tuning* (NeurIPS 2023)
  - [2] **BLIP-2**: Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training* (ICML 2023)
  - [3] **Florence-2**: Xiao et al., *Florence-2: Advancing a Unified Representation* (CVPR 2024)

**Speaker Notes**:
"This is critical: How do standard models work? [PAUSE] Models like LLaVA or GPT-4V **cannot** read SVG code. [SLOW] Even if you have the perfect mathematical definition, they ignore it. They **force** a conversion to pixels. [PAUSE] They take the elegant vector structure and smash it into a flat grid of colored dots. Only then do they try to understand it. This is the **Rasterization Bottleneck**."

---

### Slide 4: The Core Challenge: Pixels vs. Primitives
**Content**:
- ![Rasterization Problem Diagram](/Users/emanuelediluzio/.gemini/antigravity/brain/c2ec0373-bdc3-41a4-ace6-f3d20a540ff3/rasterization_problem_diagram_1770378192638.png)
- **The Inefficiency of Rasterization**:
  - 🔴 **Information Loss**: Converting XML to Pixels is irreversible (Structure Lost).
  - 🔴 **Compute Waste**: A simple diagonal line requires processing thousands of white background pixels.
- **The Vector-Native Opportunity**:
  - 🟢 **Semantic**: We treat SVG as Code (XML is a language).
  - � **Continuous**: No resolution artifacts. A circle is a mathematical curve, not a grid of dots.
  - 🟢 **Efficiency**: Dense semantic signal (few tokens) vs Sparse pixel signal (many patches).

**Speaker Notes**:
"Why is this a problem? [PAUSE] Think of an SVG as the **blueprints** of a building. Rasterization is like **throwing away the blueprints** and trying to understand the building by looking at a **blurry photograph**.
[PAUSE] **First**: Irreversible information loss. You lose the structural logic.
**Second**: Computational waste. You analyze thousands of white pixels representing empty wall, instead of just reading the single line 'Wall: 5 meters'.
It is fundamentally the wrong approach for technical data. [PAUSE 2 sec]"

---

### Slide 5: The Solution – SVG Path Embedder (SPE)
**Content**:
- ![SPE Architecture Diagram](/Users/emanuelediluzio/.gemini/antigravity/brain/c2ec0373-bdc3-41a4-ace6-f3d20a540ff3/spe_architecture_strict_style_v4_1770379197892.png)
- **The Paradigm Shift**:
  - We stop treating SVG as an **Image** and treat it as a **Sequence of Paths**.
- **The Mechanism**:
  - **Input**: Raw geometric commands (`M`, `L`, `C`)—the DNA of the shape.
  - **Output**: A compact **256-dim** latent vector (The semantic "embedding").
- **The Key Innovation**:
  - **Hyperspherical Space**: Normalized vectors (Unit Norm) ensure stability.
  - **Noise Injection**: Training with noise forces the model to learn the *concept* of the shape, not just memorize coordinates.
- **Reference**:
  - [4] **SPE**: Zini et al., *A Scalable Vector Graphics Path Auto-Encoder* (Under Review, 2025)

**Speaker Notes**:
"Our approach is the **SPE**—SVG Path Embedder. [PAUSE] We completely shift the paradigm. Instead of rasterizing pixels, we process the **DNA** of the image: the paths themselves. [POINT at diagram] We take the raw geometric commands (`Move`, `Line`, `Curve`) and project them into a dense **256-dimensional vector**. [PAUSE] The secret sauce? We force these vectors onto a **Hypersphere** (Unit Norm) and inject **Gaussian Noise** during training. This prevents the model from just memorizing coordinates; it forces it to learn the robust **concept** of the shape. [PAUSE 2 sec]"

---

### Slide 6: Our Approach – Architecture Overview
**Content**:
- ![Architecture Overview](/Users/emanuelediluzio/.gemini/antigravity/brain/c2ec0373-bdc3-41a4-ace6-f3d20a540ff3/final_architecture_clean_v2_1770381611765.png)
- **The Data Flow**:
  1. **INPUT**: Raw SVG Paths (No Rasterization).
  2. **ENCODER (SPE)**: Transforms Code to Dense Vectors (Frozen). Preserves geometric knowledge.
  3. **BRIDGE (Projection)**: Aligns Geometric Space to Textual Space. Maps 256-dim (SPE) $\to$ 4096-dim (LLM).
  4. **REASONING (LLM)**: Large Language Model generates description. (LoRA Adapters: Efficient adaptation <1% parameters).

**Speaker Notes**:
"This is the complete architecture. [PAUSE] It is a clean, end-to-end pipeline.
**First**, the raw SVG paths enter the model.
**Second**, our Frozen SPE Encoder transforms them into dense semantic vectors.
**Third**, a trainable Projection Layer acts as a bridge, translating 'Geometry' into 'Language'.
**Finally**, the LLM (Qwen2) receives these translated vectors and generates the caption. [PAUSE]
It's a direct path from Code to Meaning."

---

### Slide 7: Training Strategy
**Content**:
- **Why this configuration?**
  - **The Real Advantage**: Processing **SVG Paths** (Dense Code) instead of **Raw Images** (Sparse Pixels).
  - **Efficiency**: Dense semantic signal (few tokens) vs sparse pixel signal (many patches).
  - **Stability**: Keeping SPE **Frozen** prevents "Catastrophic Forgetting" of geometric features.
- **Critical Mechanism**: **Z-Score Normalization**
  - Essential to align the **Statistics** of the Encoder with the **LLM's Expectations**.

**Speaker Notes**:
"Why this specific configuration? [PAUSE] The main advantage is **Data Efficiency**. We process **SVG Paths**—the actual source code—instead of raw pixels. This means we treat the image as information, not just a grid of colors. [PAUSE] We use standard techniques like **LoRA** for efficiency, but that is not the core innovation. The real key here is the **Z-Score Normalization**... without it, the statistical mismatch between the geometry encoder and the language model would cause training to implode."

---

### Slide 8: Dataset Construction
**Content**:
- **Source**: Icons8 (Consistent, high-quality style).
- **Size**: ~90,000 SVG-Caption pairs.
- **Preprocessing**:
  - **Tag Stripping**: Removing XML noise (`<defs>`, metadata).
  - **Canonical ViewBox**: Normalization to 512x512.
  - **Complexity Filter**: Removing paths with > 20 segments.
- **Optimization**: **Dynamic Padding**
  - Batch-adaptive length eliminates wasted compute, boosting training speed by ~40%.

**Speaker Notes**:
"For our setup: we used **Qwen2-7B-Instruct**, trained on 90,000 pairs from Icons8. [PAUSE] We applied rigorous preprocessing: stripping XML noise, normalizing coordinates, and filtering overly complex paths. [PAUSE] We optimized training efficiency with **Dynamic Padding**—grouping samples by length to eliminate 40% of useless padding computation. [PAUSE 2 sec]"

---

### Slide 9: Preliminary Experiment – The "Text-Only" Attempt
**Content**:
- ![Text-Only Architecture (Final Polished)](/Users/emanuelediluzio/.gemini/antigravity/brain/c2ec0373-bdc3-41a4-ace6-f3d20a540ff3/text_only_arch_final_polished_1770556231214.png)
- **The Question**: "Can't we just feed SVG code to an LLM?"
- **Phase 1: Zero-Shot (No Training)**
  - **Input**: Raw SVG Code $\to$ Qwen2-7B.
  - **Result**: Complete Failure. The model treats code as generic text, outputting irrelevant descriptions.
- **Phase 2: Fine-Tuning (LoRA)**
  - **Input**: Trained on 90k SVG-Caption pairs.
  - **Result**: The model learns the **Syntax** (perfect XML tags) but fails the **Semantics**.
  - **The Hallucination Trap**: High CLIPScore (pattern matching) but very low BLEU (gibberish).
- **Conclusion**:
  - LLMs can learn the structure of the code, but not the visual meaning. We **need** a Geometric Encoder (SPE).

**Speaker Notes**:
"Before building the full system, we ran a preliminary experiment to test the limits of text-only processing.
[PAUSE] **Phase 1: Zero-Shot**. We fed raw SVG code to Qwen2. It failed completely. It just saw a wall of numbers.
[PAUSE] **Phase 2: Fine-Tuning with LoRA**. We trained it. The model learned to write perfect XML syntax... but the descriptions were **hallucinations**. It was like a student memorizing equations without understanding math. It produced valid code but meaningless descriptions. [PAUSE] This confirmed that we *need* the SPE to translate code into meaning. [PAUSE 2 sec]"

---

### Slide 10: Experimental Setup
**Content**:
- **Models Comparison**:
  - **Baselines (Raster SOTA)**: **BLIP-2**, **Florence-2**.
  - **Text-Only Baselines (Zero Shot)**: Qwen2 / Gemma / Llama-3.
  - **Our Models**: **SPE + Qwen2** / SPE + Gemma.
- **Evaluation Metrics**:
  - **CLIPScore**: Measures Semantic Alignment (Meaning) between caption and image.
  - **BLEU / ROUGE**: Measure Textual Similarity (Fluency) against ground truth.
  - **Composite Score**: `(CLIPScore / 10) + BLEU + METEOR + ROUGE`.

**Speaker Notes**:
"For experimental setup, we compared our approach against two formidable baselines. [PAUSE] First, the **Raster State-of-the-Art**: BLIP-2 and Florence-2. Second, **Text-Only** LLMs like Qwen2 and Gemma trying to read code directly.
[PAUSE] We evaluated using **CLIPScore** for semantic meaning (Does it describe the right concept?) and **BLEU/ROUGE** for textual fluency. [PAUSE 2 sec]"

---

### Slide 11: Quantitative Results
**Content**:
- ![Quantitative Results Table (Ultra-HD)](/Users/emanuelediluzio/.gemini/antigravity/brain/c2ec0373-bdc3-41a4-ace6-f3d20a540ff3/quantitative_results_table_ultra_hd.png)
- [Original Markdown for Reference]:
- | Method | CLIPScore | BLEU-1 | **Composite** |
  |--------|-----------|--------|---------------|
  | **BLIP-2 / Florence-2** (Raster) | **~32.5** | **~0.25** | **~4.0** |
  | **SPE + Qwen2 (Ours)** | **32.41** | **0.42** | **4.18** |
  
- **What are we comparing?**
  - **Baselines**: They force Rasterization (Pixels).
  - **Ours**: Processes raw geometry directly (Vectors).
- **The Result**:
  - **Competitiveness**: We match/exceed SOTA performance without rendering pixels.
  - **Superior Fluency** (BLEU +75%): Geometric grounding produces cleaner language.
  - **Efficiency**: Dense vector input vs Sparse pixel processing.

**Speaker Notes**:
"Here is the critical result. [PAUSE] We compared our approach against **State-of-the-Art Raster VLMs** like **BLIP-2** and **Florence-2**.
[POINT at table] Our results show that we have successfully **closed the gap**. We achieve performance **comparable** to these massive models, and even exceed them in fluency (BLEU +75%).
This proves that a **Native Vector Approach** is feasible: we can reach SOTA levels by leveraging the structural 'DNA' of the image, without needing to process pixels at all. [PAUSE 2 sec]"

---

### Slide 12: Ablation Studies
**Content**:
- **Z-Score Normalization**: Critical for training stability.
  - **Problem**: SPE embeddings have different statistics ($\mu$, $\sigma$) than what the LLM expects.
  - **Solution**: Normalize: $z = (x - \mu) / \sigma$ $\to$ Aligns to $\mu \approx 0, \text{var} \approx 1$.
  - **Result**: Without it: Training diverges (gradient explosion).
- **LoRA Rank**:
  - We found that a LoRA rank of 16 provides the perfect balance of efficiency and expressivity.

**Speaker Notes**:
"What makes all this possible? A critical component is **normalization**. [PAUSE] We conducted an ablation study. Without Z-Score Normalization, training **diverges**. [SLOW] Why? There is a **Distribution Mismatch**. The SPE embeddings have different statistics than the LLM's expected inputs (Unit Gaussian). [PAUSE] By normalizing relevant dimensions, we **align the manifolds**, enabling stable convergence. It is a mathematical prerequisite for the projection to work. [PAUSE 2 sec]"

---

### Slide 13: Qualitative Analysis
**Content**:
- [4 Distinct Examples Side-by-Side]
- ![Qualitative Analysis Examples](/Users/emanuelediluzio/.gemini/antigravity/brain/c2ec0373-bdc3-41a4-ace6-f3d20a540ff3/qualitative_analysis_4_examples_1770551335256.png)
- **Examples**:
  - **1. 🎓 Graduate Emoji** (`sr_122237.svg`): Baseline sees "Hot cross bun" (Hallucination) vs Ours "Student with academic cap" (Correct).
  - **2. ❌ Yellow/Black Cross** (`wd_955382.svg`): Baseline sees "Flower" or "Letter M" (Confusion) vs SPE+Gemma "Large letter X" (Geometric understanding).
  - **3. ☎️ Vintage Telephone** (`ki_0196804.svg`): Baseline sees "Blue blob" or "Generic device" vs Ours "Classic rotary telephone" (Fine-grained).
  - **4. ☀️ Sun Icon** (`ssl_231992.svg`): Baseline sees "Star" or "Yellow flower" vs Ours "Sun with radiating rays" (Geometric understanding).
- **Limitations (OCR Blindness)**:
  - If you feed it a vector drawing of a **STOP sign**, the model sees a red octagon.
  - It describes geometry perfectly ("Red octagon with white internal shapes") but fails to read the word "STOP".
  - **Reason**: Text is rendered as path curves, losing semantic character info.

**Speaker Notes**:
"Now, let's see what this means in practice. [PAUSE] Look at the Graduate Emoji [POINT]. The baseline calls it a 'Hot cross bun'. It's hallucinating. Our model correctly identifies it.
[PAUSE] **However, limitations exist.** A prime example is 'OCR Blindness'. If you show it a vector STOP sign, it sees the red octagon perfectly... but it cannot read the word 'STOP'. Why? Because to the SPE, those letters are just curves, not text. It understands the *shape*, but misses the *symbol*. [PAUSE 2 sec]"

---

### Slide 14: Future Work
**Content**:
- **Hierarchical Encoding**:
  - Using GNNs or Tree-Transformers to preserve DOM structure.
  - Current: SVG linearized as flat sequence.
- **Hybrid Dual-Stream**:
  - Merging Vector (structure) + Raster (texture).
- **Text-to-SVG**:
  - Reversing the pipeline (Generation).

**Speaker Notes**:
"For the future, we envision three paths. [PAUSE] Hierarchical encoding to respect the SVG tree structure. A Hybrid Dual-Stream approach to combine vector precision with raster texture details. And finally... reversing the pipeline to generate SVG code from text. [PAUSE 2 sec]"

---

### Slide 15: Conclusion & Future Work
**Visual:** Summary Bullet Points.
1. **Recap**: A Paradigm Shift: From Pixels to Vectors.
2. **Result**: Native Geometric Understanding.
3. **Future**: Hierarchical Encoding (for CAD/Blueprints).

**Thesis in a Nutshell**:
*"We proposed a vector-native architecture that enables LLMs to understand technical drawings by processing their geometric code directly, achieving performance comparable to state-of-the-art raster models **while preserving infinite structural fidelity**."*
**Speaker Notes:**
"In conclusion, I would like to summarize the journey of this research.
**This thesis proposes a fundamental change in how we handle technical imagery.**
We moved away from the standard practice of treating technical drawings as simple pictures made of pixels.
Instead, we built a system that respects the source material: code.
We successfully connected the mathematical precision of SVG paths directly with the semantic power of Large Language Models.
Essentially, we taught an AI to 'read' the geometry of a drawing—line by line, curve by curve—treating it as a language rather than an image.
This approach proved that skipping the pixel grid allows us to retain the exact structure of the design, leading to captions that are not just fluent, but geometrically accurate.
For the future, we plan to scale this to **Hierarchical Encoding**, allowing us to process massive CAD blueprints by understanding groups and layers."
Thank you for your attention. I am open to any questions."

---

### Slide 16: Thank You
**Visual**: 
**Thank You**
[Email / Contact Info]

**Speaker Notes**:
"Thank you very much for your attention. [PAUSE 2 sec] I am available for any questions."
