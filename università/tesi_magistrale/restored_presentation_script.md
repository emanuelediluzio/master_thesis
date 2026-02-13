# Master Thesis Presentation Script
## SVG CAPTIONING WITH TRANSFORMER MODELS: ADVANCED FINE-TUNING TECHNIQUES

---

### Slide 1: Title
**Visual**: 
**SVG CAPTIONING WITH TRANSFORMER MODELS: ADVANCED FINE-TUNING TECHNIQUES**
Candidate: Emanuele Di Luzio
Supervisor: Prof. Lorenzo Baraldi
Co-supervisor: Dott. Leonardo Zini
Academic Year: 2024-2025
University of Modena and Reggio Emilia · Department of Engineering "Enzo Ferrari"

**Speaker Notes**:
"Good morning everyone. I am Emanuele Di Luzio. Today I present my Master’s Thesis titled 'SVG Captioning with Transformer Models: Advanced Fine-tuning Techniques'. This work was conducted under the supervision of Professor Lorenzo Baraldi and Doctor Leonardo Zini at the University of Modena and Reggio Emilia."

---

### Slide 2: Agenda
**Content**:
- 01 Introduction: The Importance of Vector Graphics.
- 02 Problem Statement: Why Current Methods Fail.
- 03 Methodology I: The Decoder-Only Backbone.
- 04 Methodology II: The SPE + Decoder Architecture.
- 05 Dataset: The 90k Stratified Benchmark.
- 06 Results: Quantitative and Qualitative Analysis.
- 07 Conclusions: Future Directions.

**Speaker Notes**:
"The presentation is structured as follows. I will begin by discussing the importance of vector graphics and why current methods fail to process them effectively. I will then introduce my methodology, divided into two parts: the Decoder-Only Backbone and the SPE + Decoder Architecture. Following that, I will detail the construction of the 90k Stratified Benchmark dataset. Finally, I will present the quantitative and qualitative results, concluding with future research directions."

---

### Slide 3: The World is Built on Vectors
**Content**:
- **Key Points**:
    - **Ubiquity**: Icons, Logos, UI elements, Technical Diagrams.
    - **Advantage**: Infinite Scalability (Resolution-Independent).
    - **Goal**: Automated Semantic Understanding of Vectors.

**Speaker Notes**:
"We live in a world built on vectors. From icons and logos to UI elements and technical diagrams, they are ubiquitous. Their primary advantage is infinite scalability—they are resolution-independent. My goal is to enable the automated semantic understanding of these vectors, allowing machines to interpret them as meaningfully as we do."

---

### Slide 4: Brief Interlude: What is an SVG?
**Content**:
- **Key Points**:
    - **Structure**: SVG is XML-based textual markup.
    - **The Primitive**: The `<path>` tag is the core building block.
        - `M (x,y)`: Move to coordinates.
        - `L (x,y)`: Draw Line to.
        - `C (x1,y1, x2,y2, x,y)`: Cubic Bezier Curve.
    - **Graphics as Code**: It is not a grid of pixels; it is a sequence of mathematical instructions.

**Speaker Notes**:
"First, a brief interlude: What is an SVG? It is fundamentally XML-based textual markup. The core building block is the path tag, which uses commands like Move ($M$), Line ($L$), and Cubic Bezier ($C$) to define shapes. Ultimately, SVG is graphics as code—not a grid of pixels, but a sequence of precise mathematical instructions."

---

### Slide 5: The Core Challenge: Pixels vs. Primitives
**Content**:
- **The Inefficiency of Rasterization**:
    - **Information Loss**: Converting XML to Pixels is irreversible ($\text{SVG} \to \text{Image}$).
    - **Structure**: Grouping (`<g>`) and hierarchy are destroyed.
    - **Compute Waste**: A simple diagonal line requires processing thousands of white background pixels.
- **The Vector-Native Opportunity**:
    - **Semantic**: We treat SVG as Code (XML is a language).
    - **Continuous**: No resolution artifacts. A circle is a mathematical curve, not a grid of dots.
    - **Efficiency**: Dense semantic signal (few tokens) vs sparse pixel signal (many patches).

**Speaker Notes**:
"The core challenge is the dichotomy between Pixels and Primitives. Rasterization is inefficient: converting SVG to an image causes irreversible information loss, destroying grouping and hierarchy, and wastes compute on background pixels. The Vector-Native opportunity is to treat SVG as code. This approach is semantic, continuous (free of resolution artifacts), and efficient, offering a dense signal with fewer tokens compared to sparse pixel patches."

---

### Slide 6: Methodology I: The Decoder-Only Backbone
**Content**:
- **Why Decoder-Only?**:
    - **Causal Self-Attention**: Trains the model to predict the next token based on context.
    - **Generative Power**: Superior at producing fluent, coherent natural language compared to Encoder-Decoders.
- **Model Selection Strategy**:
    - **Qwen2-7B**: Chosen for strictly better Reasoning & Coding capabilities (crucial for structured XML data).
    - **Gemma-9B**: Google's efficient open model, strong on general knowledge.
    - **Llama-3-8B**: Meta's industry standard, used to benchmark generalization.

**Speaker Notes**:
"Methodology Part 1 focuses on the Decoder-Only Backbone. I selected this architecture for its Causal Self-Attention and superior generative power in producing fluent language. I chose Qwen2-7B for its strong reasoning and coding capabilities, essential for XML; Gemma-9B for its efficiency; and Llama-3-8B as a standard benchmark for generalization."

---

### Slide 7: Methodology I: The Decoder-Only Backbone
**Content**:
- **The Linguistic Prior**:
    - The model already knows what a "circle" or "arrow" is conceptually.
    - Our goal is to align visual features to these pre-existing linguistic concepts.
- **Input Paradigm**:
    - **Processing**: The Decoder receives the SVG (whether as Text Tokens or Visual Embeddings) exactly like a sentence, processing drawing commands (M, L, C) sequentially from start to finish.
    - We don't feed the 'raw' XML directly. We first preprocess it by converting all absolute coordinates to relative ones. This reduces numerical variance and helps the model understand local geometry better.

**Speaker Notes**:
"We leverage the 'Linguistic Prior': the model conceptually knows what a 'circle' or 'arrow' is. Our goal is to align visual features to these concepts. In our input paradigm, the decoder processes the SVG like a sentence, reading commands sequentially. We don't use raw XML; instead, we preprocess coordinates from absolute to relative to reduce variance and improve local geometric understanding."

---

### Slide 8: Theoretical Foundation: Low-Rank Adaptation (LoRA)
**Content**:
- **The Constraint**: Full Fine-tuning of 7B+ parameters is computationally prohibitive.
- **The Hypothesis**: Weight updates have a Low Intrinsic Rank.
- **The Method**:
    - Freeze the pretrained weights $W_0$.
    - Inject trainable rank decomposition matrices: $\Delta W = B \times A$.
    - Where $r \ll d$ (e.g., $rank=16$).
- **Benefit**: We train $<0.1\%$ of parameters ($A$ and $B$) while keeping the model's general knowledge intact.

**Speaker Notes**:
"The theoretical foundation is Low-Rank Adaptation (LoRA). Since full fine-tuning is computationally prohibitive, we use the hypothesis that weight updates have a low intrinsic rank. We freeze the pre-trained weights and inject small trainable matrices. This allows us to train less than 0.1% of the parameters, preserving the model's general knowledge."

---

### Slide 9: Methodology I: Text-Only Fine-tuning (LoRA)
**Content**:
- **Approach**: Treat SVG code purely as text (like Python/XML).
- **The Technique**: Low-Rank Adaptation (LoRA).
    - Instead of full retraining, we inject small matrices (A times B) into the LLM.
    - Allows efficient adaptation to "SVG Language" with <1% params.
- **Result**: Better than Zero-Shot, but suffers from "Geometric Hallucination" (blind to spatial relations).

**Speaker Notes**:
"In the Text-Only Fine-tuning approach, we treat SVG code purely as text. Using LoRA, we efficiently adapt the model to 'SVG Language' with minimal parameters. The result is better than Zero-Shot, but it still suffers from 'Geometric Hallucination'—the model sees the code but remains blind to the actual spatial relationships."

---

### Slide 10: Methodology II: The SPE + Decoder Architecture
**Content**:
- **The Missing Piece**: Explicit Geometric Understanding.
- **Our Architecture**:
    - **Input**: Visual Embeddings from SPE (Frozen).
    - **Bridge**: Projection Layer (The "Middle Layer") connects SPE $\to$ LLM.
    - **Backbone**: The LoRA-adapted LLM from the previous step.
- **Mechanism**:
    - The LLM now attends to both Visual Tokens (Geometry) and Text Tokens (Caption).
    - SPE isn't just a random encoder; it is a pre-trained Auto-Encoder that has already learned to reconstruct SVG paths. We insert a Projection Layer to align its dense, pre-learned features with the LLM's embedding space. Now, the model doesn't just read code; it 'sees' the geometry through an expert eye.

**Speaker Notes**:
"The missing piece was explicit geometric understanding, which we address with the SPE + Decoder Architecture. We use frozen Visual Embeddings from the SPE and bridge them to the LLM via a Projection Layer. The LLM now attends to both geometry and text. SPE acts as an expert, pre-trained auto-encoder, allowing the model to truly 'see' the geometry rather than just reading code."

---

### Slide 11: Dataset Construction
**Content**:
- **Source**: Icons8 (Consistent, high-quality style).
- **Size**: ~90,000 SVG-Caption pairs.
- **Preprocessing**:
    - **Tag Stripping**: Removing XML noise (`<defs>`, metadata).
    - **Canonical ViewBox**: Normalization to 512x512.
    - **Complexity Filter**: Removing paths with > 50 segments.

**Speaker Notes**:
"For data, we constructed the 90k Stratified Benchmark using Icons8 for its consistent style. Preprocessing was strict: we stripped XML noise, normalized the ViewBox to 512x512, and filtered out overly complex paths to ensure the model focuses on clean semantic signals."

---

### Slide 12: Implementation Details
**Content**:
- **Key Points**:
    - **Stratified Split**: Test set (1000 samples) balanced across categories (Nature, UI, Arrows).
    - **Dynamic Padding**:
        - Pad to `batch_max_len`, NOT global max.
        - Why it matters: Transformer Attention is Quadratic ($O(N^2)$).
        - Result: 40% Training Speedup by avoiding computation on empty padding tokens.
    - **Input Schema**:
        - `{"input": "<svg>...", "output": "A red arrow pointing..."}`

**Speaker Notes**:
"Implementation included a stratified split of 1000 samples balanced across categories. A key optimization was 'Dynamic Padding', where we pad only to the batch maximum. This reduced training time by 40% by avoiding wasted computation on empty tokens in the Transformer's quadratic attention mechanism."

---

### Slide 13: Experimental Setup
**Content**:
- **Baselines**:
    1. **Raster Models (Zero-Shot)**: BLIP-2, Florence-2 (Vision-Language Models).
    2. **Text-Only (LoRA)**: Qwen2 / Llama-3 trained on raw SVG code (No SPE).
- **Models**:
    1. **SPE + Qwen2-7B (LoRA)** vs SPE + Gemma-9B.
    2. **Qwen2-7B (LoRA)** vs Gemma-9B.
- **Metrics**:
    - **CLIPScore**: Visual Semantic Alignment.
    - **BLEU/ROUGE**: Textual Fluency.
    - **Composite**: Weighted Mean ($\frac{CLIP}{10} + BLEU + METEOR + ROUGE$).

**Speaker Notes**:
"We compared our models against two baselines: Zero-Shot Raster models like BLIP-2 and Text-Only LoRA models. The specific comparison was between SPE-augmented Qwen2/Gemma and their text-only counterparts. Metrics included CLIPScore for visual alignment, BLEU/ROUGE for fluency, and a weighted Composite mean."

---

### Slide 14: Quantitative Results
**Content**:
- **Text-Only Baseline**: High raw CLIPScore (32.30) but hallucinations.
- **SPE + Qwen2**:
    - **Best Composite Score**: 4.18.
    - **Superior Fluency**: BLEU-1 (0.42) vs Baseline (0.24).
- **Insight**: Structured embeddings act as a regularizer, ensuring valid application of language.
- "Our results were revealing. While the naive Text-Only baseline scored high on raw feature matching, it often hallucinated. The SPE + Qwen2 model achieved the best overall Composite Score and significantly higher fluency. The structured embedding helps the model 'ground' its language in reality."

**Speaker Notes**:
"Quantitative results showed that while the Text-Only baseline had a high raw CLIPScore, it suffered from hallucinations. SPE + Qwen2 achieved the best Composite Score of 4.18 and superior fluency. The structured embeddings act as a regularizer, effectively grounding the model's language in geometric reality."

---

### Slide 15: Qualitative Analysis
**Content**:
- **Case Study 1: Icon ID 38 (Graduate Emoji)**
    - **Our Model**: "Stylized human figure... orange shape". \textcolor{green}{\Checkmark (Detects Human)}
    - **Baseline**: "Hot cross bun". \textcolor{red}{\XSolidBrush (Hallucination)}
- **Case Study 2: Icon ID 12 (Cross/X)**
    - **Our Model**: "Geometric figure representing an 'X'". \textcolor{green}{\Checkmark (Precise Geometry)}

**Speaker Notes**:
"Qualitatively, the model proves its worth. In Case Study 1 (Graduate Emoji), our model correctly identifies a 'stylized human figure', whereas the baseline hallucinates a 'hot cross bun'. In Case Study 2, our model precisely recognizes the geometry of a cross, confirming its understanding of shape."

---

### Slide 16: Failure Modes
**Content**:
- **Key Points**:
    - **OCR Blindness**: Cannot read text rendered as paths (e.g., "STOP" sign seen as red octagon). The model sees the geometry (octagon) but misses the symbol (letters).
    - **Style Agnostic**: Ignores specific fill colors/textures (SPE limitation).
    - **Hallucination**: Over-interpreting abstract shapes.
- "However, limitations exist. A prime example is 'OCR Blindness'. If you feed it a vector drawing of a STOP sign, the model sees a red octagon. It describes the geometry perfectly but fails to read the word 'STOP' because the text is rendered as path curves, not characters. It sees the shape, but misses the semantic symbol."

**Speaker Notes**:
"There are limitations. The most prominent is 'OCR Blindness'. For a STOP sign, the model sees the 'red octagon' geometry perfectly but misses the text 'STOP' because it's rendered as paths, not characters. It's also style-agnostic, ignoring textures, and can sometimes over-interpret abstract shapes."

---

### Slide 17: Ablation Studies
**Content**:
- **Key Points**:
    - **Z-Score Normalization**: Critical. Without matching SPE stats to LLM stats, training diverges.
    - **LoRA Rank**: $r=16$ is optimal. Larger ranks yield diminishing returns.
    - **Prompting**: Simple "Describe this icon" works best.
- "Our ablation studies showed that Z-Score Normalization is critical; without aligning the statistical distributions of the visual encoder and the LLM, the model fails to learn. We also found that a LoRA rank of 16 provides the perfect balance of efficiency and expressivity."

**Speaker Notes**:
"Ablation studies reinforced two things: Z-Score Normalization is critical to prevent training divergence by aligning visual and language statistics. Additionally, a LoRA rank of 16 was found to be optimal, offering the perfect balance of efficiency and expressivity."

---

### Slide 18: Future Work: Hierarchical Encoding
**Content**:
- **Key Points**:
    - **Limitation**: Current model linearizes the SVG, treating it as a flat sequence (like text).
    - **The Solution**: Graph Neural Networks (GNNs) or Tree-Transformers.
    - **Preserve the DOM Tree Structure** (Parent $\to$ Child relations).
    - **Example**: Recognize that a `<circle>` inside a `<g id="button">` is part of a UI element, not just a shape.
    - **Goal**: True "Scene Graph" understanding of vector graphics.

**Speaker Notes**:
"Future work involves addressing the linearization of SVGs. We currently treat SVG as a flat sequence. Using Graph Neural Networks (GNNs) would allow us to preserve the DOM tree structure, enabling the model to understand hierarchical relationships—like seeing a circle inside a 'button' group as a UI element, achieving true 'Scene Graph' understanding."

---

### Slide 19: Future Work: Hierarchical Encoding (Duplicate Title in PDF)
**Content**:
- **Key Points**:
    - **Limitation**: Vectors struggle with textures, gradients, and embedded bitmaps.
    - **The Solution**: Dual-Stream Encoder.
    - **Stream A (Vector)**: SPE process sparse geometry (Precision).
    - **Stream B (Raster)**: CNN/ViT process rendered pixels (Texture/Color).
    - **Fusion**: Late fusion mechanisms (Cross-Attention) to combine both signals.
    - **Outcome**: The semantic precision of code + the visual richness of pixels.

**Speaker Notes**:
"*(Note: Slide title in PDF overlaps with previous slide, but content differs).*
Another key area is handling textures and gradients, where vectors struggle. We propose a Dual-Stream Encoder: one stream for sparse geometry using SPE, and another for raster pixels using CNNs. Fusing these allows us to combine the semantic precision of vector code with the visual richness of pixels."

---

### Slide 20: The Broader Vision: End-to-End Vector Systems
**Content**:
- **Beyond Captioning**:
    - **Vector-Aware Editing**: "Make the circle red" (Natural Language $\to$ SVG Command).
    - **Style Transfer**: Apply "Sketch Style" to a flat icon.
- **The Holy Grail: Bidirectional Generation**:
    - Use the captioned data to train Text-to-SVG generators.
    - Create a closed-loop system for vector design.

**Speaker Notes**:
"Finally, our broader vision is End-to-End Vector Systems. Beyond captioning, we envision Vector-Aware Editing—using natural language to modify geometry—and the 'Holy Grail': Bidirectional Generation. We can use our captioned data to train Text-to-SVG generators, creating a complete closed loop for vector design."

---

### Slide 21: Conclusions
**Content**:
- **Key Points**:
    - **Thesis Statement**: LLMs can learn to see structure directly, without rasterization.
    - **Main Achievement**: Composite Score 4.18.
        - Outperformed Text-Only (+0.23) and Raster Baselines (+0.84) in quality.
    - **Efficiency**: Validated LoRA for multimodal adaptation (<1% params).
    - **Philosophy**: Moving from "Pixels as Opaque Strings" to "Graphics as Symbolic Code".

**Speaker Notes**:
"To conclude, this work proves that LLMs can learn to see structure directly. We achieved a Composite Score of 4.18, outperforming both text-only and raster baselines. We validated the efficiency of LoRA for multimodal adaptation and advanced the philosophy of treating graphics not as opaque pixels, but as meaningful, symbolic code."

---

### Slide 22: Thank You
**Visual**: "Thank You for your time and attention"

**Speaker Notes**:
"Thank you very much for your time and attention. I am now happy to take any questions."
