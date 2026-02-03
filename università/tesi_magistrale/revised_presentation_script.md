# Master Thesis Presentation Script (Revised)
## LLM-based SVG image captioning with conceptual embeddings

---

### Slide 1: Title
**Visual**: 
**LLM-BASED SVG IMAGE CAPTIONING WITH CONCEPTUAL EMBEDDINGS**
**(Prev: SVG Captioning with Transformer Models: Advanced Fine-tuning Techniques)**
Candidate: Emanuele Di Luzio
Supervisor: Prof. Lorenzo Baraldi
Co-supervisor: Dott. Leonardo Zini
Academic Year: 2024-2025
University of Modena and Reggio Emilia · Department of Engineering "Enzo Ferrari"

**Speaker Notes**:
"Good morning everyone. I am Emanuele Di Luzio. Today I present my Master’s Thesis titled 'LLM-based SVG image captioning with conceptual embeddings'

---

### Slide 2: Agenda
**Content**:
- 01 Introduction: Importance & Task Definition.
- 02 Problem Statement: Pixels vs. Primitives.
- 03 Methodology I: The Decoder-Only Backbone.
- 04 Methodology II: The SPE + Decoder Architecture.
- 05 Implementation: Dataset & Training Strategies.
- 06 Results: Quantitative & Qualitative Analysis.
- 07 Conclusions: Future Directions.

**Speaker Notes**:
"I will begin by discussing the importance of vector graphics and why current methods fail to process them effectively. I will then introduce my methodology, divided into two parts: the Decoder-Only Backbone and the SPE + Decoder Architecture. Following that, I will detail the construction of the 90k Stratified Benchmark dataset. Finally, I will present the quantitative and qualitative results, concluding with future research directions."

---

### Slide 3: The World is Built on Vectors
**Content**:
- **Key Points**:
    - **Ubiquity**: Icons, Logos, UI elements, Technical Diagrams.
    - **Advantage**: Infinite Scalability (Resolution-Independent).
    - **Goal**: Automated Semantic Understanding of Vectors.

**Speaker Notes**:
"We live in a world built on vectors. From icons and logos to UI elements and technical diagrams, they are ubiquitous. Their primary advantage is infinite scalability—they are resolution-independent. My goal is to enable the automated semantic understanding of these vectors, allowing machines to interpret them."

---

### Slide 4: Brief Interlude: What is an SVG?
**Content**:
- **Key Points**:
    - **Structure**: SVG is XML-based textual markup.
    - **The Primitive**: The `<path>` tag is the core building block.
        - `M (x,y)`: Move to coordinates.
        - `L (x,y)`: Draw Line to.
        - `C (x1,y1, x2,y2, x,y)`: Cubic Bezier Curve.
    - **Graphics as Code**: Not a grid of pixels, but a sequence of mathematical instructions.

**Speaker Notes**:
"What is an SVG? It is XML-based textual markup. The core building block is the path tag, using commands like Move, Line, and Cubic Bezier curves. Ultimately, SVG is graphics as code—not a grid of pixels, but a sequence of precise mathematical instructions."

---

### Slide 4b: The Task: SVG Image Captioning
**Content**:
- **Goal**: Automatically generate textual descriptions for vector graphics.
- **Why It Matters**:
    - **Accessibility**: Screen readers need alt-text for icons.
    - **Search & Retrieval**: Enable semantic search in icon libraries.
    - **Automation**: Scale description generation for millions of assets.
- **The Challenge**: 
    - Traditional VLMs (BLIP-2, LLaVA) require rasterization → pixel grids.
    - Destroys structural information, wastes compute on background pixels.
- **Our Solution**: 
    - Skip rasterization. Feed SVG code directly to an LLM.
    - Use a specialized encoder (SPE) to convert geometry into embeddings.

**Speaker Notes**:
"Our task is to automatically generate captions for SVG images. This matters for accessibility, semantic search, and automation at scale. The challenge is that current Vision-Language models require rasterizing the SVG to pixels, destroying the structural information. Our solution is simple but powerful: skip rasterization entirely. We feed the SVG code directly to a language model, using a specialized encoder to convert the geometry into embeddings the LLM can understand."

---

### Slide 5: The Core Challenge: Pixels vs. Primitives
**Content**:
- **The Inefficiency of Rasterization**:
    - **Information Loss**: Converting XML to Pixels is irreversible.
    - **Structure Lost**: Semantic relationships between shapes are destroyed.
    - **Compute Waste**: A simple diagonal line requires processing thousands of background pixels.
- **The Vector-Native Opportunity**:
    - **Semantic**: Treat SVG as Code (XML is a language).
    - **Continuous**: No resolution artifacts—a circle is a curve, not pixels.
    - **Efficient**: Dense signal (few tokens) vs sparse pixel patches.

**Speaker Notes**:
"The core challenge is Pixels vs. Primitives. Rasterization causes irreversible information loss and wastes compute on background pixels. Our Vector-Native approach treats SVG as code—semantic, continuous, and efficient with dense signal from fewer tokens."

---

### Slide 6: Methodology I: The Decoder-Only Backbone
**Content**:
- **Why Decoder-Only?**:
    - **Causal Self-Attention**: Trains the model to predict the next token based on context.
    - **Generative Power**: Superior at producing fluent, coherent natural language compared to Encoder-Decoders.
- **Model Selection Strategy**:
    - **Qwen2-7B**: Chosen for strictly better Reasoning & Coding capabilities (crucial for structured XML data).
    - **Gemma-9B**: Google's efficient open model, strong on general knowledge.
    - **Llama-3-8B**: Meta's model, selected for its **Superior Linguistic Quality** (Best-in-class text generation and fluency).

**Speaker Notes**:
"Methodology Part 1 focuses on the Decoder-Only Backbone. I selected this architecture for its Causal Self-Attention and superior generative power in producing fluent language. I chose Qwen2-7B for its strong reasoning and coding capabilities, essential for XML; Gemma-9B for its efficiency; and Llama-3-8B selected for its Superior Linguistic Quality (Best-in-class text generation and fluency)."

---

### Slide 7: Methodology I: Input Paradigm
**Content**:
- **The Linguistic Prior**:
    - The model already knows what a "circle" or "arrow" is conceptually.
    - Our goal is to align visual features to these pre-existing linguistic concepts.
- **Input Paradigm**:
    - **Processing**: The Decoder receives the SVG (whether as Text Tokens or Visual Embeddings) exactly like a sentence, processing drawing commands (M, L, C) sequentially from start to finish.
    - We don't feed the 'raw' XML directly. We first **preprocess it by stripping non-geometric elements** (metadata, `<defs>`, background fills). The clean path commands are then tokenized by the LLM.

**Speaker Notes**:
"We leverage the 'Linguistic Prior': the model conceptually knows what a 'circle' or 'arrow' is. Our goal is to align visual features to these concepts. In our input paradigm, the decoder processes the SVG like a sentence, reading commands sequentially. We don't use raw XML; instead, we preprocess it by stripping non-geometric elements like metadata and background fills, then let the LLM tokenize the clean path commands."

---

### Slide 8: Theoretical Foundation: Low-Rank Adaptation (LoRA)
**Content**:
- **The Constraint**: Full Fine-tuning of 7B+ parameters is computationally prohibitive.
- **The Hypothesis**: Weight updates have a Low Intrinsic Rank.
- **The Method (LoRA)**:
    - **Freeze** original weights W₀ (7B params).
    - **Add** two small matrices: A (down-projection) + B (up-projection).
    - **Forward pass**: output = W₀·x + B·(A·x)
    - Only A and B are trained → **<1% of total params**.

**Speaker Notes**:
"The theoretical foundation is Low-Rank Adaptation (LoRA). Since full fine-tuning is computationally prohibitive, we use the hypothesis that weight updates have a low intrinsic rank. We freeze the pre-trained weights and inject small trainable matrices. This allows us to train less than 0.1% of the parameters, preserving the model's general knowledge."

---

### Slide 9: Methodology I: Text-Only Fine-tuning (LoRA)
**Content**:
- **Approach**: Treat SVG code purely as text.
    - Zero-shot LLMs on raw SVG failed completely (incoherent outputs, XML parsing confusion).
- **The Technique**: Low-Rank Adaptation (LoRA).
    - Instead of full retraining, we inject small matrices (A × B) into the LLM.
    - Allows efficient adaptation to "SVG Language" with <1% params.
- **Result**: Better than Zero-Shot, but suffers from "Geometric Hallucination".
    - **Qwen2 (Text-Only)**: High CLIPScore (32.30) but hallucinates content.

**Speaker Notes**:
"Using LoRA, we efficiently adapt the model to 'SVG Language' with minimal parameters. The result is better than Zero-Shot, but it still suffers from 'Geometric Hallucination'—the model sees the code but remains blind to the actual spatial relationships."

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

### Slide 10b: The Projection Layer (Bridge)
**Content**:
- **The Problem**: Dimension Mismatch.
    - SPE outputs embeddings of size 256.
    - LLM expects embeddings of size 4096.
- **The Solution**: A trainable linear projection.
    - `LLM_input = W_proj · SPE_output + b`
    - Transforms: 256-dim → 4096-dim
- **Training Strategy**:
    - **SPE**: Frozen (pre-trained geometric knowledge).
    - **Projection Layer**: Trainable (learns alignment).
    - **LLM**: LoRA adapters only (<1% params).

**Speaker Notes**:
"The Projection Layer bridges the visual and language worlds. The SPE produces 256-dimensional vectors encoding geometry, but Qwen2's internal dimension is 4096. We use a simple trainable matrix to transform SPE outputs into LLM-compatible inputs. Think of it as a 'translator' that converts geometric features into a language the LLM can understand. The SPE stays frozen to preserve its expertise, while the projection layer learns the optimal mapping."

---

### Slide 11: Dataset Construction
**Content**:
- **Source**: Icons8 (Consistent, high-quality style).
- **Size**: ~90,000 SVG-Caption pairs.
    - **Split**: Stratified split ensures balanced categories in Test Set.
- **Preprocessing**:
    - **Tag Stripping**: Removing XML noise (`<defs>`, metadata).
    - **Canonical ViewBox**: Normalization to 512x512 coordinate space.
    - **Complexity Filter**: Removing paths with > 50 segments (where a segment is a single draw command like `L` or `C`).

**Speaker Notes**:
"For data, we constructed the 90k Stratified Benchmark using Icons8 for its consistent style. Preprocessing was strict: we stripped XML noise, normalized the ViewBox to 512x512, and filtered out overly complex paths to ensure the model focuses on clean semantic signals."

---

### Slide 12: Implementation Details
**Content**:
- **Key Points**:
    - **Stratified Split**: Test set (400 samples) balanced across diverse categories (e.g., UI, Arrows).
- **Dynamic Padding**:
    - **Problem**: Static padding to global max (512 tokens) wastes compute.
    - **Solution**: Pad only to the longest sequence in the current batch.
        - *Example*: If batch max is 120, pad to 120 (not 512).
    - **Why it matters**: Attention is O(N²). Smaller N → exponentially faster.
    - **Result**: ~40% Training Speedup.
    - **Input Schema**:
        - `{"input": "<svg>...", "output": "A red arrow pointing..."}`

**Speaker Notes**:
"Implementation included a stratified split of 1000 samples balanced across categories. A key optimization was 'Dynamic Padding', where we pad only to the batch maximum. This reduced training time by 40% by avoiding wasted computation on empty tokens in the Transformer's quadratic attention mechanism."

---

### Slide 13: Experimental Setup
**Content**:
- **Baselines**:
    1. **Raster Models**: BLIP-2, Florence-2.
    2. **Text-Only Baseline**: Qwen2 / Llama-3 trained on raw SVG code (No SPE).
- **Our Models (Multimodal)**:
    1. **SPE + Qwen2**: The proposed architecture.
    2. **SPE + Gemma**: Investigating architecture impact.
- **Metrics**:
    - **CLIPScore**: Measures visual semantic alignment.
    - **BLEU/ROUGE**: Measures textual fluency.
    - **Composite Score**: Weighted Mean ($\frac{CLIP}{10} + BLEU + METEOR + ROUGE$).

**Speaker Notes**:
"We compared our models against two baselines: Zero-Shot Raster models like BLIP-2 and Text-Only LoRA models. The specific comparison was between SPE-augmented Qwen2/Gemma and their text-only counterparts. Metrics included CLIPScore for visual alignment, BLEU/ROUGE for fluency, and a weighted Composite mean."

---

### Slide 14: Quantitative Results
**Content**:
- **Text-Only Baseline**: High raw CLIPScore (32.30) but hallucinations.
- **SPE + Qwen2**:
    - **Best Composite Score**: **4.18** (Ranking **#1**).
    - **Superior Fluency**: BLEU-1 (0.42) vs Baseline (0.24).
- **Insight**: Structured embeddings act as a regularizer, ensuring valid application of language.
- "Our results were revealing. While the naive Text-Only baseline scored high on raw feature matching, it often hallucinated. The SPE + Qwen2 model achieved the best overall Composite Score and significantly higher fluency. The structured embedding helps the model 'ground' its language in reality."

**Speaker Notes**:
"Quantitative results showed that while the Text-Only baseline had a high raw CLIPScore, it suffered from hallucinations. SPE + Qwen2 achieved the best Composite Score of 4.18 and superior fluency. The structured embeddings act as a regularizer, effectively grounding the model's language in geometric reality."

---

### Slide 15: Qualitative Analysis
**Content**:
- **Case Study 1: Graduate Emoji (Icon #38)**
    - **SPE+Qwen2**: "Stylized human figure with orange shape" ✓ (Correct)
    - **Baseline**: "Hot cross bun" ✗ (Hallucination)

- **Case Study 2: Cross/X Icon (Icon #12)**
    - **SPE+Qwen2**: "Geometric figure representing an 'X'" ✓ (Precise)
    - **Baseline**: Generic description (Misses semantic meaning)

**Speaker Notes**:
"Qualitatively, the model proves its worth. In the Graduate Emoji case, our SPE model correctly identifies a 'stylized human figure', while the baseline hallucinates a 'hot cross bun'—completely wrong. In the Cross icon case, our model precisely identifies the 'X' shape, demonstrating true geometric understanding rather than surface-level pattern matching."

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
    - **Z-Score Normalization**: Critical for training stability.
        - **Problem**: SPE embeddings have different statistics (μ, σ) than what the LLM expects.
        - **Solution**: Normalize: z = (x - μ) / σ → Aligns to mean≈0, var≈1.
        - **Without it**: Training diverges (gradient explosion).
    - **LoRA Rank**: r=16 is optimal. Larger ranks yield diminishing returns.
    - **Prompting**: Simple "Describe this icon" works best.

**Speaker Notes**:
"Our ablation studies revealed critical insights. Z-Score Normalization is essential: the SPE produces embeddings with arbitrary statistics, but the LLM expects normalized inputs. By applying z = (x - μ) / σ, we align the visual features to the language model's expected distribution. Without this, training diverges within 50 steps. We also found that a LoRA rank of 16 provides the optimal balance—higher ranks show diminishing returns while increasing memory cost."

---

### Slide 18: Future Directions
**Content**:
- **Hierarchical Encoding**:
    - Current: SVG linearized as flat sequence.
    - Future: GNNs or Tree-Transformers to preserve DOM structure.
- **Hybrid Dual-Stream**:
    - Vector stream (SPE) + Raster stream (ViT) for textures/gradients.
    - Late fusion via Cross-Attention.
- **Broader Vision**:
    - Vector-Aware Editing: "Make the circle red" → SVG modification.
    - Bidirectional: Text-to-SVG generation using captioned data.

**Speaker Notes**:
"For future work, we see three main directions. First, Hierarchical Encoding: using GNNs to preserve DOM tree structure instead of flat sequences. Second, a Dual-Stream approach combining vector precision with raster texture understanding. Finally, our broader vision includes Vector-Aware Editing and bidirectional Text-to-SVG generation, creating a complete closed loop for vector design."

---

### Slide 21: Conclusions
**Content**:
- **Key Points**:
    - **Thesis Statement**: LLMs can learn to see structure directly, without rasterization.
- **Main Achievement**: Composite Score 4.18.
    - Outperformed **Text-Only (+0.23)** and **Raster Baselines (+0.84)** in quality.
    - **Efficiency**: Validated LoRA for multimodal adaptation (<1% params).
    - **Philosophy**: Moving from "Pixels as Opaque Strings" to "Graphics as Symbolic Code".

**Speaker Notes**:
"To conclude, this work proves that LLMs can learn to see structure directly. We achieved a Composite Score of 4.18, outperforming both text-only and raster baselines. We validated the efficiency of LoRA for multimodal adaptation and advanced the philosophy of treating graphics not as opaque pixels, but as meaningful, symbolic code."

---

### Slide 22: Thank You
**Visual**: "Thank You for your time and attention"

**Speaker Notes**:
"Thank you very much for your time and attention. I am now happy to take any questions."
