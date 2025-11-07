# MedTeller: Vision-Language Transformer Architecture

## Project: Automated Radiology Report Generation
**Course**: DATA 612 - Deep Learning  
**Platform**: UMD Zaratan HPC Cluster  
**Dataset**: Indiana University Chest X-Ray Collection (3,851 reports, 7,466 images)

---

## 1. Architecture Overview

MedTeller is a **Vision-Language Transformer** that combines:
- **Vision Encoder**: Processes chest X-ray images
- **Text Decoder**: Generates radiology reports
- **Cross-Attention Mechanism**: Bridges visual and textual information

```
Input X-Ray Image (224x224x3)
         ↓
   Vision Encoder (ViT-Base)
         ↓
   Image Features [196 x 768]
         ↓
   Cross-Attention Layer
         ↓
   Text Decoder (GPT-2)
         ↓
   Generated Report (Text)
```

---

## 2. Component Specifications

### 2.1 Vision Encoder: Vision Transformer (ViT-Base)

**Architecture**:
- **Model**: Pre-trained ViT-Base/16 (ImageNet-21k)
- **Input Size**: 224 × 224 × 3 RGB images
- **Patch Size**: 16 × 16 pixels
- **Number of Patches**: 196 (14 × 14)
- **Hidden Dimension**: 768
- **Number of Layers**: 12 transformer blocks
- **Attention Heads**: 12 heads per layer
- **MLP Dimension**: 3072
- **Parameters**: ~86M

**Processing Pipeline**:
```python
# Image preprocessing
image = resize(xray, (224, 224))
image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Patch embedding
patches = split_image_to_patches(image, patch_size=16)  # [196, 768]
patches = add_positional_encoding(patches)

# Transformer encoding
for layer in vit_layers:
    patches = multi_head_self_attention(patches)
    patches = feedforward_network(patches)

visual_features = patches  # [196, 768]
```

**Key Features**:
- Pre-trained on ImageNet for robust feature extraction
- Position embeddings capture spatial relationships
- Self-attention captures long-range dependencies in images
- Fine-tuned on medical imaging data

---

### 2.2 Text Decoder: GPT-2 Small

**Architecture**:
- **Model**: Pre-trained GPT-2 Small
- **Vocabulary Size**: 50,257 tokens (BPE tokenization)
- **Context Length**: 512 tokens
- **Hidden Dimension**: 768 (matches ViT)
- **Number of Layers**: 12 transformer blocks
- **Attention Heads**: 12 heads per layer
- **MLP Dimension**: 3072
- **Parameters**: ~124M

**Processing Pipeline**:
```python
# Text tokenization
tokens = tokenizer.encode("<BOS> " + report_text)  # [1, seq_len]
token_embeddings = embedding_layer(tokens)  # [1, seq_len, 768]

# Add positional encoding
token_embeddings = token_embeddings + positional_encoding

# Decoder with cross-attention
for layer in gpt2_layers:
    # Self-attention on previously generated tokens
    hidden = multi_head_self_attention(token_embeddings, causal_mask=True)
    
    # Cross-attention to visual features
    hidden = cross_attention(hidden, visual_features)
    
    # Feedforward
    hidden = feedforward_network(hidden)

# Output projection
logits = output_projection(hidden)  # [1, seq_len, 50257]
next_token = argmax(logits[:, -1, :])
```

**Key Features**:
- Pre-trained on large text corpus for language understanding
- Autoregressive generation (left-to-right)
- Causal masking prevents looking at future tokens
- Fine-tuned on medical report generation

---

### 2.3 Cross-Attention Mechanism

**Purpose**: Bridge visual features from ViT to text generation in GPT-2

**Architecture**:
```python
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12):
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, text_hidden, visual_features):
        # text_hidden: [batch, seq_len, 768]
        # visual_features: [batch, 196, 768]
        
        Q = self.query_proj(text_hidden)        # [batch, seq_len, 768]
        K = self.key_proj(visual_features)      # [batch, 196, 768]
        V = self.value_proj(visual_features)    # [batch, 196, 768]
        
        # Multi-head attention
        Q = Q.reshape(batch, seq_len, num_heads, head_dim)
        K = K.reshape(batch, 196, num_heads, head_dim)
        V = V.reshape(batch, 196, num_heads, head_dim)
        
        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / sqrt(head_dim)  # [batch, num_heads, seq_len, 196]
        attn_weights = softmax(scores, dim=-1)
        
        # Attend to visual features
        context = attn_weights @ V  # [batch, num_heads, seq_len, head_dim]
        context = context.reshape(batch, seq_len, 768)
        
        output = self.out_proj(context)
        return output
```

**Key Features**:
- Query from text decoder, Key/Value from vision encoder
- 12 attention heads for diverse feature interactions
- Attention weights indicate which image regions are relevant for each word
- Enables interpretability through attention visualization

---

## 3. Training Strategy

### 3.1 Loss Function

**Primary Loss**: Cross-Entropy Loss for next-token prediction
```python
loss = CrossEntropyLoss(logits, target_tokens)
```

**Auxiliary Loss (Optional)**: Disease tag prediction
```python
# Multi-label classification for medical findings
disease_logits = classifier(visual_features.mean(dim=1))  # [batch, 121]
disease_loss = BCEWithLogitsLoss(disease_logits, disease_labels)

total_loss = report_generation_loss + 0.1 * disease_loss
```

### 3.2 Optimization

**Optimizer**: AdamW
- Learning Rate: 5e-5 (with warmup)
- Weight Decay: 0.01
- Betas: (0.9, 0.999)

**Learning Rate Schedule**: Linear warmup + Cosine decay
```python
warmup_steps = 500
total_steps = num_epochs * steps_per_epoch

lr_schedule = LinearWarmup(warmup_steps) + CosineAnnealingLR(total_steps)
```

**Gradient Clipping**: Max norm = 1.0

### 3.3 Training Configuration

**Hardware (Zaratan HPC)**:
- GPU: 1× NVIDIA H100 (80GB VRAM) or V100 (32GB VRAM)
- CPUs: 16 cores
- Memory: 64GB RAM

**Hyperparameters**:
- Batch Size: 16 (with gradient accumulation if needed)
- Epochs: 50-100
- Max Sequence Length: 512 tokens
- Gradient Accumulation Steps: 4 (effective batch size = 64)

**Data Augmentation (Images)**:
- Random horizontal flip (50% probability)
- Random rotation (±10 degrees)
- Color jitter (brightness, contrast)
- Random crops and resizing

**Regularization**:
- Dropout: 0.1 in transformer layers
- Label smoothing: 0.1
- Early stopping: Patience = 5 epochs

---

## 4. Model Complexity Analysis

### 4.1 Parameter Count

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| ViT-Base Encoder | 86M | Fine-tuned (last 6 layers) |
| Cross-Attention Layers | 9M | Trainable (×12 layers) |
| GPT-2 Decoder | 124M | Fine-tuned (last 6 layers) |
| Output Projection | 38M | Trainable |
| **Total** | **257M** | **~150M trainable** |

### 4.2 Computational Complexity

**Training (per epoch)**:
- Forward pass: ~2.5 TFLOPs per image-report pair
- Backward pass: ~5 TFLOPs per image-report pair
- Total: ~7.5 TFLOPs per sample
- For 3,851 samples: ~29 PFLOPs per epoch

**Memory Requirements**:
- Model weights (FP32): ~1 GB
- Optimizer states (AdamW): ~2 GB
- Activations (batch=16): ~4 GB
- **Total VRAM**: ~8 GB (easily fits on H100/V100)

**Inference Time**:
- Vision encoding: ~10 ms
- Report generation (100 tokens): ~200 ms
- **Total**: ~210 ms per X-ray

### 4.3 Why This Architecture is Complex

1. **Multi-Modal Integration**: Combining vision and language modalities requires careful alignment of feature spaces

2. **Cross-Attention Design**: Novel mechanism to bridge visual and textual representations, requiring architectural innovation

3. **Medical Domain Adaptation**: Pre-trained models need significant fine-tuning for medical terminology and clinical reasoning

4. **Long-Range Dependencies**: Transformers handle global context, crucial for identifying subtle radiological findings

5. **Sequential Generation**: Autoregressive decoding with beam search for coherent report generation

6. **Large Scale**: 257M parameters require careful optimization and regularization

---

## 5. Implementation Details

### 5.1 Data Pipeline

```python
class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir, tokenizer, max_length=512):
        self.reports = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.reports.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize report
        report = self.reports.iloc[idx]['findings'] + ' ' + self.reports.iloc[idx]['impression']
        tokens = self.tokenizer.encode(report, max_length=self.max_length, truncation=True)
        
        return {
            'image': image,
            'input_ids': tokens[:-1],  # Teacher forcing
            'labels': tokens[1:],       # Shifted for next-token prediction
        }
```

### 5.2 Model Architecture Code

```python
class MedTeller(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Text decoder
        self.text_decoder = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            CrossAttentionLayer(hidden_dim=768, num_heads=12)
            for _ in range(12)
        ])
        
        # Freeze early layers (keep last 6 trainable)
        for param in self.vision_encoder.encoder.layer[:6].parameters():
            param.requires_grad = False
        for param in self.text_decoder.transformer.h[:6].parameters():
            param.requires_grad = False
    
    def forward(self, images, input_ids, labels=None):
        # Encode image
        visual_features = self.vision_encoder(images).last_hidden_state  # [B, 196, 768]
        
        # Decode text with cross-attention
        hidden_states = self.text_decoder.transformer.wte(input_ids)  # Token embeddings
        
        for i, (decoder_layer, cross_attn) in enumerate(zip(
            self.text_decoder.transformer.h, self.cross_attention
        )):
            # Self-attention in decoder
            hidden_states = decoder_layer(hidden_states)
            
            # Cross-attention to visual features
            hidden_states = hidden_states + cross_attn(hidden_states, visual_features)
        
        # Generate logits
        logits = self.text_decoder.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}
```

---

## 6. Evaluation Metrics

### 6.1 Natural Language Generation Metrics

1. **BLEU Score** (Bilingual Evaluation Understudy)
   - BLEU-1, BLEU-2, BLEU-4
   - Measures n-gram overlap with reference reports

2. **ROUGE Score** (Recall-Oriented Understudy for Gisting Evaluation)
   - ROUGE-L (Longest Common Subsequence)
   - Focuses on recall of important content

3. **METEOR** (Metric for Evaluation of Translation with Explicit ORdering)
   - Considers synonyms and stemming
   - Better aligned with human judgment

### 6.2 Medical-Specific Metrics

1. **CheXbert Score**
   - Uses BERT-based classifier to extract medical findings
   - Compares extracted findings between generated and reference reports
   - Macro F1-score across 14 medical conditions

2. **RadGraph F1**
   - Constructs knowledge graphs from reports
   - Compares entities and relations
   - More clinically relevant than text-only metrics

### 6.3 Human Evaluation

- Readability (1-5 scale)
- Clinical accuracy (verified by radiologists)
- Completeness of findings

---

## 7. Expected Performance

Based on similar work and our dataset size:

| Metric | Expected Range | Target |
|--------|----------------|--------|
| BLEU-4 | 0.15 - 0.25 | > 0.20 |
| ROUGE-L | 0.30 - 0.40 | > 0.35 |
| METEOR | 0.25 - 0.35 | > 0.30 |
| CheXbert F1 | 0.40 - 0.55 | > 0.45 |

**Baseline Comparison**:
- Template-based: BLEU-4 ~0.10
- Retrieval-based: BLEU-4 ~0.18
- Our approach: BLEU-4 ~0.20-0.25 (expected)

---

## 8. Challenges and Solutions

### 8.1 Technical Challenges

| Challenge | Solution |
|-----------|----------|
| **Medical terminology** | Fine-tune on medical corpus + domain-specific tokenizer |
| **Class imbalance** | Weighted loss + data augmentation + oversampling |
| **Long reports** | Hierarchical generation + sliding window attention |
| **Multi-view images** | Image pooling or sequential encoding |
| **Hallucinations** | Constrained decoding + fact verification |

### 8.2 Computational Challenges

| Challenge | Solution |
|-----------|----------|
| **Large model size** | Mixed-precision training (FP16) + gradient checkpointing |
| **Memory constraints** | Gradient accumulation + smaller batch sizes |
| **Long training time** | Multi-GPU training + efficient data loading |

---

## 9. Innovation and Contributions

1. **Adapter Architecture**: Custom cross-attention layers specifically designed for medical image-to-text translation

2. **Multi-Task Learning**: Joint training for report generation and disease classification improves both tasks

3. **Domain Adaptation**: Transfer learning from general vision-language models to radiology domain

4. **Interpretability**: Attention visualization shows which image regions influence generated text

5. **Clinical Relevance**: Focus on medically-important metrics (CheXbert, RadGraph) beyond generic NLG metrics

---

## 10. Future Enhancements

1. **Multi-Image Input**: Handle multiple views (frontal + lateral) simultaneously
2. **Structured Reports**: Generate reports following standard radiology templates
3. **Interactive Editing**: Allow radiologists to correct and refine generated reports
4. **Uncertainty Quantification**: Indicate confidence levels for findings
5. **Multi-Lingual Support**: Generate reports in multiple languages

---

## References

1. Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
2. Radford et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
3. Chen et al. (2020). "Generating Radiology Reports via Memory-driven Transformer"
4. Liu et al. (2021). "Contrastive Attention for Automatic Chest X-ray Report Generation"
5. Smit et al. (2020). "CheXbert: Combining Automatic Labelers and Expert Annotations"

---

**Document Version**: 1.0  
**Last Updated**: November 6, 2024  
**Author**: MedTeller Team  
**Platform**: UMD Zaratan HPC Cluster

