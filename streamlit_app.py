"""
MedTeller Streamlit App
=======================
Radiology Report Generation from Chest X-Ray Images

Requirements:
    pip install streamlit torch transformers pillow

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, ViTModel, ViTImageProcessor
from PIL import Image
import os
import re

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="MedTeller - Radiology Report Generator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# MODEL DEFINITION (same as training)
# ============================================================================
class GPT2WithConditioning(nn.Module):
    """GPT-2 with ViT + ClinicalBERT conditioning."""
    
    def __init__(self, base_model, cond_dim=1536):
        super().__init__()
        self.gpt2 = base_model
        self.cond_proj = nn.Linear(cond_dim, self.gpt2.config.n_embd)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, cond_vec=None, **kwargs):
        cond_emb = self.cond_proj(cond_vec)
        cond_emb = cond_emb.unsqueeze(1)
        
        token_emb = self.gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([cond_emb, token_emb], dim=1)
        
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.size(0), 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        if labels is not None:
            prefix_labels = -100 * torch.ones(
                (labels.size(0), 1),
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)
        
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

def normalize_sentences(text: str) -> str:
    """Normalize sentences by capitalizing first letter and proper spacing."""
    if not text or not text.strip():
        return text
    
    # Split on sentence punctuation
    parts = re.split(r'([.?!])', text)
    sentences = []
    
    for i in range(0, len(parts)-1, 2):
        s = parts[i].strip()
        p = parts[i+1]  # punctuation
        
        if len(s) < 3:  # Drop noise/short fragments
            continue
        
        # Capitalize first letter (handle empty string case)
        if s:
            s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
            sentences.append(s + p)
    
    # If no sentences were found, return original text
    if not sentences:
        return text
    
    return " ".join(sentences)


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_models(model_dir: str):
    """Load all models needed for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load tokenizer from model directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    vocab_size = len(tokenizer)
    
    # 2. Load base GPT-2 model from pretrained (NOT from model_dir)
    gpt2_base = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # 3. Resize embeddings to match tokenizer vocab size
    if gpt2_base.config.vocab_size != vocab_size:
        gpt2_base.resize_token_embeddings(vocab_size)
    
    # 4. Create GPT2WithConditioning wrapper
    mm_model = GPT2WithConditioning(gpt2_base, cond_dim=1536)
    
    # 5. Load the trained weights from safetensors/pytorch_model.bin
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_model_path = os.path.join(model_dir, "pytorch_model.bin")
    mm_state_path = os.path.join(model_dir, "mm_model_state.pt")
    
    if os.path.exists(mm_state_path):
        state_dict = torch.load(mm_state_path, map_location=device)
        mm_model.load_state_dict(state_dict)
        
    elif os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        
        # Check if state dict has cond_proj (full model)
        has_cond_proj = any("cond_proj" in k for k in state_dict.keys())
        
        if has_cond_proj:
            mm_model.load_state_dict(state_dict)
        else:
            mm_model.gpt2.load_state_dict(state_dict, strict=False)
            
    elif os.path.exists(pytorch_model_path):
        state_dict = torch.load(pytorch_model_path, map_location=device)
        
        has_cond_proj = any("cond_proj" in k for k in state_dict.keys())
        if has_cond_proj:
            mm_model.load_state_dict(state_dict)
        else:
            mm_model.gpt2.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"No model weights found in {model_dir}")
    
    mm_model = mm_model.to(device)
    mm_model.eval()
    
    # 6. Load ViT for image encoding
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = vit_model.to(device)
    vit_model.eval()
    
    return tokenizer, mm_model, vit_model, vit_processor, device


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================
def get_image_embedding(image: Image.Image, vit_model, vit_processor, device):
    """Extract ViT embeddings from image - using MEAN pooling to match training."""
    # Preprocess image
    inputs = vit_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = vit_model(**inputs)
        # IMPORTANT: Use mean of ALL tokens (including CLS) to match training
        # Training used: img_pooled = img_emb.mean(dim=0) where img_emb was (197, 768)
        all_tokens = outputs.last_hidden_state  # (1, 197, 768)
        img_embedding = all_tokens.mean(dim=1)  # (1, 768) - mean over all 197 tokens
    
    return img_embedding


def generate_report(
    image: Image.Image,
    tokenizer,
    mm_model,
    vit_model,
    vit_processor,
    device,
    max_length: int = 100,
    use_sampling: bool = True,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
):
    """Generate radiology report from chest X-ray image."""
    mm_model.eval()
    
    with torch.no_grad():
        # 1. Get image embedding from ViT (mean pooled, like training)
        img_emb = get_image_embedding(image, vit_model, vit_processor, device)  # (1, 768)
        
        # 2. For text embedding, use a small random vector instead of zeros
        txt_emb = torch.randn(1, 768).to(device) * 0.1
        
        # 3. Concatenate to form condition vector
        cond_vec = torch.cat([img_emb, txt_emb], dim=-1)  # (1, 1536)
        
        # 4. Start generation with BOS token
        gen_ids = tokenizer(tokenizer.bos_token, return_tensors="pt")["input_ids"].to(device)
        
        # Track recent tokens for repetition detection
        recent_tokens = []
        consecutive_punct_count = 0
        
        # 5. Autoregressive generation with improved sampling
        for step in range(max_length):
            token_emb = mm_model.gpt2.transformer.wte(gen_ids)
            cond_emb = mm_model.cond_proj(cond_vec).unsqueeze(1)
            inputs_embeds = torch.cat([cond_emb, token_emb], dim=1)
            attn_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
            
            outputs = mm_model.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
            next_logits = outputs.logits[:, -1, :].clone()
            
            # Apply repetition penalty to already generated tokens
            if repetition_penalty != 1.0:
                for token_id in set(gen_ids[0].tolist()):
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty
            
            if use_sampling:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')
                
                # Sample from filtered distribution
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            # Decode the token to check for repetition
            token_text = tokenizer.decode([next_id.item()]).strip()
            
            # Check for repetitive punctuation
            if token_text in ['.', ',', '!', '?', ';', ':']:
                consecutive_punct_count += 1
                if consecutive_punct_count >= 3:
                    # Stop if too many consecutive punctuation marks
                    break
            else:
                consecutive_punct_count = 0
            
            # Check for repetitive token patterns (same token 4+ times in recent history)
            recent_tokens.append(next_id.item())
            if len(recent_tokens) > 10:
                recent_tokens.pop(0)
            if len(recent_tokens) >= 4 and len(set(recent_tokens[-4:])) == 1:
                # Same token repeated 4 times - stop
                break
            
            gen_ids = torch.cat([gen_ids, next_id], dim=1)
            
            # Stop if EOS token
            if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
                break
        
        # Decode and clean up
        report = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        report = clean_generated_text(report)
        
    return report


def clean_generated_text(text: str) -> str:
    """Clean up generated text by removing repetitive patterns."""
    import re
    
    # Remove excessive periods (more than 1)
    text = re.sub(r'\.{2,}', '.', text)
    
    # Remove excessive spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Remove repetitive punctuation patterns
    text = re.sub(r'(\s*\.\s*){2,}', '. ', text)
    text = re.sub(r'(\s*,\s*){2,}', ', ', text)
    
    # Remove trailing punctuation spam
    text = re.sub(r'[\s\.\,\;\:]+$', '', text)
    
    # Normalize sentences (capitalize, proper spacing)
    text = normalize_sentences(text)
    
    # Add period at end if missing
    text = text.strip()
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text


# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    # Header
    st.title("🩺 MedTeller")
    st.markdown("### Radiology Report Generation")
    st.markdown("Upload a chest X-ray image to generate an automated radiology report.")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model directory input
        model_dir = st.text_input(
            "Model Directory",
            value="./full_multimodal_decoder",
            help="Path to the folder containing model.safetensors and tokenizer files"
        )
        
        st.markdown("---")
        
        # Generation settings
        st.subheader("🎛️ Generation Settings")
        max_length = st.slider("Max Length", 50, 300, 150, help="Maximum number of tokens to generate")
        use_sampling = st.checkbox("Use Sampling", value=True, help="Enable for varied outputs (recommended)")
        
        # Advanced settings (collapsed by default)
        with st.expander("Advanced Settings"):
            top_k = st.slider("Top-K", 0, 100, 50, help="Keep top K tokens for sampling (0 = disabled)")
            top_p = st.slider("Top-P (Nucleus)", 0.1, 1.0, 0.9, 0.05, help="Keep tokens with cumulative prob < P")
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.1, help="Penalize repeated tokens (higher = less repetition)")
        
        st.markdown("---")
        
        st.markdown("""
        **About MedTeller:**
        - Uses ViT for image encoding
        - GPT-2 based decoder with conditioning
        - Trained on IU X-Ray dataset
        
        """)
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        st.warning(f"⚠️ Model directory not found: `{model_dir}`")
        st.info("""
        **Please provide the correct path to your trained model folder.**
        
        The folder should contain:
        - `model.safetensors` (or `pytorch_model.bin`)
        - `tokenizer.json`
        - `vocab.json`
        - `config.json`
        - etc.
        """)
        
        # Show example structure
        st.code("""
# Expected folder structure:
full_multimodal_decoder/
├── model.safetensors      # Your trained model
├── config.json
├── tokenizer.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
└── added_tokens.json
        """)
        return
    
    # Load models
    try:
        with st.spinner("Loading models... This may take a moment."):
            tokenizer, mm_model, vit_model, vit_processor, device = load_models(model_dir)
        st.success(f" Models loaded successfully! Running on: **{device}**")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("Make sure all required files are in the model directory.")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload X-Ray Image")
        
        # Drag and drop zone with custom styling
        st.markdown("""
        <style>
        .uploadedFile {
            border: 2px dashed #1f77b4 !important;
            border-radius: 10px !important;
            padding: 20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # File uploader with drag & drop (built-in) + browse button
        uploaded_file = st.file_uploader(
            "📁 Drag and drop your chest X-ray here or click to browse",
            type=["png", "jpg", "jpeg", "dcm", "bmp", "tiff"],
            help="Supported formats: PNG, JPG, JPEG, DICOM, BMP, TIFF",
            key="xray_uploader"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)
            
            # Image info
            st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
            
            # Generate button
            st.markdown("")
            if st.button("🔬 Generate Report", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image and generating report..."):
                    try:
                        report = generate_report(
                            image=image,
                            tokenizer=tokenizer,
                            mm_model=mm_model,
                            vit_model=vit_model,
                            vit_processor=vit_processor,
                            device=device,
                            max_length=max_length,
                            use_sampling=use_sampling,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty
                        )
                        
                        # Store in session state
                        st.session_state['report'] = report
                        st.session_state['image'] = image
                        st.session_state['filename'] = uploaded_file.name
                        st.success(" Report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {e}")
    
    with col2:
        st.subheader("📋 Generated Report")
        
        if 'report' in st.session_state:
            report = st.session_state['report']
            filename = st.session_state.get('filename', 'uploaded_image')
            
            # Display report in a styled container
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 3px;
                border-radius: 12px;
                margin-bottom: 20px;
            ">
                <div style="
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 10px;
                ">
                    <h4 style="color: #1f77b4; margin-bottom: 15px;">🩺 Radiology Report</h4>
                    <p style="line-height: 1.8; color: #333;">{report}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Download options section
            st.markdown("### 📥 Download Options")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Download as TXT
                st.download_button(
                    label="📄 Download as TXT",
                    data=report,
                    file_name=f"report_{filename.split('.')[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col_dl2:
                # Download as formatted report
                formatted_report = f"""
================================================================================
                        RADIOLOGY REPORT
================================================================================

Patient Image: {filename}
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: MedTeller (GPT-2 + ViT Conditioning)

--------------------------------------------------------------------------------
                           FINDINGS
--------------------------------------------------------------------------------

{report}

--------------------------------------------------------------------------------

DISCLAIMER: This is an AI-generated report for research purposes only.
It should NOT be used for clinical diagnosis. Always consult a qualified 
radiologist for medical interpretation.

================================================================================
                        Generated by MedTeller
================================================================================
"""
                st.download_button(
                    label=" Download Full Report",
                    data=formatted_report,
                    file_name=f"full_report_{filename.split('.')[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Copy to clipboard area
            st.markdown("### Copy Report")
            st.text_area(
                "Select all and copy (Ctrl+A, Ctrl+C):",
                report,
                height=150,
                key="copy_area"
            )
            
            # Clear button
            if st.button(" Clear Report", use_container_width=True):
                del st.session_state['report']
                if 'image' in st.session_state:
                    del st.session_state['image']
                if 'filename' in st.session_state:
                    del st.session_state['filename']
                st.rerun()
                
        else:
            # Empty state with instructions
            st.markdown("""
            <div style="
                background-color: #f8f9fa;
                padding: 40px 20px;
                border-radius: 10px;
                text-align: center;
                border: 2px dashed #dee2e6;
            ">
                <h3 style="color: #6c757d;"> No Report Yet</h3>
                <p style="color: #6c757d;">
                    Upload an X-ray image on the left<br>
                    and click <strong>"Generate Report"</strong><br>
                    to see the Medteller generated radiology report here.
                </p>
            </div>
            """, unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()

