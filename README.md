# MedTeller: Vision-Language Transformer for Automated Radiology Report Generation

## Table of Contents
- [Project Overview](#project-overview)  
- [Motivation](#motivation)  
- [Problem Statement](#problem-statement)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Expected Outcomes](#expected-outcomes)  
- [Impact](#impact)  
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Demo](#demo)   

---

## Project Overview
**MedTeller** is a Vision-Language Transformer designed to automatically generate radiology reports from chest X-ray images. By combining state-of-the-art computer vision and natural language processing techniques, MedTeller aims to assist radiologists in producing accurate and consistent reports, reducing manual effort and mitigating human variability.

---

## Motivation
Radiology is critical for medical diagnosis, but interpreting images and writing reports is time-consuming. Radiologists often review hundreds of X-rays per day, which can lead to fatigue and inconsistencies. MedTeller leverages Transformer architectures to:  
- Provide draft radiology reports  
- Flag potential abnormalities  
- Ensure clinical consistency  

---

## Problem Statement
Traditional AI models in medical imaging focus on single-disease classification. In real clinical practice, radiologists require **comprehensive report generation** covering multiple findings and impressions.  

**Objective:** Develop a Vision-Language Transformer system that:  
1. Inputs a chest X-ray  
2. Identifies anatomical features and abnormalities  
3. Generates a clinically accurate, human-like radiology report  

---

## Dataset
**Primary Dataset:** IU X-Ray Dataset (Indiana University)  
- 7,470 chest X-ray images paired with 3,955 radiology reports  
- Available via NIH OpenI repository and Kaggle  

**Preprocessing Steps:**  
- Resize images to 224 × 224 pixels and normalize intensity values  
- Extract "Findings" and "Impression" sections from reports  
- Tokenize text using Byte Pair Encoding (BPE) or WordPiece tokenizer  
- Split data: 70% train, 15% validation, 15% test  

---

## Model Architecture
**Frameworks:** PyTorch, Hugging Face Transformers  

**Components:**  
- **Vision Encoder:** Pretrained ViT-Base (Vision Transformer) to extract patch embeddings from X-rays  
- **Text Decoder:** GPT-2 / ClinicalBERT fine-tuned on medical reports for natural language generation  
- **Cross-Attention Layer:** Aligns visual features with generated text tokens  
- **Auxiliary Libraries:** scikit-learn, NLTK for evaluation metrics  

**Training Objective:**  
- Minimize cross-entropy loss for report generation  
- Optional auxiliary loss for disease tag prediction  
- Optimizer: AdamW with learning rate scheduler  

---

## Evaluation Metrics
**Textual Similarity Metrics:**  
- BLEU-1/2/4  
- ROUGE-L  

---

## Expected Outcomes
- Trained Vision-Language Transformer generating radiology reports from unseen X-rays  
- Target metrics: BLEU-4 > 0.25, ROUGE-L > 0.3  
- Streamlit web demo allowing X-ray upload and report generation  

---

## Impact
MedTeller can:  
- Reduce report generation time by 40–50%  
- Assist radiologists in preliminary diagnosis  
- Improve clinical documentation efficiency  
- Serve as a foundation for future multimodal AI systems handling CT/MRI scans  

---

## Installation & Setup

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **pip** (Python package manager)
- **Git** (to clone the repository)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd MedTeller
```

### Step 2: Create a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `streamlit>=1.28.0` - Web interface framework
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.30.0` - Hugging Face transformers library
- `pillow>=9.0.0` - Image processing
- `safetensors>=0.4.0` - Safe model loading

**Note:** If you encounter issues installing PyTorch, you may need to install it separately based on your system (CPU or GPU). Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for platform-specific installation instructions.

### Step 4: Verify Model Files

Ensure that the trained model files are present in the `full_multimodal_decoder/` directory. The directory should contain:

```
full_multimodal_decoder/
├── mm_model_state.pt          # Trained model weights
├── tokenizer.json             # Tokenizer configuration
├── vocab.json                 # Vocabulary file
├── merges.txt                 # BPE merges (if applicable)
├── special_tokens_map.json    # Special tokens mapping
├── tokenizer_config.json      # Tokenizer settings
└── added_tokens.json          # Additional tokens (if any)
```

**Important:** If the model files are not present, you'll need to:
1. Train the model using the provided training scripts, OR
2. Download the pre-trained model from the repository releases/assets

---

## Running the Application

### Option 1: Use the Live Demo (No Installation Required)

You can try MedTeller immediately without any setup:
- **🌐 Live Demo:** [https://medteller.streamlit.app/](https://medteller.streamlit.app/)

Simply visit the link, upload a chest X-ray image, and get instant radiology report generation.

### Option 2: Run Locally

### Starting the Streamlit App

Once all dependencies are installed and model files are in place, you can start the web application:

```bash
streamlit run streamlit_app.py
```

The application will:
1. Load the trained model and tokenizer
2. Start a local web server
3. Automatically open your default web browser to the application URL (typically `http://localhost:8501`)

If the browser doesn't open automatically, you can manually navigate to the URL shown in the terminal output.

### Using the Application

1. **Upload an Image:**
   - Click on the upload area or drag and drop a chest X-ray image
   - Supported formats: PNG, JPG, JPEG, DICOM, BMP, TIFF

2. **Generate Report:**
   - Click the "Generate Report" button
   - Wait for the model to process the image and generate the report
   - The generated radiology report will appear in the right panel

3. **Download Results:**
   - Download the report as a plain text file
   - Download a formatted report with metadata
   - Copy the report text directly from the text area

### Model Directory Configuration

If your model files are in a different location, you can specify the path in the sidebar's "Model Directory" field. The default path is `./full_multimodal_decoder`.

---

## Demo

Try MedTeller online without any installation:
- **🌐 Live Application:** [https://medteller.streamlit.app/](https://medteller.streamlit.app/)

The live demo allows you to upload chest X-ray images and generate radiology reports instantly using our trained model.


---

**Acknowledgements:**  
- IU X-Ray Dataset (Indiana University)  
- Hugging Face Transformers  
- PyTorch  

