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
- [Demo](#demo)  
- [License](#license)  

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
- Split data: 80% train, 10% validation, 10% test  

---

## Model Architecture
**Frameworks:** PyTorch, Hugging Face Transformers  

**Components:**  
- **Vision Encoder:** Pretrained ViT-Base (Vision Transformer) to extract patch embeddings from X-rays  
- **Text Decoder:** GPT-2 / BART fine-tuned on medical reports for natural language generation  
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
- METEOR  

**Clinical Accuracy Metrics:**  
- CheXbert Score  
- RadGraph F1  

**Qualitative Verification:**  
- Cross-attention heatmap visualization  
- Manual evaluation for fluency and medical correctness  

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

## Demo
A web-based interface using **Streamlit** allows:  
1. Uploading a chest X-ray image  
2. Receiving a fully generated radiology report  

---

## License
[MIT License](LICENSE)  

---

**Acknowledgements:**  
- IU X-Ray Dataset (Indiana University)  
- Hugging Face Transformers  
- PyTorch  

