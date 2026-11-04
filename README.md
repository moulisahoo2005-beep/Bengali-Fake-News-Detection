# Bengali Fake News and Rumor Detection

## Project Overview
A Bengali news classification project using a fine-tuned BanglaBERT model and an interactive Streamlit dashboard.

## Model Information
- Model: csebuetnlp/banglabert
- Architecture: ElectraForSequenceClassification
- Classes:
  - 0: Fake News
  - 1: Authentic News
- Maximum sequence length: 512
- Saved decision threshold: 0.485

## Final Test Results
- Accuracy: 97.00%
- Macro Precision: 96.97%
- Macro Recall: 92.30%
- Macro F1: 94.44%
- Balanced Accuracy: 92.30%

## Technologies
Python, PyTorch, Hugging Face Transformers, Pandas, Scikit-learn, Matplotlib, Seaborn and Streamlit.

## Project Structure
- BanglaBERT_Final_Model/: saved trained model and tokenizer
- data/raw/: original dataset files
- data/processed/: cleaned dataset and split files
- dashboard/: Streamlit application
- results/: evaluation results and visualizations

## Important Note
Predictions are model classifications, not independent fact-checking or verification of news.
