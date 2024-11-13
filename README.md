# Medical Question Answering System

This repository is a part of the Natural Language Processing course, designed to build a Medical Question Answering System (MQAS) using the T5 transformer model. The system is trained and fine-tuned to generate answers to medical questions based on PubMed abstracts and relevant contexts. It supports BLEU, ROUGE, and BERTScore evaluation metrics to assess model performance.

## Project Structure

- **Prompt1, Prompt2, Prompt3&4&5**: Each folder contains two Jupyter notebooks (`UseT5.ipynb` and `FinetuneT5.ipynb`) that implement and fine-tune the T5 model on various medical prompts.
  - **UseT5.ipynb**: Loads a pre-trained T5 model and evaluates it on specific prompts by calculating similarity scores (BLEU, ROUGE, BERTScore) between generated answers and reference answers.
  - **FinetuneT5.ipynb**: Fine-tunes the T5 model on a subset of medical questions to improve its performance on answering context-based queries.

- **NLP-HW4-Documentation.pdf**: Comprehensive documentation explaining the methodology, datasets, model architecture, and evaluation metrics used in the project.

## Key Features

- **Pre-trained T5 Model Usage**: Implements the T5 transformer model for generating answers to medical questions. Uses pretrained weights from Hugging Face and fine-tunes them on medical question datasets for improved accuracy.
- **Evaluation Metrics**: Includes BLEU, ROUGE, and BERTScore metrics to evaluate the model's performance, providing a multi-faceted view of its accuracy and relevance.
  - **BLEU**: Measures n-gram overlap between generated and reference answers.
  - **ROUGE**: Calculates recall-based overlap, particularly useful for summarization tasks.
  - **BERTScore**: Uses contextual embeddings to measure semantic similarity between the generated and reference texts.
- **Context Retrieval**: Retrieves relevant PubMed abstracts based on cosine similarity for better context generation in answering questions.
- **Dataset Preparation**: Includes data preprocessing steps for tokenization and cleaning to make the dataset suitable for question answering.

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/MedicalQuestionAnsweringSystem.git
   cd MedicalQuestionAnsweringSystem
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Required Data**:
   - Download the PubMedQA dataset and any additional PubMed abstracts as per the dataset requirements in the notebooks.
   - Load your Kaggle API credentials to access datasets directly if needed.

## Usage

### Step 1: Running Pre-trained T5 Model

1. **Navigate to the `PromptX` folder** (replace `X` with the prompt number).
2. Open and run `2_NLP_HW4_UseT5.ipynb` to load the pre-trained T5 model.
3. Evaluate the model using BLEU, ROUGE, and BERTScore metrics.

### Step 2: Fine-tuning T5 Model

1. **Navigate to the `PromptX` folder** (replace `X` with the prompt number).
2. Open and run `3_NLP_HW4_FinetuneT5.ipynb` to fine-tune the T5 model on medical-specific questions.
3. After training, the model and tokenizer are saved locally for later use in prediction and evaluation.

### Step 3: Evaluating Model Performance

- Use the evaluation code in `2_NLP_HW4_UseT5.ipynb` to calculate BLEU, ROUGE, and BERTScore metrics.
- Compare generated answers to reference answers to assess the model’s accuracy and relevance.

### Example Workflow

1. **Prepare Data**: Run data preprocessing cells in `UseT5.ipynb` to clean and tokenize the dataset.
2. **Retrieve Contexts**: Use the context retrieval functions to fetch relevant abstracts for each question.
3. **Fine-tune and Evaluate**: Fine-tune the model using `FinetuneT5.ipynb` and evaluate the model on BLEU, ROUGE, and BERTScore metrics.

## Evaluation Metrics

- **BLEU Score**: Calculated for n-gram similarity with weights adjusted for BLEU-1 through BLEU-4.
- **ROUGE Score**: Considers ROUGE-1, ROUGE-2, and ROUGE-L scores for a comprehensive evaluation.
- **BERTScore**: Leverages BERT embeddings to capture semantic similarity, particularly useful for medical language understanding.

## Future Enhancements

- **Extended Fine-tuning**: Fine-tune the model on larger and more diverse datasets for better generalization on medical questions.
- **Real-time QA System**: Integrate the model into an interactive application or API for real-time question answering.
- **Advanced Context Retrieval**: Experiment with other retrieval methods to fetch more relevant contexts from medical databases.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please submit issues, feature requests, or pull requests to help improve the Medical Question Answering System.