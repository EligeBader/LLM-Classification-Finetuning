# 🌟 LLM Classification Finetuning Project 🌟

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen) ![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1-red) ![Sklearn](https://img.shields.io/badge/Sklearn-0.24.2-yellow) ![Transformers](https://img.shields.io/badge/Transformers-4.5.1-blue)



## 🏆 Overview
Welcome to my LLM Classification Finetuning Project! This project is part of an exciting Kaggle competition where I predict which responses users will prefer in a head-to-head battle between chatbots powered by large language models (LLMs).

![Chatbot](https://img.shields.io/badge/Chatbot-LLMs-yellow) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-red)


## 📜 Description
Large language models (LLMs) are rapidly entering our lives, but ensuring their responses resonate with users is crucial. This project tackles this challenge using real-world data from the Chatbot Arena, where users chat with two anonymous LLMs and choose their preferred response.

By developing a winning machine learning model, I aim to improve how chatbots interact with humans and ensure they better align with human preferences. This challenge aligns with the concept of "reward models" or "preference models" in reinforcement learning from human feedback (RLHF).


## 🏢 Benefits to the Company and Stakeholders

This project offers substantial benefits to the company and stakeholders by improving chatbot interactions through large language model (LLM) classification. By developing models that accurately predict user preferences, the company can enhance user satisfaction and engagement with chatbots. This leads to improved customer service, stronger customer relationships, and increased user retention.

Moreover, showcasing the company's expertise in advanced machine learning techniques, like fine-tuning LLMs, positions the company as a leader in the AI and technology space. This can attract potential clients, partners, and investors who value innovation and cutting-edge technology. Collaborating on this project also promotes a culture of learning and technological advancement within the company, driving overall growth and success.


## 💾 Dataset
The competition dataset consists of user interactions from the ChatBot Arena, where judges provide prompts to two different LLMs and indicate which response they prefer. My task is to predict these preferences accurately.


### Files
- `train.csv`: Training data with prompts, responses from two models, and judge's selections.
- `test.csv`: Test data for predictions.
- `sample_submission.csv`: Sample submission file in the correct format.

Note: The dataset contains text that may be considered profane, vulgar, or offensive.

---

## 🛠 Tools & Technologies
For this project, I used the following tools and technologies:
- **Python 3.8+**: The backbone of my project.
- **PyTorch**: For building and training machine learning models.
- **Transformers (Hugging Face)**: To leverage state-of-the-art LLMs for text generation and classification.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For preprocessing and evaluation.
- **NLTK**: For natural language processing tasks.
- **Jupyter Notebook**: For interactive coding and exploration.

---

## 🔍 Workflow

1. **Data Preprocessing**:
   - Cleaned the text data using NLTK by tokenizing, removing stop words, and lemmatizing.
   - Combined responses for TF-IDF vectorization.

2. **Model 1: Logistic Regression**:
   - Vectorized the text data using TF-IDF.
   - Trained a Logistic Regression model to predict user preferences.
   - Evaluated the model using log loss.

3. **Model 2: BERT (Bidirectional Encoder Representations from Transformers)**:
   - Used the `transformers` library to load a pre-trained BERT model.
   - Tokenized the text data and created custom datasets.
   - Fine-tuned the BERT model using PyTorch.
   - Evaluated the model using the Trainer API from the `transformers` library.

---

## 📂 Project Structure
```
- LLM_Classification_Finetuning
  - data/
    - train.csv
    - test.csv
    - sample_submission.csv
  - notebooks/
    - LLMS_Classification_Finetuning.ipynb
  - models/
    - model_bert.pth
  - README.md
```

---

## 🎯 Results
After training my models, the results are saved in the `models/` directory. I evaluated the performance of my models using the test dataset and the provided `sample_submission.csv`.

---

## 🌟 Improvements
To further enhance this project, I can explore the following:

- **Experiment with Different Models**:
  - Fine-tune other transformer models such as RoBERTa, GPT-2, or T5.
  - Try traditional machine learning models like Support Vector Machines (SVM) or Random Forest.
  - Experiment with ensemble methods to combine predictions from multiple models.

- **Optimize Hyperparameters**:
  - Use techniques like grid search to find the best hyperparameters for my models.

- **Feature Engineering**:
  - Explore additional text preprocessing techniques and feature engineering methods to improve model performance.

- **Data Augmentation**:
  - Apply data augmentation techniques to increase the diversity and size of the training dataset.

---

## 📜 License
This project is licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

## 🙏 Acknowledgements
A big thank you to Kaggle for providing the platform and the dataset for this competition. Let's make chatbots more human-friendly!
