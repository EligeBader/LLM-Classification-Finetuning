# 🌟 LLM Classification Finetuning Project 🌟

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen) ![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-orange)

## 🏆 Overview
Welcome to the LLM Classification Finetuning Project! This project is part of an exciting Kaggle competition where we predict which responses users will prefer in a head-to-head battle between chatbots powered by large language models (LLMs).

![Chatbot](https://img.shields.io/badge/Chatbot-LLMs-yellow) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-red)

## 📜 Description
Large language models (LLMs) are rapidly entering our lives, but ensuring their responses resonate with users is crucial. This project tackles this challenge using real-world data from the Chatbot Arena, where users chat with two anonymous LLMs and choose their preferred response.

By developing a winning machine learning model, we aim to improve how chatbots interact with humans and ensure they better align with human preferences. This challenge aligns with the concept of "reward models" or "preference models" in reinforcement learning from human feedback (RLHF).

## 💾 Dataset
The competition dataset consists of user interactions from the ChatBot Arena, where judges provide prompts to two different LLMs and indicate which response they prefer. Our task is to predict these preferences accurately.

### Files
- `train.csv`: Training data with prompts, responses from two models, and judge's selections.
- `test.csv`: Test data for predictions.
- `sample_submission.csv`: Sample submission file in the correct format.

## 🛠 Tools & Technologies
For this project, I used the following tools and technologies:
- **Python 3.8+**: The backbone of our project.
- **PyTorch**: For building and training machine learning models.
- **Transformers (Hugging Face)**: To leverage state-of-the-art LLMs for text generation and classification.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For preprocessing and evaluation.
- **Jupyter Notebook**: For interactive coding and exploration.


## 🎯 Results
After training your model, the results will be saved in the `models/` directory. You can evaluate the performance of your model using the test dataset and the provided `sample_submission.csv`.

## 📜 License
This project is licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. See the [LICENSE](LICENSE) file for more details.

## 🙏 Acknowledgements
A big thank you to Kaggle for providing the platform and the dataset for this competition. Let's make chatbots more human-friendly!
