# User-Driven and Adaptive Text Generation
Contains the code for the term project for the class Computational Linguistics, Winter Semester 2024/25.

## Folder Structure
```
.
├── code
│   ├── base_model.py
│   └── prompting.py
├── data
│   └── preprocess_data.ipynb
├── models
├── notebooks
├── results
├── .gitignore
└── README.md
```

## Versions
Python: 3.12.3

datasets: 2.20.0
huggingface_hub: 0.27.0
openai: 1.7.2
pandas: 2.2.2
sklearn: 1.4.2
transformers: 4.43.2
torch: 2.3.0+cpu


## Instructions
1. Clone the repository.
2. Create a file called <code>.env</code> in the main directory. In it, store your huggingface token and OPENAI api key as follows:
   
```
OPENAI_API_KEY=<your-openai-api-key-here>
HF_TOKEN=<your-huggingface-token-here>
```
3. 
