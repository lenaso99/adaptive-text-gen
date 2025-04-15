# User-Driven and Adaptive Text Generation
Contains the code for the term project for the class Computational Linguistics, Winter Semester 2024/25.

## Folder Structure
```
.
├── code
│   ├── finetuning.ipynb
│   └── prompting.py
├── data
│   ├── eng_data_pseudoprompt.xlsx
│   ├── eng_data.xlsx
│   ├── preprocess_data.ipynb
│   └── pseudoprompts.ipynb
├── results
│   ├── gpt-4o-mini_few-shot_ranking_scores_2025-04-12-12-02.xlsx
│   ├── gpt-4o-mini_few-shot_ranking_scores_2025-04-12-12-04.xlsx
│   ├── gpt-4o-mini_few-shot_updating_scores_2025-04-12-12-15.xlsx
│   ├── gpt-4o-mini_reasoning_ranking_scores_2025-04-12-12-06.xlsx
│   ├── gpt-4o-mini_reasoning_ranking_scores_2025-04-12-12-10.xlsx
│   ├── gpt-4o-mini_reasoning_updating_scores_2025-04-12-12-18.xlsx
│   ├── gpt-4o-mini_zero-shot_ranking_scores_2025-04-12-11-58.xlsx
│   ├── gpt-4o-mini_zero-shot_ranking_scores_2025-04-12-12-00.xlsx
│   └── gpt-4o-mini_zero-shot_updating_scores_2025-04-12-12-12.xlsx
├── .gitignore
├── README.md
└── report.pdf
```

## Versions
```
Python: 3.12.3

datasets: 2.20.0
openai: 1.7.2
pandas: 2.2.2
textstat: 0.7.2
transformers: 4.43.2
```


## Instructions
1. Clone the repository.
2. Create a file called <code>.env</code> in the main directory. In it, store your OPENAI api key as follows:
   
```
OPENAI_API_KEY=<your-openai-api-key-here>
```
3. Run the file <code>prompting.py</code>; enable <i>Evaluation Mode</i> to generate files in <i>results</i>.
4. For finetuning, run the cells in <code>finetuning.ipynb</code>.