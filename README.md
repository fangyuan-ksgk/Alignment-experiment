# Alignment-toy

Toy Example for Preference Alignment. We have 5 responses, and pairwise comparative annotation is provided. 
Our target is for the model to do two things: 
    1. Provide Yes / No answer to the question "Is A better than B?" 
    2. Align with annotated preference. 

## Dataset Curation: 

1. Supervised Fine-Tuning (prompt - completion)
```python
python dataset_curation.py -m 1
```

2. Supervised Fine-Tuning (chat template)
```python
python dataset_curation.py -m 2
```

3. Direct Preference Optimization (prompt - chosen - rejected)
```python
python dataset_curation.py -m 3
```

## Experiment: 
Notebook file location: ./colab/
1. Supervised Fine-Tuning with LoRA Adaptor
2. Supervised Fine-Tuning with DoRA Adaptor
