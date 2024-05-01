# Alignment experiment: What does it take to align a LLM?

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
1. Supervised Fine-Tuning with LoRA Adaptor (fine-tuning is quite inefficient for small-sized dataset)
2. Supervised Fine-Tuning with DoRA Adaptor (same as above, plus balancing fine-tune dataset means nothing due to the pre-train dataset's tilted distribution)
3. Evolutionary Model Merging (unstable and not simple enough)
4. Representation Finetuning (This one rocks)

## Evaluation:
Average Perplexity on instruction dataset adopted as evaluation metric. Evaluation ran on modal platform. 
```
modal run eval.py --model-id xxx --dataset-name xxx
```
