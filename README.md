# Alignment-toy

Toy Example for Preference Alignment

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