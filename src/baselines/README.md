This directory supports two use cases.

#### (1) Train teacher models by fine-tuning and grid search according to the original Bert/Roberta paper.
This requires three arguments: task_name is the dataset name in the GLUE benchmark, metric_name is the metric name for the task:
    
    'MNLI': 'eval_accuracy',
    'QQP': 'eval_combined_score',
    'QNLI': 'eval_accuracy',
    'SST2': 'eval_accuracy',
    'RTE': 'eval_accuracy',
    'CoLA': 'eval_matthews_correlation',
    'STSB': 'eval_combined_score',
    'MRPC': 'eval_f1',
    'imdb': 'eval_accuracy',

model_name is the model backbone in [bert-base-uncased, bert-large-uncased, roberta-base]
    
For instance, to train a BERT base teacher in RTE:

    python train_teacher.py --task_name RTE --metric_name eval_accuracy --model_name bert-base-uncased
    
The output model can be found in ./tmp/{exp_name}/{task_name}/{model_name}. Please manually move the best model to ~/checkpoints/exp/[task_name] for further training (i.e. baseline, MPCFormer).

#### (2) Train baseline models with approximations
S0 is MPCFormer_w/o{p,d} and S1 is MPCFormer_w/o{d}. To train a S0 baseline with quad and 2quad approximation on RTE, run:

    python baseline.py --task_name RTE --model_path ~/checkpoints/exp/RTE --baseline_type S0 --hidden_act quad --softmax_act 2quad
