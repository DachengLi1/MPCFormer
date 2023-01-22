---
license: apache-2.0
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- beans
metrics:
- accuracy
model-index:
- name: beans_outputs
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: beans
      type: beans
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9849624060150376
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# beans_outputs

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the beans dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0590
- Accuracy: 0.9850

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.2079        | 1.0   | 130  | 0.2249          | 0.9774   |
| 0.111         | 2.0   | 260  | 0.0825          | 1.0      |
| 0.1526        | 3.0   | 390  | 0.0626          | 1.0      |
| 0.1344        | 4.0   | 520  | 0.0590          | 0.9850   |
| 0.1078        | 5.0   | 650  | 0.0748          | 0.9850   |


### Framework versions

- Transformers 4.20.0.dev0
- Pytorch 1.11.0+cu102
- Datasets 2.2.2
- Tokenizers 0.12.1
