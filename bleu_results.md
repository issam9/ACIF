## Zero-Shot Speech-to-Text Translation BLEU Scores

### CoVoST-2 (En → X)

| Model         | DE      | ZH      |
| -------------------------------- | ------- | ------- |
| LLaMA-3.1-8B - Stage 1           |   21.41 |    31.00 |
| LLaMA-3.1-8B - Stage 2 (KD)      |   23.76 |    33.44 | 
| LLaMA-3.1-8B - Stage 2 (CE)      |   23.50 |    33.14 | 
| Qwen2.5-7B - Stage 1 (Base)      |   21.96 |    34.98 | 
| Qwen2.5-7B - Stage 2 (KD)        |   21.92 |    35.12 | 
| Qwen2.5-7B - Stage 2 (CE)        |   21.24 |    34.35 |

### Europarl-ST (En → X)

| Model         | DE      | ES      | FR      | IT      | 
| -------------------------------- | ------- | ------- | ------- | ------- | 
| LLaMA-3.1-8B - Stage 1           |   18.07 |   24.40 |   19.78 |   15.15 |  
| LLaMA-3.1-8B - Stage 2 (KD)      |   18.67 |   26.44 |   21.04 |   15.44 |
| LLaMA-3.1-8B - Stage 2 (CE)      |   18.88 |   26.18 |   20.49 |   16.15 |
| Qwen2.5-7B - Stage 1 (Base)      |   18.15 |   25.28 |   20.49 |   15.39 | 
| Qwen2.5-7B - Stage 2 (KD)        |   18.19 |   25.45 |   20.80 |   15.52 | 
| Qwen2.5-7B - Stage 2 (CE)        |   17.53 |   25.60 |   20.66 |   14.94 | 

