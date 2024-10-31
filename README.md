# AI-Spotify App Reviewer

## About
This project is part of Natural Language Processing (NLP) section in Artificial Inteligence Engineer fields.

The dataset and video player : https://drive.google.com/drive/folders/15gGFQcowYFrAch4qe3B852R7LL8zsdFv?usp=sharing
![image](https://github.com/user-attachments/assets/1f22e440-d95a-46b8-b38a-4f89d1c3f41f)


## Datasets
The datasets received from [Kaggle 3.4 Million Spotify Google Store Reviews](https://www.kaggle.com/datasets/bwandowando/3-4-million-spotify-google-store-reviews/) with the len approximately 3,377,423 (almost 3,4 million) rows and 9 columns with size amount 655.84 MB. This very big data contains many NULL values that need to be filtered depends on the usage.

## Model
The model I am using is the frontier close-source model GPT-4o Mini, developed by OpenAI. This model was chosen for its accessibility as a frontend solution, providing an easy-to-use interface and low-cost billing options. Its lightweight design makes it suitable for various applications while maintaining high performance, making it an ideal choice for my project.  This model also has been fine-tuned using the dataset I have to optimize its performance for my specific use case, the explanation of this will be conducted in training section

## GPU
Although the model I am using is designed for efficiency, I require GPU performance to accelerate program execution. Therefore, I chose the T4 GPU in Google Colab Pro for the metadata creation and fine-tuning process.

## Data Cleaning
This explanation are based on `training.ipynb`. To support in running the program, since I ran in notebooks (Google Colab), we need to install required libraries.
```
!pip install openai safetensors accelerate
```

We also need to import the modules and connect to openai API
```
import openai
import os
import json
import re
import pandas as pd
import numpy as np
```
Before begin the training, I first performed data cleaning. I used the `FILLNA` function to prevent the delete of any important database. After that, considering the constraints of GPU resources, I randomly selected 1,000 samples from the entire DataFrame to use for fine-tuning. 
Next, I prepare the **training** and **valid** dataset by spliting the whole dataframe sampled into ratio 70%:30%. So the training has to be in require format that already being told in [Platform OpenAI Docs](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset). The training format must be in meta data `jsonl` in form like this:

`{"messages": [{"role": "system", "content": "system prompt as the assistant guidance"}, {"role": "user", "content": "Any database, known-information, and question input"}, {"role": "assistant", "content": "Expected AI answer"}]}`

I combined the datasets of every row to be in the **"user role"** and create few question to make language learning for the model and the result is being save to `./dataset/train-json-file-fix.jsonl` and `./dataset/valid-json-file-fix.jsonl`.

## Training
Now, I prepared the training and validation JSON files for fine-tuning. I used the model **gpt-4o-mini-2024-07-18** for this process by uploading using this scripts:

```
with open(train_file_path, "rb") as f:
   train_file = openai.files.create(file=f, purpose="fine-tune")
with open(valid_file_path, "rb") as f:
   valid_file = openai.files.create(file=f, purpose="fine-tune")
```

Then, we create the fine-tuning key aspect by choosing hyperparameter using this snipped code:
```
modelFINETUNING = 'gpt-4o-mini-2024-07-18'
openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=valid_file.id,
    model=modelFINETUNING,
    seed=42,
    hyperparameters={"n_epochs": 5, "batch_size": 8}, # using gpu (cuda)
    integrations = {"type": "wandb", "wandb": {"project": "gpt-pricer"}},
    suffix="spotify-chatbot-reviewer"
)
```

The integrations are optional for displaying visualization during training process in [wandb.ai](https://wandb.ai/). You may setting your own parameter by changing the epochs (how many times the dataset will be being read) and batch_size (number of samples that will be propagated through the network at one time).


<p align="center">
  <strong>Plot Mean Token Accuracy (Left: Train, Right: Valid)</strong>
</p>
  <img src="https://github.com/user-attachments/assets/28154194-a101-443f-a56b-1462b3542a8f" alt="W B Chart 31_10_2024, 19 17 47" width="40%" style="display: inline-block; margin-right: 10px;"/>
  <img src="https://github.com/user-attachments/assets/58cb1210-21df-46b5-880d-9bf5a1cd9af9" alt="W B Chart 31_10_2024, 19 18 33" width="40%" style="display: inline-block;"/>
</p>

<p align="center">
  <strong>Plot Loss (Left: Train, Right: Valid)</strong>
</p>
  <img src="https://github.com/user-attachments/assets/3761bcc7-056d-4cef-8944-36c4e93d94aa" alt="W B Chart 31_10_2024, 19 17 47" width="40%" style="display: inline-block; margin-right: 10px;"/>
  <img src="https://github.com/user-attachments/assets/292b3278-b0b3-442e-97a7-ca5cff9872d5" alt="W B Chart 31_10_2024, 19 18 33" width="40%" style="display: inline-block;"/>
</p>

Overall, the 'train' dataset shows a significant accuracy value of approximately 1, but it is not quite perfect. Meanwhile, the 'valid' dataset demonstrates relatively sharp performance despite having limited data. From the loss values alone, it is evident that the 'valid' dataset experiences a substantial loss, exceeding 0.2 in a few steps (horizontal bars). Since, the question is still variety, it can be one of the reason that the model considered as underfit, so we need to train more data with pattern question to gain better results. The fine-tuned model named **"ft:gpt-4o-mini-2024-07-18:personal:spotify-chatbot-reviewer-updated:AO57Uu5E"**.

## Inference


