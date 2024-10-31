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
This explanation are based on `training.ipynb`. Before begin the training, I first performed data cleaning. I used the `FILLNA` function to prevent the delete of any important database. After that, considering the constraints of GPU resources, I randomly selected 1,000 samples from the entire DataFrame to use for fine-tuning. 
Next, I prepare the **training** and **valid** dataset by spliting the whole dataframe sampled into ratio 70%:30%. So the training has to be in require format that already being told in [Platform OpenAI Docs](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset). The training format must be in meta data `jsonl` in form like this:

`{"messages": [{"role": "system", "content": "system prompt as the assistant guidance"}, {"role": "user", "content": "Any database, known-information, and question inpuy"}, {"role": "assistant", "content": "Expected AI answer"}]}`

I combined the datasets of every row to be in the **"user role"** to make language learning for the model and the result is being save to `./dataset/train-json-file-fix.jsonl` and `./dataset/valid-json-file-fix.jsonl`.

## Training
Now, I prepared the training and validation JSON files for fine-tuning. I used the model **gpt-4o-mini-2024-07-18** for this process by uploading using this scripts:

```
with open(train_file_path, "rb") as f:
   train_file = openai.files.create(file=f, purpose="fine-tune")
```

Then, we create the fine-tuning key aspect by choosing hyperparameter using this snipped code:
```
modelFINETUNING = 'gpt-4o-mini-2024-07-18'
openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=valid_file.id,
    model=modelFINETUNING,
    seed=42,
    hyperparameters={"n_epochs": 2, "batch_size": 8}, # using gpu (cuda)
    integrations = {"type": "wandb", "wandb": {"project": "gpt-pricer"}},
    suffix="spotify-chatbot-reviewer"
)
```

The integrations are optional for displaying visualization during training process in [https://wandb.ai/]
