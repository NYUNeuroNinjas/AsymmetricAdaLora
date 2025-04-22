# AsymmetricAdaLora
In this work, we analyze the effectiveness of parameter-efficient fine-tuning (PEFT) techniques to build upon Low-Rank Adaptation (LoRA) modules, as well as data augmentation for natural language processing, in the training and validation of the RoBERTa architecture with less than 1 million trainable parameters. Trained and evaluated on the AG News dataset, the model reaches 85.4\% accuracy on the testing data.
## Official Submission and Model Checkpoint
The notebook and saved model checkpoints that produced the inferences for our official Kaggle leaderboard position are located in /official_submission. To produce an inference with this model, include this code before the final test:
```
from peft import PeftModel

# Load the base model
base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

# Load the saved LoRA adapters
model = PeftModel.from_pretrained(base_model, "./official_submission/model_checkpoint")
```
## Final Model
Our final model referenced in the report can be found in AsymmetricAdaLora.ipynb and AsymmetricAdaLora.pdf.
