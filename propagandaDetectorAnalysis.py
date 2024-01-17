import torch
import numpy
import scipy
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, BertModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader

# pip install transformers==4.24.0
# pip install datasets==2.7.1
# pip install evaluate==0.3.0

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
use_gpu = True  # Change this flag as needed

if use_gpu:
  # Check the GPU is detected
  if not torch.cuda.is_available():
    print("ERROR: No GPU detected. Please add a GPU; if you're using Colab, use their UI.")
    assert False
  # Get the GPU device name.
  device_name = torch.cuda.get_device_name()
  n_gpu = torch.cuda.device_count()
  print("Found device: {}, n_gpu: {}".format(device_name, n_gpu))
else:
  # Check that no GPU is detected
  if torch.cuda.is_available():
    print("ERROR: GPU detected.")
    print("Remove the GPU or set the use_gpu flag to True.")
    assert False
  print("No GPU found. Using CPU.")
  print("WARNING: Without a GPU, your code will be extremely slow.")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

pretrained_bert = 'bert-base-uncased'
LABELS = [
    ['Name_Calling,Labeling'],
    ['Repetition'],
    ['Slogans'],
    ['Appeal_to_fear-prejudice'],
    ['Doubt'],
    ['Exaggeration,Minimisation'],
    ['Flag-Waving'],
    ['Loaded_Language'],
    ['Reductio_ad_hitlerum'],
    ['Bandwagon'],
    ['Causal_Oversimplification'],
    ['Obfuscation,Intentional_Vagueness,Confusion'],
    ['Appeal_to_Authority'],
    ['Black-and-White_Fallacy'],
    ['Thought-terminating_Cliches'],
    ['Red_Herring'],
    ['Straw_Men'],
    ['Whataboutism'],
    ['']
]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_split(dataset):
  labelsToDrop = ["document_id", "label"] #[str(label) for label in LABELS]

  texts = numpy.array(dataset.drop(labelsToDrop, axis=1))
  texts = [str(sample[0]) for sample in texts]

  labels = numpy.array(dataset.drop(["text", "document_id"], axis=1))
  for i in range(0, len(labels)):
    if labels[i] in LABELS:
      labels[i] = LABELS.index(labels[i])
    else:
      labels[i] = LABELS.index([''])
  labels = [sample[0] for sample in labels]
  
  return texts, labels


# Training data
train_texts, train_labels = build_split(pd.read_csv("./datasets/train_csv/data.csv"))
test_texts, test_labels = build_split(pd.read_csv("./datasets/test_csv/data.csv"))
dev_texts, dev_labels = build_split(pd.read_csv("./datasets/dev_csv/data.csv"))

NUM_TRAIN = len(train_labels)
NUM_DEV = len(dev_labels)
NUM_TEST = len(test_labels)

print("train split: {} reviews".format(len(train_labels)))
print("dev split: {} reviews".format(len(dev_labels)))
print("test split: {} reviews".format(len(test_labels)))

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_confidence_intervals(accuracy, sample_size, confidence_level):
  """ calling this with arguments (0.8, 100, .95) returns
  the lower and upper bounds of a 95% confidence interval
  around the accuracy of 0.8 on a test set of size 100."""
  z_score = -1 * scipy.stats.norm.ppf((1-confidence_level)/2)
  standard_error = numpy.sqrt(accuracy * (1-accuracy) / sample_size)
  lower_ci = accuracy - standard_error*z_score
  upper_ci = accuracy + standard_error*z_score
  return lower_ci, upper_ci

# Example: if you had 80% accuracy on an N=250 sized test set, your CI is [75.0%...85.0%]
# get_confidence_intervals(0.8, 250, .95)
# Example: For a much larger test set, your CI is much smaller
# get_confidence_intervals(0.8, 10000, .95)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(pretrained_bert)
model = BertModel.from_pretrained(pretrained_bert,
                                  output_hidden_states=True).to(device)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_bert_features(input_texts):
    features = []
    for i, text in enumerate(input_texts):
        input = tokenizer.encode(text, truncation=True,
                                    return_tensors="pt").to(device)
        hidden_states = model(input).hidden_states
        feature = None

        feature = []
        cls_token_index = 0
        for layer in range(1, 13):
            hidden_state_arr = hidden_states[layer].detach().cpu().numpy()
            #print(hidden_state_arr[0][cls_token_index])
            feature.append(hidden_state_arr[0][cls_token_index])

        feature = numpy.array(feature)

        assert feature.shape == (12, 768)
        features.append(feature)

    return numpy.stack(features)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Extract features for the training and test sets
from timeit import default_timer as timer

start = timer()
train_features = extract_bert_features(train_texts)
test_features = extract_bert_features(test_texts)
end = timer()
print("Extracted features in {:.1f} minutes".format((end-start)/60))

assert train_features.shape == (NUM_TRAIN, 12, 768)
assert test_features.shape == (NUM_TEST, 12, 768)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def countTruncatedInput(train_texts, tokenizer, maxTokens): #maxTokens = 510
  train_truncated = 0
  test_truncated = 0

  for i, text in enumerate(train_texts):
      tokens = tokenizer.tokenize(text)
      if len(tokens) > maxTokens:
        train_truncated += 1

  for i, text in enumerate(test_texts):
      tokens = tokenizer.tokenize(text)
      if len(tokens) > maxTokens:
        test_truncated += 1


  print("train: {} inputs truncated".format(train_truncated))
  print("test: {} inputs truncated".format(test_truncated))

countTruncatedInput(train_texts, tokenizer, 510)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def logisticRegressionClassifierOnLayers(numLayers, train_features, test_features, train_labels, test_labels): #numLayers = 12
  for i in range(numLayers):
    y_pred = None

    lr_model = LogisticRegression(max_iter=10000)

    layer_train_features = numpy.array(train_features)[: , i]
    layer_test_features = numpy.array(test_features)[: , i]

    lr_model.fit(layer_train_features, train_labels)
    y_pred = lr_model.predict(layer_test_features)

    acc = (y_pred == test_labels).sum()/len(test_labels)
    print("Layer {}: {:.3f} accuracy, 95% CI [{:.3f}, {:.3f}]".format(i+1, acc, *get_confidence_intervals(acc, NUM_TEST, 0.95)))

logisticRegressionClassifierOnLayers(12, train_features, test_features, train_labels, test_labels)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PropagandaDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels
    self.tokenizer = tokenizer

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

train_encodings = tokenizer(train_texts, truncation=True)
dev_encodings = tokenizer(dev_texts, truncation=True)
test_encodings = tokenizer(test_texts, truncation=True)

train_dataset = PropagandaDataset(train_encodings, train_labels)
dev_dataset = PropagandaDataset(dev_encodings, dev_labels)
test_dataset = PropagandaDataset(test_encodings, test_labels)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Source: https://huggingface.co/transformers/training.html
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = numpy.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def model_init():
  return AutoModelForSequenceClassification.from_pretrained(
    pretrained_bert, num_labels=len(LABELS))

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    evaluation_strategy="epoch",     # evaluation occurs after each epoch
    logging_dir='./logs',            # directory for storing logs
    logging_strategy="epoch",        # logging occurs after each epoch
    log_level="error",               # set logging level
    optim="adamw_torch",             # use pytorch's adamw implementation
    learning_rate=3e-5,
    seed=42,

)

trainer = Trainer(
    model_init=model_init,            # method instantiates model to be trained
    args=training_args,               # training arguments, defined above
    train_dataset=train_dataset,      # training dataset
    eval_dataset=dev_dataset,         # evaluation dataset
    compute_metrics=compute_metrics,  # function to be used in evaluation
    tokenizer=tokenizer,              # enable dynamic padding
)

trainer.train()
val_accuracy = trainer.evaluate()['eval_accuracy']

print()
print()
print("FINAL: Validation Accuracy {:.3f}, 95% CI [{:.3f}, {:.3f}]".format(val_accuracy, *get_confidence_intervals(val_accuracy, NUM_DEV, 0.95)))