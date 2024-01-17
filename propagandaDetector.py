import torch
import numpy
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, BertModel, AutoModelForSequenceClassification, Trainer, TrainingArguments


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

tokenizer = AutoTokenizer.from_pretrained(pretrained_bert)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def trainModel():
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

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    model = BertModel.from_pretrained(pretrained_bert,
                                    output_hidden_states=True).to(device)

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

    trainer.save_model('propagandaDetectorModel')

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def giveLabel(input_texts):
    trainedModel = AutoModelForSequenceClassification.from_pretrained('propagandaDetectorModel').to(device)
    tokenizer = AutoTokenizer.from_pretrained('propagandaDetectorModel')
    encoded = tokenizer(input_texts, truncation=True, padding="max_length", return_tensors="pt").to(device)

    logits = trainedModel(**encoded).logits

    tensorResults = torch.argmax(logits, axis=1)

    results = tensorResults.detach().cpu().numpy()

    resultsArr = results.tolist()

    labels = [LABELS[index] for index in resultsArr]
    
    return labels

# If you updated files in the datasets, you can retrain the model:
# trainModel()

# Examples of running the propaganda detection ( [''] means "not propaganda"):
# print(giveLabel(["Ukrainian President Vladimir Zelensky has maintained since the start of the conflict that his military will retake all of Ukraine’s former territories, including Crimea. However, his long-promised summer counteroffensive failed to land Ukraine more than a handful of frontline villages and resulted in the loss of over 125,000 troops and 16,000 pieces of heavy equipment, according to the latest figures from the Russian Defense Ministry."]))
# print(giveLabel(["Something is wrong with our country.",
#                  "The conversation was quite stressful",
#                  "I'm happy today."
#                 ]))
# print(giveLabel(["Eating fresh fruits and vegetables is good for your health"]))
