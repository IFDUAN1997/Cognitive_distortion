import csv
import torch
from transformers import AutoTokenizer, BertTokenizer, AutoModelForSequenceClassification

# Load the model from the .pth file
model_path = 'roberta_modelD10Psy862.pth'
# model = torch.load(model_path)
model_dict = torch.load(model_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='../pretrained_models/chinese-roberta-wwm-ext', state_dict=model_dict)
model.eval()
#print(model)

# Load the tokenizer
bert_path = "../pretrained_models/chinese-roberta-wwm-ext/"
tokenizer = BertTokenizer(vocab_file=bert_path+"vocab.txt")  # 初始化分词器

def single():
    # Tokenize your input sentences
    input_text = "所有人都欺负我，我觉得或者毫无意义了，他们骂我，欺负我，侮辱我"
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")

    # Pass the input to the model to get the output
    with torch.no_grad():
        output = model(input_ids)

    logits = output.logits

    # If you're doing classification, get the predicted class label
    predicted_class = torch.argmax(logits, dim=1).item()

    print(predicted_class)


def readFromFile(input_file,output_file):
    # Open the input file
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        sentences = f.readlines()

    # Predict the sentiment for each sentence and write the results to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence', 'sentiment'])
        for sentence in sentences:
            # Tokenize the sentence
            input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')
            
            # Pass the input to the model to get the output
            with torch.no_grad():
                output = model(input_ids)
            logits = output.logits

            # Get the predicted class label
            
            predicted_class = torch.argmax(logits, dim=1).item()

            # Write the sentence and predicted sentiment to the CSV file
            writer.writerow([sentence.strip(), predicted_class])

input_file="input.txt"
output_file="output.csv"
readFromFile(input_file,output_file)