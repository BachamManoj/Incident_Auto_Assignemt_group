import os
import pandas as pd
import torch
import pickle
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

MODEL_DIR = "bert_model"
ENCODER_PATH = "label_encoder.pkl"

if not os.path.exists(MODEL_DIR):

    df = pd.read_excel("incident_realistic_10000.xlsx")
    df = df[['Short description', 'Description', 'Assignment group']].dropna()

    df['text'] = df['Short description'] + " " + df['Description']

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Assignment group'])

    X_train, X_test, y_train, y_test = train_test_split(
        
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, list(y_train))
    test_dataset = Dataset(test_encodings, list(y_test))

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(set(y_train))
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    pickle.dump(label_encoder, open(ENCODER_PATH, "wb"))

# Load model

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
label_encoder = pickle.load(open(ENCODER_PATH, "rb"))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    short_desc = data.get("short_description", "")
    description = data.get("description", "")

    text = short_desc + " " + description

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    assignment_group = label_encoder.inverse_transform([predicted_class.item()])[0]

    return jsonify({
        "assignment_group": assignment_group,
        "confidence": float(confidence.item())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)