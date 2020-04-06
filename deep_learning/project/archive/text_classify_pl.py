import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torchtext
import pytorch_lightning as pl
from sklearn import metrics


def load_dataframe(json_data):
    data = []
    with open(json_data) as f:
        raw = json.load(f)
    for data_split, lines in raw.items():
        for line, label in lines:
            data.append(dict(text=line, label=label, split=data_split))
    df = pd.DataFrame(data)
    return df


def save_subset_csv(df, data_split, folder="."):
    assert data_split in {"train", "test", "val"}
    data_splits = [data_split, "oos_" + data_split]
    df = df[df["split"].isin(data_splits)]
    df = df[["text", "label"]]

    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path_out = folder / (data_split + ".csv")
    df.to_csv(path_out, index=False)
    print("Dataframe saved:", df.shape, path_out)
    return path_out


def get_dataset(data_dir, field_text, field_label, data_split):
    df = load_dataframe(pathlib.Path(data_dir) / "data_full.json")
    path_csv = save_subset_csv(df, data_split)
    fields = [("text", field_text), ("label", field_label)]
    # TODO-2
    # TorchText has its own dataset class with helpful features for text data
    # The dataset will apply the fields aka data transforms
    # Initialize a torchtext dataset that can read csvs!
    # Also, the csvs have header rows so we should skip those
    ################################################################
    dataset = torchtext.data.TabularDataset(
        path_csv, format="csv", fields=fields, skip_header=True
    )
    ################################################################
    return dataset


def get_loader(dataset, device, batch_size):
    # TODO-3
    # TorchText also has special loaders, called iterators
    # Initialize a iterator to help batch the data, transfer to device and shuffle
    # The iterator should also batch examples of similar lengths together
    ################################################################
    iterator = torchtext.data.BucketIterator(
        dataset, batch_size, device=device, shuffle=True
    )
    ################################################################
    return iterator


def show_batch(batch, field_text, field_label):
    print("Showing batch")
    texts = field_text.reverse(batch.text)
    labels = batch.label.cpu().numpy()
    for i in range(len(texts)):
        label = field_label.vocab.itos[labels[i]]
        print("Label:", label)
        print("Text:", texts[i])


class TextClassifier(pl.LightningModule):
    def __init__(self, data_dir, device, batch_size, lr=1e-3, num_hidden=128):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr
        num_vocab, num_labels = self.prepare_data(data_dir)

        # TODO-3
        # Let's make a simple NLP model :)
        # Previously, the text field helped to tokenize and turn the sentences into integers
        # Now, we need an embedding layer to assign each integer to a vector
        # The the vector will represent the "meaning" of each word token
        # The output shape of the embedding layer should be (num_tokens, batch_size, num_hidden)
        # Next, we need a layer to learn the sequential relationship between
        # the words in the input sentences... How about some long short-term memory?
        # The layer will return an output of shape (num_tokens, batch_size, num_hidden),
        # as well as the internal states which we don't need to use for now
        # Finally, a linear layer to make predictions
        ################################################################
        self.embed = torch.nn.Embedding(num_vocab, num_hidden)
        self.lstm = torch.nn.LSTM(input_size=num_hidden, hidden_size=num_hidden)
        self.linear = torch.nn.Linear(num_hidden, num_labels)
        ################################################################

    def forward(self, x):
        x = self.embed(x)
        x, states = self.lstm(x)
        x = torch.mean(x, dim=0)
        x = self.linear(x)
        return x

    def get_loss(self, batch):
        outputs = self.forward(batch.text)
        loss = self.criterion(outputs, batch.label)
        preds = torch.argmax(outputs, dim=-1)
        acc = metrics.accuracy_score(batch.label.cpu(), preds.cpu())
        return loss, acc

    def training_step(self, batch, batch_nb):
        loss, acc = self.get_loss(batch)
        log = dict(train_loss=loss, train_acc=acc)
        return dict(loss=loss, log=log)

    def validation_step(self, batch, batch_nb):
        loss, acc = self.get_loss(batch)
        return dict(val_loss=loss, val_acc=acc)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = np.mean([x["val_acc"] for x in outputs])
        log = dict(val_loss=avg_loss, val_acc=avg_acc)
        return dict(avg_val_loss=avg_loss, avg_val_acc=avg_acc, log=log)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def prepare_data(self, data_dir):
        # TODO-1
        # TorchText describes the data in terms of fields aka data columns
        # The fields define how to transform the data from strings to tensors
        # Initialize torchtext data fields for texts and labels
        # For texts, the field will should apply reversible tokenization and lowercasing
        # For labels, the field should not be sequential
        # https://torchtext.readthedocs.io/en/latest/data.html is your best friend!
        ################################################################
        f_text = torchtext.data.ReversibleField(lower=True)
        f_label = torchtext.data.LabelField()
        ################################################################

        splits = ["train", "val", "test"]
        self.datasets = {s: get_dataset(data_dir, f_text, f_label, s) for s in splits}
        f_text.build_vocab(self.datasets["train"])
        f_label.build_vocab(self.datasets["train"])
        show_batch(next(iter(self.train_dataloader())), f_text, f_label)
        return len(f_text.vocab), len(f_label.vocab)

    @pl.data_loader
    def train_dataloader(self):
        return get_loader(self.datasets["train"], self.device, self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return get_loader(self.datasets["val"], self.device, self.batch_size)


def main(data_dir, batch_size, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = TextClassifier(data_dir, device, batch_size)
    trainer = pl.Trainer(max_epochs=epochs, gpus=-1)
    trainer.fit(model)


if __name__ == "__main__":
    main(data_dir="datasets", batch_size=32, epochs=100, lr=1e-3)
