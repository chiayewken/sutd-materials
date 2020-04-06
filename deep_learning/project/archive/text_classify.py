import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchtext
import tqdm


def load_full_dataframe(json_data):
    data = []
    with open(json_data) as f:
        raw = json.load(f)
    for type_split, lines in raw.items():
        for line, label in lines:
            data.append(dict(text=line, label=label, split=type_split))
    df = pd.DataFrame(data)
    return df


def split_full_df(df):
    dfs = dict(train=None, test=None, val=None)
    fields = ["text", "label"]
    for s in dfs.keys():
        values = [s, "oos_" + s]
        dfs[s] = df[df["split"].isin(values)][fields]
        print("Dataframe shape:", s, dfs[s].shape)
    return dfs


def save_dataframes(dfs, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for s in dfs.keys():
        paths[s] = folder / (s + ".csv")
        dfs[s].to_csv(paths[s], index=False)
        print("Dataframe saved:", paths[s])
    return paths


def get_dataset(path_csv, field_text, field_label):
    # TODO-2
    # TorchText has its own dataset class with helpful features for text data
    # The dataset will apply the fields aka data transforms
    # Initialize a torchtext dataset that can read csvs!
    # Also, the csvs have header rows so we should skip those
    ################################################################
    fields = [("text", field_text), ("label", field_label)]
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


class Net(torch.nn.Module):
    def __init__(self, num_vocab, num_labels, num_hidden=128):
        super().__init__()

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


def run_epoch(net, loader, criterion, optimizer, training):
    if training:
        net.train()
        context_grad = torch.enable_grad
    else:
        net.eval()
        context_grad = torch.no_grad

    correct = 0
    total = 0
    losses = []
    with context_grad():
        for batch in loader:
            if training:
                optimizer.zero_grad()
            inputs = batch.text
            labels = batch.label
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            losses.append(loss.item())
            if training:
                loss.backward()
                optimizer.step()

    acc = np.round(correct / total, decimals=3)
    loss = np.round(np.mean(losses), decimals=3)
    return dict(loss=loss, acc=acc)


def train(net, criterion, optimizer, num_epochs, loaders, earlystop):
    best_loss = 1e9
    results = dict(best_weights=None, metrics=[])

    for e in tqdm.tqdm(range(num_epochs)):
        metrics = dict(
            train=run_epoch(net, loaders["train"], criterion, optimizer, training=True),
            val=run_epoch(net, loaders["val"], criterion, optimizer, training=False),
        )
        results["metrics"].append(metrics)
        print(dict(epoch=e, metrics=metrics))

        loss_monitor = metrics["val"]["loss"]
        if loss_monitor < best_loss:
            best_loss = loss_monitor
            results["best_weights"] = copy.deepcopy(net.state_dict())
        elif earlystop:
            break

    return results


def main(data_dir, batch_size, epochs, lr, earlystop):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dfs = split_full_df(load_full_dataframe(f"{data_dir}/data_full.json"))
    paths = save_dataframes(dfs, folder="temp")
    print("Sample of original data format:", dfs["train"].sample(10))

    # TODO-1
    # TorchText describes the data in terms of fields aka data columns
    # The fields define how to transform the data from strings to tensors
    # Initialize torchtext data fields for texts and labels
    # For texts, the field will should apply reversible tokenization and lowercasing
    # For labels, the field should not be sequential
    # https://torchtext.readthedocs.io/en/latest/data.html is your best friend!
    ################################################################
    field_text = torchtext.data.ReversibleField(lower=True)
    field_label = torchtext.data.LabelField()
    ################################################################

    datasets = {s: get_dataset(p, field_text, field_label) for s, p in paths.items()}
    field_text.build_vocab(datasets["train"])
    field_label.build_vocab(datasets["train"])
    loaders = {s: get_loader(ds, device, batch_size) for s, ds in datasets.items()}
    show_batch(next(iter(loaders["train"])), field_text, field_label)

    net = Net(num_vocab=len(field_text.vocab), num_labels=len(field_label.vocab))
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    _ = train(net, criterion, optimizer, epochs, loaders, earlystop)


if __name__ == "__main__":
    main(data_dir="datasets", batch_size=32, epochs=100, lr=1e-3, earlystop=True)
