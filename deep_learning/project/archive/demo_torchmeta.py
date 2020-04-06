import os

import torch
import torchmeta
import tqdm


def update_parameters(
    model: torchmeta.modules.MetaModule,
    loss: torch.Tensor,
    step_size: float = 0.5,
    first_order: bool = False,
) -> dict:
    # Update the parameters of the model, with one step of gradient descent.
    grads = torch.autograd.grad(
        loss, model.meta_parameters(), create_graph=(not first_order)
    )

    params = {}
    for (name, param), grad in zip(model.meta_named_parameters(), grads):
        params[name] = param - step_size * grad

    return params


def get_accuracy(logits: torch.FloatTensor, targets: torch.LongTensor) -> torch.Tensor:
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def conv3x3(in_channels, out_channels, **kwargs):
    return torchmeta.modules.MetaSequential(
        torchmeta.modules.MetaConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, **kwargs
        ),
        torchmeta.modules.MetaBatchNorm2d(
            out_channels, momentum=1.0, track_running_stats=False
        ),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
    )


class ConvolutionalNeuralNetwork(torchmeta.modules.MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = torchmeta.modules.MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
        )

        self.classifier = torchmeta.modules.MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(
            inputs, params=torchmeta.modules.utils.get_subdict(params, "features")
        )
        features = features.view((features.size(0), -1))
        logits = self.classifier(
            features, params=torchmeta.modules.utils.get_subdict(params, "classifier")
        )
        return logits


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def train(
    folder,
    num_shots=5,
    num_ways=5,
    first_order=True,
    step_size=0.4,
    hidden_size=64,
    output_folder=None,
    batch_size=16,
    num_batches=100,
    num_workers=2,
    criterion=torch.nn.CrossEntropyLoss(),
):
    device = get_device()
    dataset = torchmeta.datasets.helpers.omniglot(
        folder,
        shots=num_shots,
        ways=num_ways,
        shuffle=True,
        test_shots=15,
        meta_train=True,
        download=True,
    )
    dataloader = torchmeta.utils.data.BatchMetaDataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    model = ConvolutionalNeuralNetwork(1, num_ways, hidden_size=hidden_size)
    model.to(device=device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    with tqdm.tqdm(dataloader, total=num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch["train"]
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)

            test_inputs, test_targets = batch["test"]
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)

            outer_loss = torch.tensor(0.0, device=device)
            accuracy = torch.tensor(0.0, device=device)
            for (
                task_idx,
                (train_input, train_target, test_input, test_target),
            ) in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):
                # print(
                #     dict(
                #         train_input=train_input.shape,
                #         train_target=train_target.shape,
                #         test_input=test_input.shape,
                #         test_target=test_target.shape,
                #     )
                # )

                train_logit = model(train_input)
                inner_loss = criterion(train_logit, train_target)

                model.zero_grad()
                params = update_parameters(
                    model, inner_loss, step_size=step_size, first_order=first_order
                )

                test_logit = model(test_input, params=params)
                outer_loss += criterion(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            outer_loss = outer_loss / batch_size
            accuracy = accuracy / batch_size

            outer_loss.backward()
            meta_optimizer.step()

            pbar.set_postfix(accuracy="{0:.4f}".format(accuracy.item()))
            if batch_idx >= num_batches:
                break

    # Save model
    if output_folder is not None:
        filename = os.path.join(
            output_folder,
            "maml_omniglot_" "{0}shot_{1}way.pt".format(num_shots, num_ways),
        )
        with open(filename, "wb") as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)


if __name__ == "__main__":
    train(".")
