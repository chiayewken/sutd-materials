import pytorch_lightning as pl
import torch
import torchmeta


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


class DataParams:
    def __init__(self, folder, num_workers=1):
        self.num_workers = num_workers
        self.folder = folder


class MetaParams:
    def __init__(
        self,
        step_size=0.4,
        first_order=True,
        batch_size=16,
        num_batches=100,
        num_shots=5,
        num_ways=5,
        lr=1e-3,
    ):
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.lr = lr
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.first_order = first_order
        self.step_size = step_size


class ConvNet(torchmeta.modules.MetaModule, pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_features,
        hidden_size=64,
        mp: MetaParams = None,
        dp: DataParams = None,
    ):
        super(ConvNet, self).__init__()
        self.mp = mp
        self.dp = dp
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.criterion = torch.nn.CrossEntropyLoss()

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        train_inputs, train_targets = batch["train"]
        test_inputs, test_targets = batch["test"]

        outer_loss = torch.tensor(0.0, device=get_device())
        accuracy = torch.tensor(0.0, device=get_device())
        for (
            task_idx,
            (train_input, train_target, test_input, test_target),
        ) in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):
            train_logit = self.forward(train_input)
            inner_loss = self.criterion(train_logit, train_target)

            self.zero_grad()
            params = update_parameters(
                self,
                inner_loss,
                step_size=self.mp.step_size,
                first_order=self.mp.first_order,
            )

            test_logit = self.forward(test_input, params=params)
            outer_loss += self.criterion(test_logit, test_target)

            with torch.no_grad():
                accuracy += get_accuracy(test_logit, test_target)

        outer_loss = outer_loss / self.mp.batch_size
        accuracy = accuracy / self.mp.batch_size
        log = dict(outer_loss=outer_loss, accuracy=accuracy)
        return dict(loss=outer_loss, progress_bar=log)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.mp.lr)

    @pl.data_loader
    def train_dataloader(self):
        dataset = torchmeta.datasets.helpers.omniglot(
            self.dp.folder,
            shots=self.mp.num_shots,
            ways=self.mp.num_ways,
            shuffle=True,
            test_shots=15,
            meta_train=True,
            download=True,
        )
        return torchmeta.utils.data.BatchMetaDataLoader(
            dataset,
            batch_size=self.mp.batch_size,
            shuffle=True,
            num_workers=self.dp.num_workers,
        )


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def train(folder):
    mp = MetaParams()
    dp = DataParams(folder)
    model = ConvNet(1, mp.num_ways, mp=mp, dp=dp)
    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    train(".")
