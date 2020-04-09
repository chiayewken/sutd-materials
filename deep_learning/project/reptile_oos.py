from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets import Splits, IntentMetaLoader
from models import LinearClassifierWithOOS
from reptile import ReptileSystem, HyperParams
from utils import acc_score


class ReptileSystemWithOOS(ReptileSystem):
    def __init__(
        self,
        hparams: HyperParams,
        loaders: Dict[str, IntentMetaLoader],
        net: torch.nn.Module,
    ):
        super().__init__(hparams, loaders, net)
        self.loaders_oos = self.get_loaders_oos()
        self.criterion_oos = torch.nn.BCEWithLogitsLoss()

    def get_loaders_oos(self) -> Dict[str, DataLoader]:
        loaders = {}
        self.loaders: Dict[str, IntentMetaLoader]
        for k, meta_loader in self.loaders.items():
            embeds = torch.from_numpy(meta_loader.embeds)
            loaders[k] = DataLoader(
                dataset=TensorDataset(embeds),
                batch_size=self.hparams.bs_inner,
                shuffle=True,
            )
        return loaders

    def run_batch(
        self, batch: Tuple[torch.Tensor, torch.LongTensor], is_train=True
    ) -> Dict[str, float]:
        # loader_oos = self.loaders_oos[Splits.train if is_train else Splits.val]
        # inputs_oos = None
        # for batch_oos in loader_oos:
        #     inputs_oos = batch_oos[0]
        #     break

        with self.get_gradient_context(is_train)():
            inputs, targets = batch
            if is_train:
                self.opt_inner.zero_grad()
            outputs, outputs_pos = self.net(inputs)
            loss = self.criterion(outputs, targets)
            acc = acc_score(outputs, targets)

            # _, outputs_neg = self.net(inputs_oos)
            # loss_pos, acc_pos = self.get_oos_loss(outputs_pos, is_positive=True)
            # loss_neg, acc_neg = self.get_oos_loss(outputs_neg, is_positive=False)
            # loss_oos = (loss_pos + loss_neg) / 2
            # acc_oos = (acc_pos + acc_neg) / 2
            # loss = (loss + loss_oos) / 2

            if is_train:
                loss.backward()
                self.opt_inner.step()

        return dict(
            loss=loss.item(),
            acc=acc.item(),
            # loss_oos=loss_oos.item(),
            # acc_oos=acc_oos.item(),
        )

    def get_oos_loss(
        self, outputs: torch.Tensor, is_positive: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = torch.ones_like(outputs) * int(is_positive)
        loss = self.criterion_oos(outputs, targets)
        acc = acc_score(outputs, targets)
        return loss, acc


def run_intent(root: str):
    hparams = HyperParams(root=root, steps_outer=500, steps_inner=50, bs_inner=10)
    loaders = {s: IntentMetaLoader(hparams, s) for s in ["train", "val"]}
    net = LinearClassifierWithOOS(
        num_in=loaders[Splits.train].embedder.size_embed, num_out=hparams.num_ways
    )
    system = ReptileSystemWithOOS(hparams, loaders, net)
    system.run_train()


def main(root="temp"):
    run_intent(root)


if __name__ == "__main__":
    main()
