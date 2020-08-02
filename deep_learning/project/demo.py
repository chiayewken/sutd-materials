from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from sklearn import decomposition

from datasets import Splits, MetaTask, IntentEmbedBertDataset, MetaBatch
from reptile import MetaLearnSystem, get_hparams_intent
from utils import HyperParams, cosine_distance


class DemoSystem(MetaLearnSystem):
    def __init__(self, hp: HyperParams):
        super().__init__(hp)

    def show_intro(self):
        if self.hp.algo == "reptile":
            st.write(
                """
            Reptile is an optimization-based approach to meta-learning, and is theoretically similar to 
            first-order MAML (related to the famous MAML algorithm).
            It only requires black-box access to optimizers such as SGD or Adam, 
            and has similar accuracy and speed while being relatively simple to implement.
            """
            )
        elif self.hp.algo == "prototype":
            st.write(
                """
            Prototypical networks are an example of metric-learning applied
            to the meta-learning context. They are conceptually related to clustering methods
            such as K-means where centroids used to define cluster membership.
            Classification predictions are obtained by ranking distances to the representations
            of prototypes for each class label. As it has a simpler inductive bias, it
            can even be used for zero-shot prediction.
            """
            )

    def get_human_readable(self, x: torch.Tensor) -> Tuple[List[str], List[str]]:
        dataset: IntentEmbedBertDataset = self.loaders[Splits.train].ds_orig
        embeds = torch.from_numpy(dataset.embeds).to(self.device)
        distances = cosine_distance(x, embeds)
        ranking = distances.argsort(dim=-1, descending=False)
        indices = ranking[:, 0].cpu().numpy()
        texts = list(np.array(dataset.texts)[indices])
        labels = list(np.array(dataset.labels)[indices])
        return texts, labels

    def get_unique_labels(self, y: torch.Tensor, labels: List[str]) -> List[str]:
        # Index of label class can be random and not in order of appearance
        assert len(y) == len(labels)
        unique = [""] * len(set(labels))
        assert len(unique) == self.hp.num_ways
        indices = y.cpu().numpy()
        for i in range(len(labels)):
            unique[indices[i]] = labels[i]
        return unique

    def get_label2texts(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, List[str]]:
        texts, labels = self.get_human_readable(batch[0])
        label2texts = {lab: [] for lab in labels}
        for i in range(len(texts)):
            label2texts[labels[i]].append(texts[i])
        return label2texts

    def show_task(self, task: MetaTask):
        x, y = task.train
        texts, labels = self.get_human_readable(x)
        st.write(f"Target labels for task:")
        st.write(str(self.get_unique_labels(y, labels)))
        if st.checkbox(label="Show All Samples"):
            keys = list(vars(task).keys())
            k = st.selectbox("Data Split", options=keys)
            st.write({k: self.get_label2texts(getattr(task, k))})

    def analyze_model_outputs(self, task: MetaTask):
        plt.clf()
        fig, axes = plt.subplots(ncols=3)
        color_cycle = self.get_colors()
        pca = None
        unique_labels = None
        for i, data_split in enumerate(vars(task).keys()):
            x, y = getattr(task, data_split)
            with torch.no_grad():
                outputs = self.net(x).cpu().numpy()
            if unique_labels is None:
                texts, labels = self.get_human_readable(x)
                unique_labels = self.get_unique_labels(y, labels)
            if pca is None:
                pca = decomposition.PCA(n_components=2)
                pca.fit(outputs)
            outputs_pca = pca.transform(outputs)
            colors = [color_cycle[i] for i in y.cpu().numpy()]
            axes[i].scatter(*zip(*outputs_pca), c=colors)
            axes[i].title.set_text(str(data_split))
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        st.pyplot()

    def analyze_task(self, meta_batch: MetaBatch):
        st.header("Task Analysis")
        st.write(
            """
        Each task or episode is created by randomly sampling a subset of classes
        from the original dataset, with a fixed number of examples per class.
        This is to simulate the few-shot learning setting across diverse tasks.
        The meta-learner is trained to generalize quickly to unseen tasks.
        """
        )
        st.write(f"Number of samples per class: {self.hp.num_shots}")
        st.write(f"Number of classes: {self.hp.num_ways}")
        st.write(f"Train/val/test samples: **{self.hp.num_ways * self.hp.num_shots}**")
        tasks_all = meta_batch.get_tasks()
        i = st.selectbox(label="Task Index", options=range(len(tasks_all))[::-1])
        task = tasks_all[i]
        self.show_task(task)
        st.write("Overall test Set performance for this task:")
        st.write((self.loop_inner(task)))
        st.subheader("Model Output Visualization")
        self.analyze_model_outputs(task)
        return task

    def get_colors(self):
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        return prop_cycle.by_key()["color"][: self.hp.num_ways]

    def analyze_sample(self, task: MetaTask):
        st.header("Sample Analysis")
        keys = list(vars(task).keys())[::-1]
        data_split = st.selectbox("Data Split from Task", options=keys)
        x, y = getattr(task, data_split)
        texts, labels = self.get_human_readable(x)
        unique_labels = self.get_unique_labels(y, labels)

        if st.checkbox(label="Use custom text input", value=True):
            i = 0
            x = self.get_custom_sample()
        else:
            i = texts.index(st.selectbox("Sample Text", options=texts))
            st.write(f"Ground truth label: **{labels[i]}**")
        with torch.no_grad():
            probs = self.forward(x[[i]]).softmax(dim=-1).squeeze().cpu().numpy()

        plt.clf()
        plt.bar(unique_labels, probs, color=self.get_colors())
        plt.title("Model Predictions")
        plt.ylabel("Softmax Probabilily")
        plt.xticks(rotation=10)
        st.pyplot()

    def get_custom_sample(self, text="what is the meaning of tomato") -> torch.Tensor:
        text = st.text_input(label="Custom Text", value=text)
        dataset: IntentEmbedBertDataset = self.loaders[Splits.train].ds_orig
        embed = dataset.embedder.embed_texts([text], cache_name="", delete_cache=True)
        return torch.from_numpy(embed).to(self.device)

    def run_demo(self):
        self.load()
        self.show_intro()
        meta_batch = MetaBatch(self.samples[Splits.test], self.device)
        task = self.analyze_task(meta_batch)
        self.analyze_sample(task)
        print("Demo Done")


def main():
    st.title("Meta-learning for Text Classification")
    st.write(
        """
    Meta-learning is the process of learning how to learn. 
    A meta-learner takes in a distribution of tasks, 
    where each task is a learning problem, and it produces a quick learner — 
    a learner that can generalize from a small number of examples. 
    One well-studied meta-learning problem is few-shot classification, 
    where each task is a classification problem where the learner only sees 
    1–5 input-output examples from each class, and then it must classify new inputs. 
    """
    )
    st.header("Meta-learning Algorithm")
    algo = st.selectbox("Algorithm", options=["reptile", "prototype"])
    hp = get_hparams_intent(algo=algo)
    system = DemoSystem(hp)
    system.run_demo()


if __name__ == "__main__":
    main()
