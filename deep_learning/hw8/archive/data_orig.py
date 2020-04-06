import torch
import glob
import unicodedata
import string
import pathlib
from torchvision.datasets.utils import download_and_extract_archive

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)


def download(root: str) -> pathlib.Path:
    root = pathlib.Path(root)
    url = "https://download.pytorch.org/tutorial/data.zip"
    data_dir = root / "data"

    if not data_dir.exists():
        download_and_extract_archive(url, str(root))
    assert data_dir.exists()
    return data_dir


def findFiles(path):
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# Build the category_lines dictionary, a list of lines per category
root = ""
category_lines = {}
all_categories = []
data_dir = download(root)

# for filename in findFiles("../data/names/*.txt"):
#     category = filename.split("/")[-1].split(".")[0]
for path in data_dir.glob("names/*.txt"):
    category = path.stem
    all_categories.append(category)
    # lines = readLines(filename)
    lines = readLines(path)
    category_lines[category] = lines

n_categories = len(all_categories)
print(dict(all_letters=all_letters, n_letters=n_letters))
print(dict(all_categories=all_categories, data_dir=data_dir, n_categories=n_categories))
