import pathlib
import zipfile

import pandas as pd
import requests
import tqdm


def download(url, save_dir=".", ignore_exist=False):
    name = pathlib.Path(url).name
    r = requests.get(url, allow_redirects=True, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    save_path = pathlib.Path(save_dir) / name

    if not save_path.exists() or ignore_exist:
        with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True) as bar:
            with open(save_path, "wb") as f:
                for data in r.iter_content(block_size):
                    bar.update(len(data))
                    f.write(data)

            if total_size != 0 and bar.n != total_size:
                raise ValueError("Download size mismatch")

    print(dict(download=dict(url=url, save_path=save_path)))
    return save_path


def unzip(path, save_dir=".", ignore_exist=False):
    path = pathlib.Path(path)
    save_dir = pathlib.Path(save_dir)
    assert path.suffix == ".zip"
    with zipfile.ZipFile(path) as f:
        top_members = [pathlib.Path(p).parts[0] for p in f.namelist()]
        top_members = sorted(set(top_members))
        top_members = [save_dir / m for m in top_members]
        is_exist = all([m.exists() for m in top_members])

        if not is_exist or ignore_exist:
            f.extractall(save_dir)
    print(dict(unzip=dict(path=path, top_members=top_members)))
    return top_members


def print_df_sample(df, num_sample=10, random_state=42):
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.sample(num_sample, random_state=random_state))


class OmniglotSplits:
    background = "background"
    evaluation = "evaluation"


def get_metadata(root="temp"):
    pattern = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_{}.zip"
    data = []

    for data_split in [OmniglotSplits.background, OmniglotSplits.evaluation]:
        path = download(pattern.format(data_split), save_dir=root)
        top_members = unzip(path, save_dir=root)
        assert len(top_members) == 1
        image_dir = top_members[0]

        for alphabet_dir in image_dir.iterdir():
            alphabet = alphabet_dir.stem
            for char_dir in alphabet_dir.iterdir():
                char = char_dir.stem
                for path_image in char_dir.iterdir():
                    d = dict(
                        path_image=path_image,
                        alphabet=alphabet,
                        char=char,
                        data_split=data_split,
                    )
                    data.append(d)

    df = pd.DataFrame(data)
    print(dict(dataframe=df.shape))
    print_df_sample(df)
    return df


if __name__ == "__main__":
    get_metadata()
