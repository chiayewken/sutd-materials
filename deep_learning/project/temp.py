from typing import List

import spacy
import spacy.lang.en
import spacy.tokens


class SpacyTokenizer:
    def __init__(self):
        # Requirement to run in console first: "python -m spacy download en_core_web_sm"
        self.nlp: spacy.lang.en.English = spacy.load("en_core_web_sm")

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        out = []
        doc: spacy.tokens.Doc
        for doc in self.nlp.pipe(texts, disable=["tagger", "parser", "ner", "textcat"]):
            token: spacy.tokens.Token
            out.append([token.text for token in doc])
        return out


def main():
    texts = ["This, is a sentence with words!", "This is another one..."]
    tokenizer = SpacyTokenizer()
    print(tokenizer.tokenize(texts))


if __name__ == "__main__":
    main()
