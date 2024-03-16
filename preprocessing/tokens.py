import re
from typing import List, Union

class Tokenizer():
    def __init__(self, docs: List[str]=[], init_map: dict={}, whitespace: bool=False):
        self.whitespace = whitespace
        self.docs = docs
        self.map = init_map
        self.trained = False
        if docs and not init_map:
            self._map(docs)

    def _tokenize_doc(self, doc: str):
        t = [i.lower() for i in re.split(r"([?_!,.()\'\"]|--|\s)", doc)]
        t = [i for i in t if i] if self.whitespace else [i.strip() for i in t if i.strip()]
        return t

    def _map(self, docs: Union[List[str], str]):
        docs = [docs] if isinstance(docs, str) else docs
        tkn_list = self._tokenize_doc(" ".join(docs))
        if not self.trained:
            tkn_list.extend(["<|endoftext|>", "<|unk|>"])
        self.map = {
            **self.map,
            **{t:(i+len(self.map)) for i,t in enumerate(sorted(list(set(tkn_list).difference(self.map))))},
        }
        self.reverse_map = {v:k for k,v in self.map.items()}
        self.trained = True

    def encode(self, doc: str):
        if set(self._tokenize_doc(doc)).difference(self.map):
            self._map(doc)
        return [self.map[t] for t in self._tokenize_doc(doc)]

    def decode(self, ids):
        text = " ".join([self.reverse_map[i] for i in ids])
        return re.sub(r"\s+([,.?!\"()\"])", r"\1", text)


class SimpleTokenizerV2:
    """
    Requires pre-processed vocab dictionary to be passed
    """
    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
