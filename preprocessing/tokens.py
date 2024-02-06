import re
from typing import List, Union

class Tokenizer():
    def __init__(self, docs: List[str]=[], init_map: dict={}, whitespace: bool=False):
        self.whitespace = whitespace
        self.docs = docs
        self.map = init_map
        if docs and not init_map:
            self._map(docs)

    def _tokenize_doc(self, doc: str):
        t = [i.lower() for i in re.split(r"([?_!,.()\'\"]|--|\s)", doc)]
        t = [i for i in t if i] if self.whitespace else [i.strip() for i in t if i.strip()]
        return t

    def _map(self, docs: Union[List[str], str]):
        docs = [docs] if isinstance(docs, str) else docs
        tkn_list = self._tokenize_doc(" ".join(docs))
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
