from preprocessing.loaders import DatasetV1
from preprocessing.tokens import Tokenizer

raw_text = "Someone at some time did some thing. This had an effect on many other things, at various later times. The long terms consequences of state changes are difficult to predict in complex, evolving systems."
op = Tokenizer()
vocab = sorted(list(set(op._tokenize_doc(raw_text))))
raw_split = op._tokenize_doc(raw_text)


def test_dataset():
    len = 10
    step = 2
    dataset = DatasetV1(raw_text, op, len, step)
    ind = 0
    assert op._tokenize_doc(op.decode(dataset.__getitem__(0)[0].tolist())) == raw_split[step*ind:len+step*ind]
    assert op._tokenize_doc(op.decode(dataset.__getitem__(0)[1].tolist())) == raw_split[step*ind+1:len+step*ind+1]
    ind = 1
    assert op._tokenize_doc(op.decode(dataset.__getitem__(1)[0].tolist())) == raw_split[step*ind:len+step*ind]
    assert op._tokenize_doc(op.decode(dataset.__getitem__(1)[1].tolist())) == raw_split[step*ind+1:len+step*ind+1]
    assert dataset.__len__() == 15
