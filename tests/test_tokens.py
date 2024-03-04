from preprocessing.tokens import Tokenizer

test1 = "Hello, world."
test2 = "World-- burning."

def test_tokenizer_empty():
	op = Tokenizer()
	encoded1 = op.encode(test1)
	assert encoded1 == [2,0,3,1]
	encoded2 = op.encode(test2)
	assert encoded2 == [3,4,5,1]

def test_decode():
	op = Tokenizer([test1, test2])
	assert op.decode([4,0,3,5,2]) == "hello, burning world."
