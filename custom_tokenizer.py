import pandas as pd
from tokenizers.trainers import BpeTrainer

my_vocab_size = 32000
my_limit_alphabet = 6000
my_special_tokens = ["<usr>", "<pad>", "<sys>", "<unk>", "<mask>"]
# tokenizer 에서 사용될 special_tokens이다. 필수 토큰은 위와 같다.
user_defined_symbols = ['<s>','</s>']
# 이제부터는 부가적인 토큰이다. 문장의 시작과 끝을 알리는 토큰을 추가했다.

unused_token_num = 100
unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
# KoELECTRA Github를 참고하여, unused 토큰을 약 200개 추가했다. 범용성을 높일 수 있다.
user_defined_symbols = user_defined_symbols + unused_list
my_special_tokens = my_special_tokens + user_defined_symbols

paths = ['data/train.tsv', 'dat/test.tsv']
# 학습에 사용될 Corpus들을 넣으면 된다. 

tokenizer = BpeTrainer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True, 
    # 만약 cased model이라면 반드시 False로 해야한다, 또한 한글의 경우 cased model로 하면 글자가 자소분리된다.
    lowercase=True,
    # 대소문자 구분 여부를 의미한다. 한글의 경우 무의미하므로 신경쓰지 않아도 된다.
    wordpieces_prefix="##"
)

tokenizer.train(
    files=paths,
    limit_alphabet=my_limit_alphabet,
    vocab_size=my_vocab_size,
    min_frequency=5,
    # pair가 5회이상 등장할시에만 학습
    show_progress=True,
    # 진행과정 출력 여부
    special_tokens=my_special_tokens
)