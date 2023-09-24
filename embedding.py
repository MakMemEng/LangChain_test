import numpy as np
import gensim

# モデルの読み込み
model_path = "entity_vector/entity_vector.model.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


# コサイン類似度
def cos_similarity(a, b):
    return np.dot(a, b) / ((np.sqrt(np.dot(a, a))) * (np.sqrt(np.dot(b, b))))


print(model["おはよう"])
print(model["こんばんは"])

print(cos_similarity(model["おはよう"], model["こんばんは"]))
print(cos_similarity(model["おはよう"], model["ひじき"]))
