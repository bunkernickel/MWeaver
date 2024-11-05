import torch

class EmbeddingLoader:
    def __init__(self, embedding_path, embedding_dim, device):
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.word_to_index = {}
        self.index_to_word = {}
        self.embeddings = []
        self.device = device

    def load_embeddings(self, vocabulary):
        print("Loading embeddings...")
        vocab_set = set(vocabulary)
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                values = line.strip().split()
                if len(values) != self.embedding_dim + 1:
                    continue  # Skip invalid lines
                word = values[0]
                if word in vocab_set:
                    vector = torch.tensor([float(x) for x in values[1:]], device=self.device)
                    self.word_to_index[word] = len(self.embeddings)
                    self.index_to_word[len(self.embeddings)] = word
                    self.embeddings.append(vector)
        self.embeddings = torch.stack(self.embeddings)
        print(f"Loaded {len(self.embeddings)} word vectors.")

    def get_embedding(self, word):
        idx = self.word_to_index.get(word)
        if idx is not None:
            return self.embeddings[idx]
        else:
            return torch.zeros(self.embedding_dim, device=self.device)

    def get_index(self, word):
        return self.word_to_index.get(word, -1)