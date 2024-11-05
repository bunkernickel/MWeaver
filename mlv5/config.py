# config.py

embedding_dim = 50  # Dimensionality of the embeddings (e.g., 50, 100, 200, 300)
embedding_path = 'data/glove.6B.50d.txt'

num_agents = 7      # Number of agents in the simulation
num_steps = 500         # Number of time steps in the simulation
post_probability = 0.05  # Probability that an agent will post at each time step

decay_rate = 0.01
reinforcement_rate = 0.05

# Vocabulary of words (ensure these words are present in the embeddings)
vocabulary = [
    'love', 'hate', 'peace', 'war', 'happy', 'sad', 'music', 'silence', 'life', 'death',
    'freedom', 'slavery', 'truth', 'lie', 'light', 'dark', 'good', 'evil', 'friend', 'enemy',
    'hope', 'fear', 'joy', 'anger', 'dream', 'nightmare', 'justice', 'crime', 'beauty', 'ugly',
    # Add more words as needed
]
