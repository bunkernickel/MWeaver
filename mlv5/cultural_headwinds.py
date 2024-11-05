import torch

class CulturalHeadwinds:
    def __init__(self, semantic_foundation, agent_orientation):
        self.semantic_foundation = semantic_foundation
        self.agent_orientation = agent_orientation

    def apply(self, word, embedding):
        # Get virtue score from semantic orientation
        virtue_score = self.agent_orientation.evaluate_word(word)
        
        # Scale embedding based on virtue score
        # Positive scores amplify, negative scores suppress
        scaling_factor = 1.5 if virtue_score > 0.2 else (0.5 if virtue_score < -0.2 else 1.0)
        return embedding * scaling_factor