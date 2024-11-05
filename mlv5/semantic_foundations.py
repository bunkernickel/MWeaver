import torch
import numpy as np
from sklearn.decomposition import PCA

class SemanticFoundation:
    def __init__(self, embedding_loader, num_foundation_dimensions=3, device='cuda'):
        self.device = device
        self.embedding_loader = embedding_loader
        self.embedding_dim = embedding_loader.embedding_dim
        
        # Create foundation axes through semantic space
        self.foundation_axes = self._initialize_foundation_axes(num_foundation_dimensions)
        
        # Get all embeddings as tensor for vectorized operations
        self.all_embeddings = self._stack_embeddings()
        
        # Cache normalized directions for efficiency
        self.normalized_axes = {
            axis: direction / torch.norm(direction)
            for axis, direction in self.foundation_axes.items()
        }
        
    def _initialize_foundation_axes(self, num_dimensions):
        """Initialize foundational semantic axes"""
        # Start with basic oppositions in embedding space
        foundation_pairs = {
            'moral': ('good', 'evil'),
            'truth': ('true', 'false'), 
            'worth': ('valuable', 'worthless'),
            'affect': ('love', 'hate'),
            'power': ('strong', 'weak')
        }
        
        axes = {}
        for name, (pos, neg) in foundation_pairs.items():
            try:
                pos_embed = self.embedding_loader.get_embedding(pos)
                neg_embed = self.embedding_loader.get_embedding(neg)
                # Create axis from difference between embeddings
                axis = pos_embed - neg_embed
                axes[name] = axis
            except:
                print(f"Could not create axis for {name}")
                
        # Use PCA to find additional major axes of variation
        embeddings_matrix = self._stack_embeddings().cpu().numpy()
        pca = PCA(n_components=num_dimensions)
        pca.fit(embeddings_matrix)
        
        # Add PCA components as additional axes
        for i, component in enumerate(pca.components_):
            axes[f'semantic_{i}'] = torch.tensor(component, device=self.device)
            
        return axes
        
    def _stack_embeddings(self):
        """Stack all embeddings into a single tensor"""
        embeddings = []
        self.word_to_idx = {}
        for idx, word in enumerate(self.embedding_loader.word_to_index.keys()):
            embedding = self.embedding_loader.get_embedding(word)
            embeddings.append(embedding)
            self.word_to_idx[word] = idx
        return torch.stack(embeddings)
        
    def get_word_orientation(self, word):
        """Get word's orientation along all foundation axes"""
        if word not in self.word_to_idx:
            return None
            
        embedding = self.embedding_loader.get_embedding(word)
        orientations = {}
        
        for axis_name, axis_direction in self.normalized_axes.items():
            # Project word onto axis
            projection = torch.dot(embedding, axis_direction)
            orientations[axis_name] = projection.item()
            
        return orientations
        
    def get_semantic_similarity(self, word1, word2):
        """Get semantic similarity between words"""
        if word1 not in self.word_to_idx or word2 not in self.word_to_idx:
            return 0.0
            
        emb1 = self.embedding_loader.get_embedding(word1)
        emb2 = self.embedding_loader.get_embedding(word2)
        
        return torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        
    def evaluate_virtue(self, word, agent_orientation):
        """Evaluate word's virtue based on agent's orientation"""
        word_orientation = self.get_word_orientation(word)
        if not word_orientation:
            return 0.0
            
        virtue_score = 0.0
        for axis, agent_value in agent_orientation.items():
            if axis in word_orientation:
                # Alignment between agent's values and word's orientation
                alignment = agent_value * word_orientation[axis]
                virtue_score += alignment
                
        return virtue_score
        
class AgentOrientation:
    def __init__(self, semantic_foundation, seed_words=None, random_influence=0.3):
        self.foundation = semantic_foundation
        self.random_influence = random_influence
        
        # Initialize orientation either from seed words or randomly
        self.orientation = self._initialize_orientation(seed_words)
        
    def _initialize_orientation(self, seed_words):
        """Initialize agent's orientation either from seeds or randomly"""
        orientation = {}
        
        if seed_words:
            # Average the orientations of seed words
            orientations = []
            for word in seed_words:
                word_orient = self.foundation.get_word_orientation(word)
                if word_orient:
                    orientations.append(word_orient)
                    
            if orientations:
                # Average orientation along each axis
                for axis in orientations[0].keys():
                    values = [o[axis] for o in orientations]
                    orientation[axis] = np.mean(values)
                    
        # Add random variation
        for axis in self.foundation.foundation_axes.keys():
            if axis not in orientation:
                orientation[axis] = 0.0
            # Add random perturbation
            orientation[axis] += np.random.normal(0, self.random_influence)
            
        return orientation
        
    def evaluate_word(self, word):
        """Evaluate a word based on agent's orientation"""
        return self.foundation.evaluate_virtue(word, self.orientation)
        
    def update_orientation(self, words, learning_rate=0.1):
        """Update orientation based on experienced words"""
        for word in words:
            word_orientation = self.foundation.get_word_orientation(word)
            if word_orientation:
                for axis, value in word_orientation.items():
                    # Move slightly toward or away from word's orientation
                    self.orientation[axis] += learning_rate * value
                    
        # Normalize orientation values
        max_abs = max(abs(v) for v in self.orientation.values())
        if max_abs > 0:
            for axis in self.orientation:
                self.orientation[axis] /= max_abs