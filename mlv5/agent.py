import torch
import random
import numpy as np
from collections import Counter, defaultdict
from semantic_foundations import AgentOrientation

class Agent:
    def __init__(self, agent_id, embedding_loader, semantic_foundation, device,
                 decay_rate=0.01, reinforcement_rate=0.05,
                 innovation_rate=0.2, conformity_bias=0.7):
        # Numerical stability parameters
        self.min_pref = 1e-5
        self.max_pref = 1e5

        self.agent_id = agent_id
        self.embedding_loader = embedding_loader
        self.semantic_foundation = semantic_foundation
        self.device = device

        # Initialize vocabulary and mappings
        self.vocabulary = list(embedding_loader.word_to_index.keys())
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocabulary)}

        # Initialize agent's semantic orientation
        seed_words = random.sample(self.vocabulary, 3)
        self.orientation = AgentOrientation(semantic_foundation, seed_words)
        
        # Personality parameters
        self.innovation_rate = innovation_rate
        self.conformity_bias = conformity_bias
        
        # Learning rates
        self.decay_rate = decay_rate
        self.reinforcement_rate = reinforcement_rate
        
        # Initialize preferences and embeddings
        self.word_preferences = self._initialize_word_preferences()
        self.embeddings = self._initialize_embeddings()
        
        # History tracking
        self.used_words = set()
        self.timeline = []
        self.community_vocab_usage = defaultdict(float)

    def _initialize_word_preferences(self):
        """Initialize word preferences based on semantic orientation"""
        preferences = torch.ones(len(self.vocabulary), device=self.device)
        
        for idx, word in self.index_to_word.items():
            virtue_score = self.orientation.evaluate_word(word)
            virtue_score = np.clip(virtue_score, -2.0, 2.0)
            preferences[idx] *= (1.0 + virtue_score)
            
        preferences += torch.randn_like(preferences) * 0.1
        preferences.clamp_(min=self.min_pref, max=self.max_pref)
        return preferences

    def _initialize_embeddings(self):
        """Initialize embeddings with semantic influence"""
        embeddings_list = []
        for word in self.vocabulary:
            base_embedding = self.embedding_loader.get_embedding(word)
            virtue_score = self.orientation.evaluate_word(word)
            virtue_score = np.clip(virtue_score, -2.0, 2.0)
            scaling_factor = 1.5 if virtue_score > 0.2 else (0.5 if virtue_score < -0.2 else 1.0)
            transformed_embedding = base_embedding * scaling_factor
            embeddings_list.append(transformed_embedding)
        return torch.stack(embeddings_list)

    def _safe_softmax(self, tensor):
        """Compute softmax with numerical stability"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return torch.ones_like(tensor) / tensor.size(0)
            
        tensor = tensor - tensor.max()
        tensor = torch.clamp(tensor, min=-20, max=20)
        exp = torch.exp(tensor)
        sum_exp = exp.sum() + 1e-10
        
        if sum_exp == 0 or torch.isnan(sum_exp) or torch.isinf(sum_exp):
            return torch.ones_like(tensor) / tensor.size(0)
            
        return exp / sum_exp

    def get_community_influence(self, recent_messages, num_samples=5):
        """Calculate community vocabulary preferences"""
        community_prefs = defaultdict(float)
        
        if not recent_messages:
            return community_prefs
            
        for idx, msg in enumerate(recent_messages):
            time_weight = np.exp(-0.1 * (len(recent_messages) - idx))
            for word in msg['content']:
                semantic_weight = 1.0 + np.clip(self.orientation.evaluate_word(word), -2.0, 2.0)
                community_prefs[word] += time_weight * semantic_weight
                
        total = sum(community_prefs.values()) or 1
        return {word: min(count/total, self.max_pref) for word, count in community_prefs.items()}

    def send_message(self, message_id_counter):
        """Generate and send a message"""
        recent_messages = self.timeline[-10:] if self.timeline else []
        community_prefs = self.get_community_influence(recent_messages)
        
        message_words = []
        message_length = random.randint(3, 7)
        
        for _ in range(message_length):
            if random.random() < self.innovation_rate:
                probs = self._safe_softmax(self.word_preferences)
                word_idx = torch.multinomial(probs, 1).item()
                word = self.index_to_word[word_idx]
            else:
                combined_prefs = {}
                for word in self.vocabulary:
                    personal_pref = self.word_preferences[self.word_to_index[word]].item()
                    community_pref = community_prefs.get(word, 0)
                    semantic_pref = 1.0 + np.clip(self.orientation.evaluate_word(word), -2.0, 2.0)
                    
                    combined_prefs[word] = np.clip(
                        (self.conformity_bias * community_pref +
                        (1 - self.conformity_bias) * personal_pref) * semantic_pref,
                        self.min_pref, self.max_pref
                    )
                
                word = max(combined_prefs.items(), key=lambda x: x[1] + random.gauss(0, 0.1))[0]
            
            message_words.append(word)

        indices = [self.word_to_index[word] for word in message_words]
        message_vector = self.embeddings[indices].mean(dim=0)
        
        return {
            'id': message_id_counter,
            'sender': self.agent_id,
            'content': message_words,
            'vector': message_vector,
            'semantic_profile': {word: self.orientation.evaluate_word(word) 
                               for word in message_words}
        }

    def receive_messages(self, messages):
        """Process received messages"""
        self.timeline.extend(messages)
        for message in messages:
            self.process_message(message)

    def process_message(self, message):
        """Process a single message"""
        words = message['content']
        self.update_embeddings(words)
        
        semantic_impact = sum(np.clip(self.orientation.evaluate_word(word), -2.0, 2.0) 
                            for word in words)
        if abs(semantic_impact) > 0.5:
            self.orientation.update_orientation(words, learning_rate=0.05)

    def update_embeddings(self, used_words):
        """Update embeddings and preferences"""
        # Decay all preferences
        self.word_preferences *= (1 - self.decay_rate)
        
        # Update preferences for used words
        for word in used_words:
            if word in self.word_to_index:
                idx = self.word_to_index[word]
                semantic_factor = 1.0 + np.clip(self.orientation.evaluate_word(word), -2.0, 2.0)
                update = np.clip(self.reinforcement_rate * semantic_factor, -1.0, 1.0)
                
                # Create a scalar tensor for the update
                update_tensor = torch.tensor(update, device=self.device)
                self.word_preferences[idx] += update_tensor
                
                # Add noise as a scalar
                noise = torch.randn(1, device=self.device).item() * 0.05
                self.word_preferences[idx] += noise
        
        # Clamp preferences
        self.word_preferences.clamp_(min=self.min_pref, max=self.max_pref)
        
        # Update embeddings
        indices = [self.word_to_index[word] for word in used_words if word in self.word_to_index]
        if indices:
            semantic_factors = torch.tensor([
                1.0 + np.clip(self.orientation.evaluate_word(self.index_to_word[idx]), -2.0, 2.0)
                for idx in indices
            ], device=self.device)
            
            updates = self.reinforcement_rate * semantic_factors.unsqueeze(1)
            updates = torch.clamp(updates, -1.0, 1.0)
            
            self.embeddings[indices] += updates
            self.used_words.update(used_words)

    def get_embedding_strength(self, word):
        """Get the strength of a word's embedding"""
        idx = self.word_to_index.get(word)
        if idx is not None:
            return self.embeddings[idx].norm().item()
        return 0.0