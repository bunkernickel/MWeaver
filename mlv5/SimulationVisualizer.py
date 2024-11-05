import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
import time
from collections import defaultdict

class SimulationVisualizer:
    def __init__(self, agents, social_network, embedding_loader, semantic_foundation, update_interval=0.5):
        # CUDA setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        else:
            print("CUDA not available, using CPU")
            
        self.agents = agents
        self.G = social_network
        self.embedding_loader = embedding_loader
        self.update_interval = update_interval
        self.embedding_dim = embedding_loader.embedding_dim
        
        # Community tracking
        self.community_colors = {}
        self.previous_communities = None
        self.community_history = []
        
        # Setup tracking metrics
        self.sentiment_history = defaultdict(list)
        self.embedding_history = []
        
        # Initialize plots
        plt.ion()
        self.setup_plots()

        self.semantic_foundation = semantic_foundation
        
        # Add tracking for semantic orientations
        self.orientation_history = defaultdict(list)
        
    def update_semantic_tracking(self):
        """Track changes in agent orientations"""
        for agent in self.agents:
            for axis, value in agent.orientation.orientation.items():
                self.orientation_history[f"Agent{agent.agent_id}_{axis}"].append(value)
                
    def update(self, step):
        """Main update method called during simulation"""
        self.update_sentiment_tracking()
        self.update_semantic_tracking()
        if step % 5 == 0:
            self.update_embedding_space()
            self.update_community_detection()
        self.draw_plots()
        plt.pause(self.update_interval)
        
    def update_sentiment_tracking(self):
        """Track the evolution of word strengths across agents"""
        for word in self.embedding_loader.word_to_index.keys():
            strengths = [agent.get_embedding_strength(word) for agent in self.agents]
            valid_strengths = [s for s in strengths if not np.isnan(s)]
            if valid_strengths:
                avg_strength = np.mean(valid_strengths)
                self.sentiment_history[word].append(avg_strength)
            else:
                self.sentiment_history[word].append(0.0)
                
    def update_embedding_space(self):
        """Project agent embeddings to 2D space using t-SNE"""
        embeddings = []
        for agent in self.agents:
            if agent.timeline:
                recent_messages = agent.timeline[-5:]
                message_vectors = [msg['vector'].cpu().numpy() for msg in recent_messages]
                if message_vectors:
                    avg_embedding = np.mean(message_vectors, axis=0)
                else:
                    avg_embedding = np.random.normal(0, 0.1, self.embedding_dim)
            else:
                avg_embedding = np.random.normal(0, 0.1, self.embedding_dim)
            embeddings.append(avg_embedding)
            
        if embeddings:
            embeddings_array = np.array(embeddings)
            if np.var(embeddings_array) < 1e-10:
                embeddings_array += np.random.normal(0, 0.1, embeddings_array.shape)
            
            try:
                tsne = TSNE(n_components=2, 
                           perplexity=min(30, len(self.agents)-1),
                           init='pca', 
                           random_state=42)
                self.embedding_positions = tsne.fit_transform(embeddings_array)
                self.embedding_history.append(self.embedding_positions)
            except Exception as e:
                print(f"t-SNE failed: {e}")
                self.embedding_positions = np.random.normal(0, 1, (len(self.agents), 2))
                
    def setup_plots(self):
        """Initialize the matplotlib figure and subplots"""
        self.fig = plt.figure(figsize=(15, 10))
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
    def match_communities(self, current_communities, previous_communities):
        """Match current communities with previous ones based on maximum overlap"""
        if not previous_communities:
            return current_communities
            
        matched_communities = []
        used_previous = set()
        
        for current in current_communities:
            max_overlap = 0
            best_match = None
            
            for i, previous in enumerate(previous_communities):
                if i in used_previous:
                    continue
                    
                overlap = len(current.intersection(previous))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = i
                    
            if best_match is not None:
                used_previous.add(best_match)
                matched_communities.append(current)
            else:
                matched_communities.append(current)
                
        return matched_communities
        
    def update_community_detection(self):
        """Detect communities and maintain consistent coloring"""
        try:
            G_undirected = self.G.to_undirected()
            current_communities = list(nx.community.louvain_communities(
                G_undirected, 
                resolution=1.0,
                seed=42
            ))
            
            if self.previous_communities:
                current_communities = self.match_communities(
                    current_communities,
                    self.previous_communities
                )
            
            for i, community in enumerate(current_communities):
                if frozenset(community) not in self.community_colors:
                    self.community_colors[frozenset(community)] = plt.cm.tab20(i % 20)
            
            self.previous_communities = current_communities
            metrics = self.calculate_community_metrics(current_communities)
            self.community_history.append((current_communities, metrics))
            
        except Exception as e:
            print(f"Community detection failed: {e}")
            if self.previous_communities:
                self.community_history.append((self.previous_communities, {}))
            else:
                self.community_history.append(([set(self.G.nodes())], {}))
                
    def calculate_community_metrics(self, communities):
        """Calculate various metrics for communities"""
        metrics = {
            'sizes': [len(c) for c in communities],
            'modularity': nx.community.modularity(
                self.G.to_undirected(),
                communities
            ),
            'internal_density': [],
            'external_density': []
        }
        
        for comm in communities:
            subgraph = self.G.subgraph(comm)
            internal_edges = subgraph.number_of_edges()
            possible_internal = len(comm) * (len(comm) - 1)
            
            if possible_internal > 0:
                internal_density = internal_edges / possible_internal
            else:
                internal_density = 0
                
            metrics['internal_density'].append(internal_density)
            
        return metrics
        
    def draw_plots(self):
        """Draw all visualization components"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
            
        # 1. Network Graph with Communities
        pos = nx.spring_layout(self.G, k=1/np.sqrt(len(self.G.nodes())), seed=42)
        
        if self.community_history:
            communities, metrics = self.community_history[-1]
            
            # Draw edges first
            nx.draw_networkx_edges(self.G, pos, alpha=0.2, ax=self.ax1)
            
            # Draw nodes colored by community
            for comm in communities:
                color = self.community_colors[frozenset(comm)]
                nx.draw_networkx_nodes(
                    self.G, pos,
                    nodelist=list(comm),
                    node_color=[color],
                    node_size=100,
                    ax=self.ax1
                )
                
            # Add labels
            nx.draw_networkx_labels(self.G, pos, font_size=8, ax=self.ax1)
            
        self.ax1.set_title("Social Network Communities")
        
        # 2. Sentiment Evolution
        for word, strengths in self.sentiment_history.items():
            if len(strengths) > 1:
                self.ax2.plot(strengths, label=word, alpha=0.7)
        self.ax2.set_title("Word Strength Evolution")
        if len(self.sentiment_history) > 0:
            self.ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          fontsize='small', ncol=2)
        
        # 3. Embedding Space
        if hasattr(self, 'embedding_positions'):
            scatter = self.ax3.scatter(
                self.embedding_positions[:, 0],
                self.embedding_positions[:, 1],
                c=range(len(self.agents)),
                cmap='viridis'
            )
            self.ax3.set_title("Agent Embedding Space")
            
        # 4. Community Metrics
        if self.community_history:
            _, metrics = self.community_history[-1]
            
            # Plot community sizes
            sizes = metrics['sizes']
            self.ax4.bar(range(len(sizes)), sizes, alpha=0.7)
            self.ax4.set_title(f"Community Sizes\nModularity: {metrics['modularity']:.3f}")
            self.ax4.set_xlabel("Community ID")
            self.ax4.set_ylabel("Size")
            
        plt.tight_layout()

        self.ax4.clear()
        for key, values in self.orientation_history.items():
            if len(values) > 1:  # Only plot if we have history
                agent_id = key.split('_')[0]
                axis = key.split('_')[1]
                self.ax4.plot(values, label=f"{agent_id}-{axis}", alpha=0.5)
        
        self.ax4.set_title("Semantic Orientation Evolution")
        self.ax4.set_xlabel("Time")
        self.ax4.set_ylabel("Orientation Value")
        if len(self.orientation_history) > 0:
            self.ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          fontsize='small', ncol=2)