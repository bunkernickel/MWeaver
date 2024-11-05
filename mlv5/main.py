import torch
import random
import matplotlib.pyplot as plt
from config import (
    embedding_dim, embedding_path, num_agents, num_steps, post_probability,
    decay_rate, reinforcement_rate, vocabulary
)
from SimulationVisualizer import SimulationVisualizer
from utils import initialize_agents, initialize_social_network
from embeddings import EmbeddingLoader
from tqdm import tqdm
from agent import Agent
import networkx as nx
from semantic_foundations import SemanticFoundation, AgentOrientation

def main():
    # Set random seed for reproducibility (optional)
    random.seed(42)
    torch.manual_seed(42)

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load embeddings
    embedding_loader = EmbeddingLoader(embedding_path, embedding_dim, device)
    embedding_loader.load_embeddings(vocabulary)

    semantic_foundation = SemanticFoundation(embedding_loader, num_foundation_dimensions=5, device=device)


    # Initialize agents
    agents = []
    for i in range(num_agents):
        agent = Agent(i, embedding_loader, semantic_foundation, device, 
                     decay_rate=decay_rate, reinforcement_rate=reinforcement_rate)
        agents.append(agent)
    # Initialize social network
    G = initialize_social_network(agents)


    visualizer = SimulationVisualizer(agents, G, embedding_loader, semantic_foundation, update_interval=.5)

    # Simulation loop
    message_id_counter = 0
    message_log = []  # Store all messages sent
    for step in tqdm(range(num_steps), desc="Simulation Steps"):
        messages = []
        # Each agent decides whether to post a message
        for agent_id in G.nodes:
            agent = G.nodes[agent_id]['agent']
            if random.random() < post_probability:
                message_id_counter += 1
                message = agent.send_message(message_id_counter)
                messages.append(message)
                message_log.append(message)
        # Deliver messages to followers
        for message in messages:
            sender_id = message['sender']
            followers = list(G.predecessors(sender_id))  # Agents who follow the sender
            for follower_id in followers:
                follower_agent = G.nodes[follower_id]['agent']
                follower_agent.receive_messages([message])
        visualizer.update(step)

    # Analyze results
    print("Analyzing results...")
    # Example: Compute average word strengths across all agents
    word_strengths = {}
    for agent in agents:
        for word in vocabulary:
            strength = agent.get_embedding_strength(word)
            if word not in word_strengths:
                word_strengths[word] = []
            word_strengths[word].append(strength)

    average_strengths = {word: sum(strengths) / len(strengths) for word, strengths in word_strengths.items()}

    # Plot word strengths
    words = list(average_strengths.keys())
    strengths = [average_strengths[word] for word in words]

    plt.figure(figsize=(12, 6))
    plt.bar(words, strengths)
    plt.xlabel('Words')
    plt.ylabel('Average Word Strength')
    plt.title('Average Word Strengths Across Agents')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Visualize the social network
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos)
    plt.title('Social Network Graph')
    plt.axis('off')
    plt.show()

    # Additional analysis can be added here (e.g., tracking message propagation)

if __name__ == "__main__":
    main()