# Cultural Headwinds Simulation with Social Network

## Overview

This simulation models agents with different cultural headwinds interacting in a social network. Agents post and consume content, and their interactions influence their cultural embeddings. The simulation aims to explore how ideas circulate and how social structures emerge from individual interactions.

## File Structure

- `agent.py`: Contains the `Agent` class with methods for posting and receiving messages.
- `cultural_headwinds.py`: Defines cultural biases for agents.
- `embeddings.py`: Loads pre-trained word embeddings.
- `main.py`: Main script to run the simulation.
- `utils.py`: Utility functions for initializing agents and the social network.
- `config.py`: Configuration parameters.
- `README.md`: Instructions and information.
- `requirements.txt`: List of required Python packages.
- `data/`: Directory to store pre-trained embeddings (e.g., GloVe).

## Requirements

- Python 3.6 or higher
- Libraries:
  - torch
  - numpy
  - matplotlib
  - networkx
  - tqdm

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/simulation.git
cd simulation/
