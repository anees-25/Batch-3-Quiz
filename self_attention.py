import torch  # Import PyTorch
import torch.nn as nn  # Import neural network modules
import torch.nn.functional as F  # Import functions like softmax

class SelfAttention(nn.Module):  # Define Self-Attention class

    def __init__(self, embed_size):  # Initialize the class
        super().__init__()  # Initialize parent class

        self.query = nn.Linear(embed_size, embed_size)  # Create Query layer
        self.key = nn.Linear(embed_size, embed_size)  # Create Key layer
        self.value = nn.Linear(embed_size, embed_size)  # Create Value layer

    def forward(self, x):  # Define forward pass

        Q = self.query(x)  # Generate Query vectors
        K = self.key(x)  # Generate Key vectors
        V = self.value(x)  # Generate Value vectors

        scores = Q @ K.transpose(-2, -1)  # Calculate Query-Key similarity scores

        scores = scores / (K.size(-1) ** 0.5)  # Scale attention scores

        attention = F.softmax(scores, dim=-1)  # Convert scores into attention weights

        output = attention @ V  # Multiply attention weights with Value vectors

        return output  # Return final attention output