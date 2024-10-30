import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import copy

class EnhancedPatternLearner(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(EnhancedPatternLearner, self).__init__()
        
        # Calculate sizes for better architecture
        hidden_size = max(512, input_size * 2)
        
        # Deep network with residual connections
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Multiple parallel pathways for better pattern recognition
        self.path1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # Pattern recognition layers
        self.pattern_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # Add batch dimension if not present
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = self.input_layer(x)
        
        # Parallel processing
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        
        # Combine paths
        combined = torch.cat([path1_out, path2_out], dim=-1)
        
        # Final processing
        output = self.pattern_layers(combined)
        
        # Remove batch dimension if it was added
        if output.shape[0] == 1:
            output = output.squeeze(0)
            
        return output

class EnhancedMatrixRL:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_size = None
        self.output_size = None
        self.best_model = None
        self.best_loss = float('inf')
        self.patterns = {}
        
    def _analyze_patterns(self, input_matrix: List[List[int]], output_matrix: List[List[int]]):
        """Analyze and store patterns between input and output"""
        input_arr = np.array(input_matrix)
        output_arr = np.array(output_matrix)
        
        # Store matrix shapes
        self.patterns['input_shape'] = input_arr.shape
        self.patterns['output_shape'] = output_arr.shape
        
        # Calculate scaling factor
        self.patterns['scale_factor'] = output_arr.shape[0] // input_arr.shape[0]
        
        # Store unique values
        self.patterns['unique_values'] = np.unique(output_arr)
        
        # Analyze value positions
        for val in self.patterns['unique_values']:
            self.patterns[f'val_{val}_positions'] = np.where(output_arr == val)
    
    def _prepare_data(self, input_matrix: List[List[int]], output_matrix: List[List[int]]) -> tuple:
        """Enhanced data preparation with pattern information"""
        input_flat = np.array(input_matrix).flatten()
        output_flat = np.array(output_matrix).flatten()
        
        # Add positional encoding
        input_pos = np.array([[i/len(input_flat), j/len(input_flat)] 
                            for i, j in enumerate(range(len(input_flat)))])
        input_data = np.concatenate([input_flat, input_pos.flatten()])
        
        return (torch.FloatTensor(input_data).to(self.device), 
                torch.FloatTensor(output_flat).to(self.device))
    
    def train_single(self, input_matrix: List[List[int]], output_matrix: List[List[int]], 
                    epochs: int = 5000, patience: int = 100, verbose: bool = True) -> None:
        """Enhanced training for single example with early stopping"""
        # Initialize model if needed
        if self.model is None:
            self._analyze_patterns(input_matrix, output_matrix)
            self.input_size = len(np.array(input_matrix).flatten()) + 2 * len(np.array(input_matrix).flatten())
            self.output_size = len(np.array(output_matrix).flatten())
            self.model = EnhancedPatternLearner(self.input_size, self.output_size).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        
        X, y = self._prepare_data(input_matrix, output_matrix)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(X)
            loss = nn.MSELoss()(output, y)
            
            # Add pattern-based loss
            predicted = output.detach().cpu().numpy()
            target = y.cpu().numpy()
            pattern_loss = np.mean(np.abs(predicted.reshape(-1) - target.reshape(-1)))
            total_loss = loss + 0.1 * torch.tensor(pattern_loss).to(self.device)
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Learning rate scheduling
            self.scheduler.step(total_loss)
            
            # Early stopping check
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                self.best_model = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
                
                # Show current prediction
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(X).cpu().numpy().reshape(len(output_matrix), -1)
                    pred_rounded = np.round(pred)
                    accuracy = np.mean(pred_rounded == np.array(output_matrix))
                    print(f"Current Accuracy: {accuracy*100:.2f}%")
        
        # Load best model
        self.model.load_state_dict(self.best_model)
    
    def train_multiple(self, input_matrices: List[List[List[int]]], 
                      output_matrices: List[List[List[int]]], 
                      epochs_per_example: int = 5000) -> None:
        """Train on multiple examples with enhanced learning"""
        print("Starting enhanced training on multiple examples...")
        
        for i, (input_matrix, output_matrix) in enumerate(zip(input_matrices, output_matrices)):
            print(f"\nTraining on example {i+1}:")
            self.train_single(input_matrix, output_matrix, epochs_per_example)
            
            # Verify accuracy after training
            pred = self.predict(input_matrix)
            accuracy = np.mean(np.array(pred) == np.array(output_matrix))
            print(f"Example {i+1} Final Accuracy: {accuracy*100:.2f}%")
    
    def predict(self, input_matrix: List[List[int]]) -> List[List[int]]:
        """Enhanced prediction with pattern matching"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
        
        # Prepare input
        X, _ = self._prepare_data(input_matrix, [[0] * self.output_size])
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            
        # Reshape and round to nearest valid value
        output_size = int(np.sqrt(self.output_size))
        predicted = output.cpu().numpy().reshape(output_size, output_size)
        
        # Enhanced rounding with pattern matching
        unique_values = self.patterns['unique_values']
        rounded = np.zeros_like(predicted)
        for i in range(predicted.shape[0]):
            for j in range(predicted.shape[1]):
                # Find closest valid value
                rounded[i, j] = unique_values[np.argmin(np.abs(predicted[i, j] - unique_values))]
        
        return rounded.tolist()

# Example usage
def test_enhanced_rl():
    # Initialize model
    rl = EnhancedMatrixRL()
    
    # Training example
    train_data = {
        "input": [[0, 7, 7], [7, 7, 7], [0, 7, 7]],
        "output": [
            [0, 0, 0, 0, 7, 7, 0, 7, 7],
            [0, 0, 0, 7, 7, 7, 7, 7, 7],
            [0, 0, 0, 0, 7, 7, 0, 7, 7],
            [0, 7, 7, 0, 7, 7, 0, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [0, 7, 7, 0, 7, 7, 0, 7, 7],
            [0, 0, 0, 0, 7, 7, 0, 7, 7],
            [0, 0, 0, 7, 7, 7, 7, 7, 7],
            [0, 0, 0, 0, 7, 7, 0, 7, 7]
        ]
    }
    
    # Train
    rl.train_single(train_data["input"], train_data["output"])
    
    # Test
    test_input = [[7, 0, 7], [7, 0, 7], [7, 7, 0]]
    predicted = rl.predict(test_input)
    
    print("\nTest Input:")
    for row in test_input:
        print(row)
    
    print("\nPredicted Output:")
    for row in predicted:
        print(row)

if __name__ == "__main__":
    test_enhanced_rl()