from typing import List, Tuple, Dict
import numpy as np
from collections import defaultdict

class MatrixPatternSolver:
    def __init__(self):
        self.training_data = []
        self.patterns = {}
        
    def add_training_example(self, input_matrix: List[List[int]], output_matrix: List[List[int]]):
        """Add a training example to learn from."""
        self.training_data.append((input_matrix, output_matrix))
        
    def analyze_patterns(self):
        """Analyze patterns in the training data."""
        if not self.training_data:
            return
        
        # Get dimensions
        input_shape = np.array(self.training_data[0][0]).shape
        output_shape = np.array(self.training_data[0][1]).shape
        
        # Convert to numpy arrays for easier manipulation
        for input_matrix, output_matrix in self.training_data:
            input_arr = np.array(input_matrix)
            output_arr = np.array(output_matrix)
            
            # Analyze block patterns
            self._analyze_block_patterns(input_arr, output_arr)
            
            # Analyze repetition patterns
            self._analyze_repetition_patterns(input_arr, output_arr)
            
    def _analyze_block_patterns(self, input_arr: np.ndarray, output_arr: np.ndarray):
        """Analyze how input blocks might be repeated or transformed in output."""
        input_height, input_width = input_arr.shape
        output_height, output_width = output_arr.shape
        
        # Calculate scaling factors
        height_scale = output_height // input_height
        width_scale = output_width // input_width
        
        self.patterns['scaling'] = (height_scale, width_scale)
        
    def _analyze_repetition_patterns(self, input_arr: np.ndarray, output_arr: np.ndarray):
        """Analyze how values are repeated in the output matrix."""
        # Look for repetition patterns in rows and columns
        self.patterns['row_repeat'] = self._find_row_repetition(output_arr)
        self.patterns['col_repeat'] = self._find_col_repetition(output_arr)
        
    def _find_row_repetition(self, arr: np.ndarray) -> Dict:
        """Find patterns of row repetition."""
        patterns = {}
        rows = arr.shape[0]
        
        for i in range(rows):
            for j in range(i + 1, rows):
                if np.array_equal(arr[i], arr[j]):
                    if i not in patterns:
                        patterns[i] = []
                    patterns[i].append(j)
                    
        return patterns
    
    def _find_col_repetition(self, arr: np.ndarray) -> Dict:
        """Find patterns of column repetition."""
        return self._find_row_repetition(arr.T)
    
    def generate_output(self, input_matrix: List[List[int]]) -> List[List[int]]:
        """Generate output matrix based on learned patterns."""
        input_arr = np.array(input_matrix)
        input_height, input_width = input_arr.shape
        
        if not self.patterns:
            self.analyze_patterns()
            
        # Get scaling factors from patterns
        height_scale, width_scale = self.patterns.get('scaling', (3, 3))  # Default to 3x3 if not found
        
        # Initialize output matrix
        output_height = input_height * height_scale
        output_width = input_width * width_scale
        output = np.zeros((output_height, output_width), dtype=int)
        
        # Apply the basic block pattern
        for i in range(output_height):
            for j in range(output_width):
                # Map output position back to input position
                input_i = i // height_scale
                input_j = j // width_scale
                
                # Copy value from input
                output[i][j] = input_arr[input_i][input_j]
                
        # Apply repetition patterns if found
        if 'row_repeat' in self.patterns:
            for source_row, repeat_rows in self.patterns['row_repeat'].items():
                for target_row in repeat_rows:
                    if target_row < output_height:
                        output[target_row] = output[source_row]
                        
        # Return as list of lists
        return output.tolist()

def test_pattern_solver():
    """Test the pattern solver with provided examples."""
    # Initialize solver
    solver = MatrixPatternSolver()
    
    # Add training examples
    training_data = [
        {
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
        },
        {
            "input": [[4, 0, 4], [0, 0, 0], [0, 4, 0]],
            "output": [
                [4, 0, 4, 0, 0, 0, 4, 0, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 4, 0, 0, 0, 0, 0, 4, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 0, 0, 0, 0]
            ]
        }
    ]
    
    for example in training_data:
        solver.add_training_example(example["input"], example["output"])
    
    # Test case
    test_input = [[7, 0, 7], [7, 0, 7], [7, 7, 0]]
    
    # Generate output
    output = solver.generate_output(test_input)
    
    print("Test Input:")
    for row in test_input:
        print(row)
    
    print("\nGenerated Output:")
    for row in output:
        print(row)
        
    return output

# Run the test
test_pattern_solver()