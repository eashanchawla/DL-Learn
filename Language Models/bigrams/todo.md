# To-Do List for Building a Bigram Language Model and Neural Network Approach

## Part 1: Bigram Language Model Using Counts

1. **Load and Explore the Dataset**
   - Load `names.txt` from the GitHub repository.
   - Read all names into a Python list.
   - Analyze the dataset to determine the total number of names, and find the shortest and longest name lengths.

2. **Create Bigrams**
   - For each name, add special start and end tokens.
   - Generate all consecutive character pairs (bigrams) from each name.

3. **Count Bigram Frequencies**
   - Initialize a dictionary to store bigram counts.
   - Iterate through all bigrams and count their occurrences.

4. **Convert Counts to a 2D Tensor**
   - Create a mapping from each character to a unique integer index.
   - Initialize a 2D PyTorch tensor to store bigram counts based on the character indices.
   - Populate the tensor with the counted bigram frequencies.

5. **Visualize Bigram Counts**
   - Use `matplotlib` to create a visual representation of the bigram counts tensor.
   - Annotate the visualization with corresponding character pairs and their counts.

6. **Implement Sampling from the Bigram Model**
   - Normalize the bigram counts tensor to obtain probability distributions for each preceding character.
   - Write a sampling function that:
     - Starts with the start token.
     - Iteratively samples the next character based on the current character's probability distribution.
     - Stops sampling when the end token is generated.
   - Generate multiple sample names using the sampling function.

7. **Calculate the Loss Function**
   - Define the negative log likelihood loss based on the bigram probabilities and the actual data.
   - Compute the loss for the entire dataset.

8. **Apply Model Smoothing**
   - Add fake counts (e.g., add one to all bigram counts) to smooth the probability distributions.
   - Recompute the normalized probabilities with the smoothed counts.
   - Recalculate the loss with the smoothed model.

## Part 2: Neural Network Approach

1. **Prepare the Training Set**
   - Create input-target pairs from the bigrams where the input is the first character and the target is the second character.
   - Convert these character pairs into integer indices based on the previously defined mapping.
   - Transform the input and target lists into PyTorch tensors.

2. **One-Hot Encode Inputs**
   - Implement one-hot encoding for the input tensor to create binary vectors representing each character.
   - Ensure the one-hot encoded tensor has the correct shape corresponding to the number of classes.

3. **Define the Neural Network**
   - Initialize a single linear layer neural network with appropriate input and output dimensions.
   - Randomly initialize the weights of the neural network.

4. **Implement the Forward Pass**
   - Pass the one-hot encoded inputs through the linear layer to obtain logits.
   - Apply the softmax function to the logits to obtain probability distributions for the next character.

5. **Compute the Loss Function**
   - Calculate the negative log likelihood loss by comparing the predicted probabilities with the actual targets.
   - Ensure the loss is averaged over all training examples.

6. **Backpropagation and Weight Update**
   - Zero out any existing gradients.
   - Perform backpropagation to compute gradients of the loss with respect to the network's weights.
   - Update the weights using gradient descent to minimize the loss.

7. **Iterate Training Over Multiple Epochs**
   - Run multiple iterations of forward pass, loss computation, backpropagation, and weight updates.
   - Monitor the loss to ensure it decreases over time, indicating improving model performance.

8. **Implement Regularization (Optional)**
   - Add a regularization term to the loss function to prevent overfitting by penalizing large weights.
   - Adjust the regularization strength as needed to balance between fitting the data and keeping weights small.

9. **Compare Neural Network with Bigram Model**
   - Ensure that both approaches yield similar performance metrics.
   - Validate that the neural network effectively learns the bigram probabilities through training.

10. **Sampling from the Neural Network Model**
    - Use the trained neural network to generate new names by sampling characters based on the predicted probability distributions.
    - Implement a sampling loop similar to the bigram model but utilizing the neural network for probability predictions.
