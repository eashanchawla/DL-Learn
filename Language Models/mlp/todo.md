## Refined To-Do List

1. **Understand the Language Modeling Problem**
   - Recognize the task: given a context of previous characters, predict the next character.
   - Acknowledge the challenge of exponential growth in context (the more characters you look back at, the more possible sequences).

2. **Prepare the Dataset**
   - Choose a `block_size` (e.g., 3) for how many previous characters are used to predict the next one.
   - Convert each word (or name) into a list of character indices.
   - Create input (X) and output (Y) pairs, where X is the context of length `block_size` and Y is the next character’s index.
   - Split the overall dataset into **train**, **validation**, and **test** sets (e.g., 80/10/10).

3. **Create the Embedding Layer**
   - Initialize a lookup table (matrix) `C` of size `[vocab_size, embed_dim]`.
   - Each row in `C` corresponds to the embedding for one character in the vocabulary.
   - Understand that indexing into `C` with a character ID retrieves that character’s embedding.

4. **Define the MLP Architecture**
   - **Input**: Flattened embeddings for `block_size` characters (e.g., if `block_size=3` and `embed_dim=10`, input size is 30).
   - **Hidden Layer**: Choose the size (e.g., 100, 200, 300). Apply a nonlinearity (e.g., `tanh`).
   - **Output Layer**: Has `vocab_size` units (e.g., 27 for 26 letters + special “dot”).
   - Output logits will be used for classification (one logit per character).

5. **Loss Function**
   - Use **cross-entropy loss** (e.g., `F.cross_entropy(logits, targets)` in PyTorch).
   - This implicitly applies a softmax to the logits and compares with the target character index.

6. **Train the Model (Mini-Batch Gradient Descent)**
   - Shuffle your training data and create mini-batches.
   - For each mini-batch:
     1. Embed the batch of input tokens.
     2. Flatten or concatenate embeddings to form MLP inputs.
     3. Compute logits via forward pass.
     4. Compute cross-entropy loss.
     5. Zero out gradients: `optimizer.zero_grad()`.
     6. Backpropagate: `loss.backward()`.
     7. Update parameters: `optimizer.step()`.
   - Monitor the training loss and validation loss to diagnose overfitting or underfitting.

7. **Hyperparameter Experiments**
   - Vary the embedding dimension, hidden layer size, and `block_size`.
   - Adjust learning rate, batch size, and training iterations.
   - Optionally apply learning rate scheduling (decay over time or reduce when loss plateaus).

8. **Evaluate Performance**
   - Regularly compute the **validation loss** to guide hyperparameter choices.
   - Check **test loss** only after finalizing all hyperparameters, to avoid overfitting on the validation set.

9. **Sampling from the Model**
   - Start with a context of `block_size` “dot” tokens (or any special start token).
   - Repeatedly:
     1. Embed the current context.
     2. Compute logits and convert to probabilities (via softmax).
     3. Sample the next character index from the probabilities.
     4. Append the sampled character index to the context (sliding window or appended sequence).
   - Convert indices to characters and observe the generated “names” or sequences.

10. **Optional Visualizations & Extra Steps**
    - Plot training/validation loss curves over iterations.
    - If using 2D embeddings, visualize them to see clusters (e.g., vowels vs. consonants).
    - Explore more complex model variants or additional tricks from Bengio et al. (2003) or other papers.
