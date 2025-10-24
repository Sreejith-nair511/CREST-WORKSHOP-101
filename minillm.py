import random
from collections import defaultdict

class SimpleLanguageModel:
    """
    A minimal class demonstrating the core concept of an LLM: 
    predicting the next word (token) based on prior context.

    This model uses a simple statistical approach (a Markov chain of order 1) 
    to simulate the 'learned connections' performed by a Transformer.
    
    Relates to your concepts:
    - Tokenization: Handled by simple word splitting.
    - Transformer: The 'next_word_probabilities' dictionary stores the connections.
    - Generative AI: The 'generate' method produces new text.
    """
    
    def __init__(self):
        # Stores the probability distribution of the next word given the current word.
        # Structure: {'current_word': ['next_word_1', 'next_word_2', ...]}
        self.next_word_probabilities = defaultdict(list)
        self.is_trained = False

    def train(self, text_corpus: str):
        """
        Trains the model by analyzing a body of text to build a 
        next-word statistical map (simulating the Transformer's learned weights).
        """
        # --- Tokenization (Analogous to the Tokenization step in an LLM) ---
        # 1. Clean and split the text into a list of words (tokens).
        # We strip punctuation for simplicity.
        cleaned_text = text_corpus.lower().replace('.', ' ').replace(',', ' ').strip()
        tokens = cleaned_text.split()
        
        if len(tokens) < 2:
            print("Corpus too short to train.")
            return

        # 2. Build the statistical map.
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i+1]
            
            # For every word, record all the words that follow it.
            # The length of the list implicitly represents the probability distribution.
            self.next_word_probabilities[current_token].append(next_token)

        self.is_trained = True
        print(f"Model trained on {len(tokens)} tokens.")

    def _get_next_word(self, current_word: str) -> str | None:
        """
        Predicts the next word based on the available probabilities.
        """
        # Check if the current word has been seen before in the training data
        if current_word in self.next_word_probabilities:
            # Randomly select a next word from the list of observed followers.
            # This selection process simulates the probabilistic nature of LLM output.
            return random.choice(self.next_word_probabilities[current_word])
        else:
            return None

    def generate(self, start_word: str, max_length: int = 10) -> str:
        """
        Generates a sequence of text starting from a seed word 
        (Analogous to the Generative AI step).
        """
        if not self.is_trained:
            return "Error: Model must be trained before generation."
        
        # Start with the initial token
        current_word = start_word.lower()
        generated_text = [current_word]
        
        # Loop to predict and append the next token until max_length is reached
        for _ in range(max_length - 1):
            next_word = self._get_next_word(current_word)
            
            if next_word:
                generated_text.append(next_word)
                current_word = next_word
            else:
                # Stop if no possible next word is found
                break
                
        # Simple capitalization for a better look
        final_text = " ".join(generated_text)
        return final_text.capitalize() + "."

# --- Runnable Example ---
if __name__ == "__main__":
    # 1. Define the training data (The "Input" for the LLM)
    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "The dog is lazy and sleeps a lot. "
        "A brown fox is quick."
    )
    
    # 2. Initialize the model
    model = SimpleLanguageModel()
    
    # 3. Train the model (Tokenization and Transformer simulation)
    model.train(corpus)
    
    # 4. Generate new content (Generative AI simulation)
    print("\n--- Generation Examples ---")
    
    # Example 1: Starting with 'The'
    seed_1 = "The"
    generated_1 = model.generate(seed_1, max_length=6)
    print(f"Seed: '{seed_1}' -> Output: {generated_1}")
    
    # Example 2: Starting with 'lazy'
    seed_2 = "lazy"
    generated_2 = model.generate(seed_2, max_length=5)
    print(f"Seed: '{seed_2}' -> Output: {generated_2}")

    # Example 3: Starting with a word not in the corpus
    seed_3 = "sunshine"
    generated_3 = model.generate(seed_3, max_length=5)
    print(f"Seed: '{seed_3}' -> Output: {generated_3}")
