# Towards Monosemanticity: Decomposing Language Models With Dictionary Learning
- Antropic, 2023

`The key idea of this paper is to use a SAE to try to understand what each neuron in the MLP layer of a one-layer transformer mean(like, what vague concept does each neuron refer to.).`

The goal of this paper is to make features disentangled. This means that each feature should refer to a single key idea. Also a the neurons that activate when we talk about this feature should be the same.

- A feature is a linear direction in activation space that corresponds to a meaningful concept or property (e.g., "cat," "negation," or "striped texture"). Features are often represented as linear combinations of neurons, meaning they are distributed across multiple neurons rather than being tied to a single one.

- The Monosemanticity paper aims to find and understand the features that a neural network encodes in its activations. These features are conceptualized as linear combinations of neurons in the network, rather than being tied to individual neurons.

- Objective of the Paper:
    - To identify these features in the network's latent space.
    - To analyze whether these features can be made monosemantic (representing one interpretable concept without ambiguity).
    - To study and potentially reduce superposition, where multiple features overlap in the same neurons or directions.

- the superposition hypothesis postulates that neural networks “want to represent more features than they have neurons”
![alt text](image.png)
- 
- This is because many neurons are `polysemantic`: they respond to mixtures of seemingly unrelated inputs. We use the SAE to make them monosemantic. During training of a NN, each feature is learned as a linear combinatoin of neurons(instead of each neuron correspoding to a feature). This makes it hard to understand what each neuron means.

- In Toy Models of Superposition, we described three strategies to finding a sparse and interpretable set of features if they are indeed hidden by superposition: (1) creating models without superposition, perhaps by encouraging activation sparsity; (2) using dictionary learning to find an overcomplete feature basis in a model exhibiting superposition; and (3) hybrid approaches relying on a combination of the two. Since the publication of that work, we've explored all three approaches. We eventually developed counterexamples which persuaded us that the sparse architectural approach (approach 1) was insufficient to prevent polysemanticity, and that standard dictionary learning methods (approach 2) had significant issues with overfitting.


## Monosemanticity
Monosemanticity refers to the property of a component (e.g., a neuron or a feature) in a model having a single, clear, and interpretable meaning or functionality. A monosemantic component responds consistently to one type of input or encodes one specific concept, making it easier to understand and predict its behavior.

- we use a weak dictionary learning algorithm called a sparse autoencoder to generate learned features from a trained model that offer a more monosemantic unit of analysis than the model's neurons themselves

Results
- Sparse Autoencoders extract relatively monosemantic features very well.
- Sparse autoencoders produce relatively universal features.
- Features appear to "split" as we increase autoencoder size.
- 