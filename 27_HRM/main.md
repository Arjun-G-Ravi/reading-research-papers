# Heirarchical Reasoning Models (HRM)
- Sapient Intelligence, Singapore

# What does it do?
- Classic LLMs struggle with reasoning tasks that require multiple steps or complex logic.
- Inspired from human brain
- Got insane benchmark scores from a 27M parameter model with zero pretraining and only a 1000 training samples
![alt text](image.png)
# What is it?
![alt text](image-1.png)
- Language is a tool for human communication, not the substrate of thought itself . You dont have to use language to think, like reasoning models use.
- It features two coupled recurrent modules: a high-level (H) module for abstract,
deliberate reasoning, and a low-level (L) module for fast, detailed computations.This structure avoids the rapid convergence of standard recurrent models through a process we term “hierarchical convergence.” The slow-updating H-module advances only after the fast-updating L-module has completed multiple computational steps and reached a local equilibrium, at which point the L-module is reset to begin a new computational phase.

There is a slow planner(H) and a fast executor(L). The H gives a high-level task for L to do. The L is superfast and does multiple fast iterations to do the task. Note that all this happens within the network in one go, and the model actually runs only once for the whole task. So if it is an LLM, then all the task will be performed in one go.
The H provides a stable guiding signal for L.

There is a system called Adaptive Computation Time (ACT) which allows RNNs to dynamically adjust the number of computational steps based on the complexity of the input. However, ACT can lead to instability during training, as the model may struggle to determine the optimal number of steps for each input, resulting in erratic behavior and convergence issues. 

- does this thing scale well
- can i use this to make an LLM
  - or can i use this as a tool to assist LLM reasoning
- H will run for N steps and L will run for T steps, but who decides N and T?
  
# Deep supervision
Deep supervision allows the model to train efficiently and maintain stability by:

1.  **Periodic Learning:** It is inspired by the neuroscientific principle that periodic neural oscillations regulate when learning occurs in the brain.
2.  **Segmented Computation:** The model breaks down its forward computation into smaller sequential parts called **segments**.
3.  **Gradient Isolation (The Key Step):** After each segment is executed and a loss is computed, the resulting hidden state is immediately **"detached"** from the computation graph.

This detachment means that gradients from a later segment cannot flow backward through and influence an earlier segment. This effectively simplifies the complex recursive gradient calculation down to an efficient 1-step approximation.

The overall benefit is that deep supervision provides frequent feedback to the High-level module and acts as a regularization mechanism, leading to superior empirical performance and increased stability in deep equilibrium models.![alt text](image-2.png)