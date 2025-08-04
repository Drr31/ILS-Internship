# SYNCUP - Week 1: Foundations & Theoretical Preparation
Objective:
Establish a solid foundational understanding of Neural Networks and the CoVerNet paper, alongside refreshing essential mathematical concepts that are critical for the implementation and analysis of neural network coverage metrics.


Activities Undertaken:
1. Reading and Understanding the CoVerNet Paper
Thorough review of the paper “CoVerNet: Toward CoVerage Testing for Neural Networks Based on Formally Verified Equivalence Classes”.

- Extracted key concepts including:

- Definition of coverage in neural networks.

- Partitioning input space into equivalence classes.

- The concept of Prior Equivalence Classes (PEC) and their formal verification.

- Methods for automatic property generation and behavioral coverage.



2. Neural Network Fundamentals Review
Revisited architecture basics:

- Layers (input, hidden, output).

- Common activation functions: (ReLU, ...

- Forward propagation and backpropagation algorithms.

- Loss functions relevant for classification.

- Emphasis on understanding how the neural network processes input features and produces outputs.

3. Detailed Linear Algebra Refresher
Focusing on mathematical tools underpinning neural network operations and clustering methods used in CoVerNet.

   - Vectors and Vector Spaces:
     - Definition of vectors in Euclidean distance:
     - Operations: addition, scalar multiplication.
     - Basis and dimension concepts.


   - Matrices and Matrix Operations:
      - Matrix definition and notation.
      - Matrix multiplication: conditions and properties.
      - Transpose and inverse matrices.
   

   - Dot Product and Norms:
      - Dot product definition $a⋅b=∑i=1 --> n, a[i]b[i].$
      - Euclidean norm $L ^ 2.$ norm $∥a∥= a⋅a​.$
      - Role of norms in distance calculations for clustering and equivalence classes.

   - Eigenvalues and Eigenvectors (Overview):
       - Definition and intuition.
       - Their significance in understanding transformations and PCA (used sometimes for dimensionality reduction).

   - Distance Metrics in Input Space:
  
         - Euclidean distance: $ d(x,y)=∥x−y∥ $
         - Its use in K-means clustering for grouping input vectors into equivalence classes.



4. Additional Theoretical Concepts
- Overview of clustering algorithms, particularly K-means, since it's central to the PEC step.
- Basics of formal methods applied to neural network verification, setting the stage for later chapters.




```python

```
