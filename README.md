# MetaLearnUnsupPlast
Metalearning unsupervised plasticity rules for semi-supervised learning through evolutionary algorithms.

Each agent in a population is equipped with a plasticity rule for 
unsupervised learning. The plasticity rules are parameterized by a set of meta-parameters.

When agents are given unlabeled inputs (using UnsupLifetime), their parameters are updated 
using the plasticity rules. This defines an embedding.

For labeled examples, embeddings are passed into a supervised learning algorithm. The validation or 
test error is defined as the meta-loss function for a given agent. The meta-loss quantifies the effectiveness
of the embedding for supervised learning. 

Every generation, new agents (with new metaparameters) are created based on the 
meta-loss of the previous generation's agents. Agents with lower meta-loss propagate 
their meta-parameters more effectively to the next generation. In this way,
the average meta-loss of the population should decrease across generations.

Therefore, the population evolves unsupervised plasticity rules for semi-supervised learning.
This is a form of meta-learning through evolutionary learning. 
