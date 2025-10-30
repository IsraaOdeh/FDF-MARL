# FDF-MARL
FDF-MARL: Federated Multi-Agent Reinforcement Learning for Demographic Fairness
Abstract
Developing responsible, reliable, and trustworthy AI systems requires addressing key pillars such as fairness, privacy, and accuracy.
This work tackles the challenge of achieving demographic fairness in Federated Learning (FL) while maintaining accuracy, preserving
privacy, and ensuring computational efficiency. Most existing research on demographic fairness focuses on general machine learning
algorithms, while relatively few studies address fairness in privacy-critical systems where confidentiality is non-negotiable. To
this end, we propose Federated Multi-Agent Reinforcement Learning for Demographic Fairness in Federated Learning
(FDF-MARL), where agents are trained to jointly optimize demographic fairness and accuracy while balancing the importance of
each group, ensuring that no group dominates over others. During deployment, the trained agents’ policies enable federated clients to
autonomously decide whether to participate in FL communication rounds. This creates an automated federated learning framework
with intelligent clients, each equipped with a customized policy that enhances both global and local performance. Furthermore,
to enable knowledge sharing among agents without exposing confidential client information, we introduce FDF-MARL-AVG, an
extension of FDF-MARL obtained by resuming training after aggregating agents’ policies post-convergence. The proposed models
achieve a balance between fairness and accuracy, delivering the highest F1 scores alongside competitive fairness and computational
efficiency on two public datasets. Additionally,
