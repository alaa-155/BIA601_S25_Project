# Selected Paper

## Primary reference
**Wei Zhang, Zonghua Wu (2024)**  
**E-commerce recommender system based on improved K-means commodity information management model**  
Journal: **Heliyon**  
DOI: `10.1016/j.heliyon.2024.e29045`
PMC: `PMC11063989`

## Why this paper was selected
1. It is inside the required time window (2024-2026).
2. It is directly about e-commerce recommendation.
3. It uses a Genetic Algorithm in a real technical role.
4. Its main logic can be adapted to the assignment dataset.

## How the project adapts the paper
The original paper uses a Genetic Algorithm to improve K-means initialization and reduce weak local solutions.

This project keeps the same **scientific spirit** - using evolutionary optimization to improve recommendation quality - but adapts the GA to a more suitable point for the course dataset:
- first, build a hybrid recommender from behavior, ratings, category affinity, price affinity, and popularity
- second, apply the Genetic Algorithm as a reranking optimizer over the best candidate products

So the project is **paper-inspired, technically adapted, and demonstrable through a working web application**.
