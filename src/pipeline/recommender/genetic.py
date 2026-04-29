from __future__ import annotations
import numpy as np
from pipeline.config import TOP_K

# Select the best recommendation list using a genetic algorithm.
# Each chromosome represents one possible list of recommended products.
# The algorithm improves recommendations through selection, crossover,
# mutation, and keeping the best solution across generations
def ga_select(
    self,
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_cat_idx: np.ndarray,
    candidate_popularity: np.ndarray,
    k: int = TOP_K,
) -> np.ndarray:
    total_candidates = len(candidate_ids)
    if total_candidates <= k:
        return candidate_ids[:k]
    # Define the number of evolution rounds and the number of candidate
    # recommendation lists used in each generation
    generations = 6
    population_size = 12
    half = max(k, total_candidates // 2)
    # Create the initial population of possible recommendation lists.
    # Each list contains product IDs selected from the candidate pool.
    population = _seed_population(self.rng, total_candidates, population_size, half, k)
    # Track the best recommendation list found so far.
    best = max(population, key=lambda chromosome: _fitness(chromosome, candidate_scores, candidate_cat_idx, candidate_popularity, k))
    # Repeat the selection, crossover, and mutation process
    # for several generations to improve the recommendations.
    for _ in range(generations):
        # Elitism: keep the best recommendation list from the current generation.
        elite = max(population, key=lambda chromosome: _fitness(chromosome, candidate_scores, candidate_cat_idx, candidate_popularity, k))
        if _fitness(elite, candidate_scores, candidate_cat_idx, candidate_popularity, k) > _fitness(best, candidate_scores, candidate_cat_idx, candidate_popularity, k):
            best = elite.copy()
        population = [elite.copy()]
        while len(population) < population_size:
            # Select two parent recommendation lists using tournament selection.
            # Stronger lists have a higher chance of producing the next generation.
            first = _tournament(self.rng, population, candidate_scores, candidate_cat_idx, candidate_popularity, k)
            second = _tournament(self.rng, population, candidate_scores, candidate_cat_idx, candidate_popularity, k)
            # Create a child recommendation list by combining the two parents,
            # then apply mutation to explore new recommendation combinations.
            child = _mutate(self.rng, _crossover(self.rng, first, second, total_candidates, k), total_candidates, k)
            population.append(child)

    order = sorted(best, key=lambda idx: candidate_scores[idx], reverse=True)
    return candidate_ids[np.array(order)]

# Evaluate the quality of one recommendation list.
# The fitness score combines relevance, category diversity, and novelty.
def _fitness(chromosome: list[int], scores: np.ndarray, cat_idx: np.ndarray, popularity: np.ndarray, k: int) -> float:
    relevance = float(np.mean(scores[chromosome]))
    diversity = len(set(cat_idx[chromosome])) / k
    novelty = float(np.mean(1.0 - popularity[chromosome]))
    # Give relevance the highest weight, while diversity and novelty
    # help avoid recommending very similar or overly popular products only.
    return 0.86 * relevance + 0.10 * diversity + 0.04 * novelty

# Create the initial population of recommendation lists.
# The first list is based on the strongest candidates, while the others
# include shuffled combinations to give the genetic algorithm variety.
def _seed_population(rng: np.random.Generator, total: int, pop_size: int, half: int, k: int) -> list[list[int]]:
    population: list[list[int]] = []
    for index in range(pop_size):
        limit = total if index < pop_size // 3 else half
        source = np.arange(limit)
        population.append(list(rng.choice(source, size=k, replace=False)))
    return population

# Select the strongest parent from a small random group of chromosomes.
# This gives better recommendation lists a higher chance to continue.
def _tournament(
    rng: np.random.Generator,
    population: list[list[int]],
    scores: np.ndarray,
    cat_idx: np.ndarray,
    popularity: np.ndarray,
    k: int,
) -> list[int]:
    ids = rng.integers(0, len(population), 4)
    return max((population[int(idx)] for idx in ids), key=lambda chromosome: _fitness(chromosome, scores, cat_idx, popularity, k))

# Combine two parent recommendation lists to create a new child list.
# The child keeps unique product positions and is completed from the candidate pool.
def _crossover(rng: np.random.Generator, first: list[int], second: list[int], total: int, k: int) -> list[int]:
    cut = int(rng.integers(2, k - 1))
    child = first[:cut] + [item for item in second if item not in first[:cut]]
    for idx in range(total):
        if len(child) == k:
            break
        if idx not in child:
            child.append(idx)
    return child[:k]

# Randomly modify one product in the recommendation list.
# Mutation helps the algorithm explore new solutions and avoid getting stuck.
def _mutate(rng: np.random.Generator, chromosome: list[int], total: int, k: int) -> list[int]:
    updated = chromosome.copy()
    if rng.random() < 0.60:
        position = int(rng.integers(0, k))
        for idx in range(total):
            if idx not in updated:
                updated[position] = idx
                break
    if rng.random() < 0.35:
        first, second = [int(value) for value in rng.integers(0, k, 2)]
        updated[first], updated[second] = updated[second], updated[first]
    repaired: list[int] = []
    for idx in updated:
        if idx not in repaired:
            repaired.append(idx)
    for idx in range(total):
        if idx not in repaired and len(repaired) < k:
            repaired.append(idx)
    return repaired[:k]
