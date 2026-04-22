from __future__ import annotations

import numpy as np

from pipeline.config import TOP_K


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

    generations = 6
    population_size = 12
    half = max(k, total_candidates // 2)
    population = _seed_population(self.rng, total_candidates, population_size, half, k)
    best = max(population, key=lambda chromosome: _fitness(chromosome, candidate_scores, candidate_cat_idx, candidate_popularity, k))

    for _ in range(generations):
        elite = max(population, key=lambda chromosome: _fitness(chromosome, candidate_scores, candidate_cat_idx, candidate_popularity, k))
        if _fitness(elite, candidate_scores, candidate_cat_idx, candidate_popularity, k) > _fitness(best, candidate_scores, candidate_cat_idx, candidate_popularity, k):
            best = elite.copy()
        population = [elite.copy()]
        while len(population) < population_size:
            first = _tournament(self.rng, population, candidate_scores, candidate_cat_idx, candidate_popularity, k)
            second = _tournament(self.rng, population, candidate_scores, candidate_cat_idx, candidate_popularity, k)
            child = _mutate(self.rng, _crossover(self.rng, first, second, total_candidates, k), total_candidates, k)
            population.append(child)

    order = sorted(best, key=lambda idx: candidate_scores[idx], reverse=True)
    return candidate_ids[np.array(order)]


def _fitness(chromosome: list[int], scores: np.ndarray, cat_idx: np.ndarray, popularity: np.ndarray, k: int) -> float:
    relevance = float(np.mean(scores[chromosome]))
    diversity = len(set(cat_idx[chromosome])) / k
    novelty = float(np.mean(1.0 - popularity[chromosome]))
    return 0.86 * relevance + 0.10 * diversity + 0.04 * novelty


def _seed_population(rng: np.random.Generator, total: int, pop_size: int, half: int, k: int) -> list[list[int]]:
    population: list[list[int]] = []
    for index in range(pop_size):
        limit = total if index < pop_size // 3 else half
        source = np.arange(limit)
        population.append(list(rng.choice(source, size=k, replace=False)))
    return population


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


def _crossover(rng: np.random.Generator, first: list[int], second: list[int], total: int, k: int) -> list[int]:
    cut = int(rng.integers(2, k - 1))
    child = first[:cut] + [item for item in second if item not in first[:cut]]
    for idx in range(total):
        if len(child) == k:
            break
        if idx not in child:
            child.append(idx)
    return child[:k]


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
