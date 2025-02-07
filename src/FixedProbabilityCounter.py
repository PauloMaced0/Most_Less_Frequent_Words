import random
from collections import Counter

class FixedProbabilityCounter:
    def __init__(self, probability=1/32):
        self.probability = probability
        self.counts = Counter()

    def add(self, item):
        if random.random() < self.probability:
            self.counts[item] += 1

    def get_count(self, item):
        return self.counts[item] * int(1 / self.probability)

    def get_top_k(self, k):
        """Return the top-k items by approximate count (descending)."""
        if not self.counts or k <= 0:
            return []
        
        # Build a list of (item, approximate_count) pairs
        items_with_estimates = [(item, self.get_count(item)) for item in self.counts]
        # Sort descending by approximate count
        items_with_estimates.sort(key=lambda x: x[1], reverse=True)
        
        return items_with_estimates[:k]

    def get_bottom_k(self, k):
        """Return the bottom-k items by approximate count (ascending)."""
        if not self.counts or k <= 0:
            return []
        
        items_with_estimates = [(item, self.get_count(item)) for item in self.counts]
        # Sort ascending by approximate count
        items_with_estimates.sort(key=lambda x: x[1])
        
        return items_with_estimates[:k]
