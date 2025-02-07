import heapq

class SpaceSavingCounter:
    def __init__(self, k):
        self.k = k
        self.counter = {}
        self.heap = []

    def add(self, item):
        if item in self.counter:
            self.counter[item] += 1
        elif len(self.counter) < self.k:
            self.counter[item] = 1
            heapq.heappush(self.heap, (1, item))
        else:
            # Replace the item with the smallest count
            min_count, min_item = heapq.heappop(self.heap)
            self.counter.pop(min_item)
            self.counter[item] = min_count + 1
            heapq.heappush(self.heap, (min_count + 1, item))

    def get_top_k(self):
        return sorted(self.counter.items(), key=lambda x: -x[1])
