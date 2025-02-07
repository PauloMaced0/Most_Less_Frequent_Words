import re
import os
import tracemalloc
import time
import nltk
from collections import Counter
from scipy.stats import kendalltau
from nltk.corpus import stopwords
from src.FixedProbabilityCounter import FixedProbabilityCounter
from src.SpaceSavingCounter import SpaceSavingCounter

nltk.data.path.append("./")

# Download stopwords if you haven't already
nltk.download('stopwords', download_dir="./")

# Create sets of stopwords in English and Portuguese
stopwords_en = set(stopwords.words("english"))
stopwords_pt = set(stopwords.words("portuguese"))
combined_stopwords = stopwords_en.union(stopwords_pt)

def _preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Replace non-alphabetic characters with space
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space, then strip
    words = text.split()

    # Filter out any word that is in the combined stopword list
    filtered_words = [word for word in words if word not in combined_stopwords and len(word) > 1]
    
    return filtered_words

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return _preprocess_text(text)

def run_fixed_probability_multiple_times(words, probability=1/32, num_runs=10):
    """
    Runs FixedProbabilityCounter 'num_runs' times on the given word list,
    accumulates the sampled counts, and returns a dict of {word: averaged_estimated_count}.
    """

    # 1) We'll accumulate the raw sampled counts in sum_sampled
    sum_sampled = Counter()

    for _ in range(num_runs):
        # optional: set a seed if reproducibility is desired
        # random.seed(run_index)

        fpc = FixedProbabilityCounter(probability=probability)
        for w in words:
            fpc.add(w)

        # fpc.counts is a Counter of how many times each word was *sampled*
        for w, c in fpc.counts.items():
            sum_sampled[w] += c

    # 2) Build averaged_estimates
    averaged_estimates = {}
    for w, sampled_sum in sum_sampled.items():
        # average number of times w was sampled
        avg_sampled = sampled_sum / num_runs

        # scale by (1 / probability) to get frequency estimate
        estimated_freq = avg_sampled * (1 / probability)
        averaged_estimates[w] = estimated_freq

    return averaged_estimates


def get_top_k_from_dict(d, k):
    """Return the top-k items (word, count) from a dict, sorted descending."""
    if k <= 0:
        return []
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return items[:k]

def get_bottom_k_from_dict(d, k):
    """Return the bottom-k items (word, count) from a dict, sorted ascending."""
    if k <= 0:
        return []
    items = sorted(d.items(), key=lambda x: x[1])  # ascending
    return items[:k]

# Precision metric: how many items in approx’s top/bottom are also in exact’s top/bottom
def compute_precision(exact_list, approx_list):
    """
    Given two lists of (word, count) pairs, compute how many words overlap
    (intersection) divided by the number of words in the approximate list.
    """
    exact_set = set([w for w, _ in exact_list])
    approx_set = set([w for w, _ in approx_list])
    intersect = exact_set.intersection(approx_set)
    # Precision = (# of matching words) / (# of words in approx_list)
    return len(intersect) / len(approx_list) if approx_list else 0.0

def display_top_words(file_name, top_k=10):
    """
    Displays the top_k most frequent words for the given file_name
    using:
      1) Exact counting
      2) Fixed probability
      3) Space saving
    """
    print(f"\nProcessing {file_name}")
    
    words = process_file(file_name)

    exact_counter = Counter(words)
    exact_top = exact_counter.most_common(top_k)

    prob_counter = FixedProbabilityCounter(probability=1/32)
    for w in words:
        prob_counter.add(w)
    approx_top_prob = prob_counter.get_top_k(top_k)  # returns list of (word, estimated_count)

    space_saving = SpaceSavingCounter(k=100)
    for w in words:
        space_saving.add(w)
    approx_top_ss = space_saving.get_top_k()[:top_k]  # returns list of (word, approximate_count)

    # 5. Print the results
    print(f"   [Exact]            Top-{top_k}:", exact_top)
    print(f"   [FixedProbability] Top-{top_k}:", approx_top_prob)
    print(f"   [SpaceSaving]      Top-{top_k}:", approx_top_ss)

def compare_top_k_order_exact_vs_spacesaving(words, k=10):
    exact_counter = Counter(words)
    exact_top_k = exact_counter.most_common(10)

    space_saving = SpaceSavingCounter(k=k)
    for w in words:
        space_saving.add(w)
    approx_top_k = space_saving.get_top_k()[:10]

    # Extract just the word order
    exact_top_k_words = [w for (w, _) in exact_top_k]
    approx_top_k_words = [w for (w, _) in approx_top_k]

    # Build a combined set of words
    combined_words = list(set(exact_top_k_words + approx_top_k_words))

    # Assign ranks in each list:
    # If a word isn't in the top-k for that list, we can either
    # give it rank k+1, or we ignore it. Let's do rank = k+1 if not found.
    rank_exact = {}
    for i, w in enumerate(exact_top_k_words):
        rank_exact[w] = i + 1  # ranks start at 1
    # words not in top-k get rank k+1
    for w in combined_words:
        if w not in rank_exact:
            rank_exact[w] = k + 1

    rank_approx = {}
    for i, w in enumerate(approx_top_k_words):
        rank_approx[w] = i + 1
    for w in combined_words:
        if w not in rank_approx:
            rank_approx[w] = k + 1

    exact_ranks = [rank_exact[w] for w in combined_words]
    approx_ranks = [rank_approx[w] for w in combined_words]

    tau, _ = kendalltau(exact_ranks, approx_ranks)

    strict_order_matches = sum(e == a for e, a in zip(exact_top_k_words, approx_top_k_words))

    return {
        "kendall_tau": tau,
        "strict_position_matches": strict_order_matches
    }

def measure_method(method_name, func, *args, **kwargs):
    """
    Measures execution time and approximate peak memory usage of a function 'func'
    which implements one of the methods (exact, fixed probability, or space saving).
    
    Returns a dict: {
        'method': <method_name>,
        'time_sec': <execution_time_in_seconds>,
        'memory_kb': <peak_memory_in_kilobytes>
    }
    """
    # 1) Start measuring memory usage
    tracemalloc.start()
    
    # 2) Record start time
    start_time = time.perf_counter()
    
    # 3) Call the function
    output = func(*args, **kwargs)
    
    # 4) End time
    end_time = time.perf_counter()
    
    # 5) Capture snapshot
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Convert peak memory usage to KB (default is bytes)
    peak_kb = peak / 1024.0
    
    return {
        'method': method_name,
        'time_sec': end_time - start_time,
        'memory_kb': peak_kb,
        'output': output  # if you need the actual result of func, store here
    }

def exact_count_method(words):
    """
    Exact method: build a full Python Counter over all words.
    Return the top-10 for demonstration (but not strictly necessary).
    """
    c = Counter(words)
    return c.most_common(10)

def fixed_probability_method(words, probability=1/32):
    """
    Builds and runs a FixedProbabilityCounter, then returns top-10 items.
    """
    fpc = FixedProbabilityCounter(probability=probability)
    for w in words:
        fpc.add(w)
    return fpc.get_top_k(10)

def space_saving_method(words, k=100):
    """
    Builds and runs a SpaceSavingCounter, then returns top-10 items.
    """
    ssc = SpaceSavingCounter(k=k)
    for w in words:
        ssc.add(w)
    return ssc.get_top_k()[:10]

def clean_filename(filename):
    """
    Removes the file extension and replaces underscores with spaces.
    
    Args:
        filename (str): The original filename.
    
    Returns:
        str: The cleaned filename.
    """
    # Remove the file extension
    name_without_ext = os.path.splitext(filename)[0]
    # Replace underscores with spaces
    clean_name = name_without_ext.replace('_', ' ')
    return clean_name
