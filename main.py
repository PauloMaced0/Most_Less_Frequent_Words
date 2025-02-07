import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from src.utils import (
    clean_filename,
    compare_top_k_order_exact_vs_spacesaving,
    process_file,
    compute_precision,
    run_fixed_probability_multiple_times,
    get_top_k_from_dict,
    get_bottom_k_from_dict,
    display_top_words,
    measure_method,
    exact_count_method,
    fixed_probability_method,
    space_saving_method
)

def main():
    folder_path = "./books"
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    # We'll store data for plotting
    ks_to_test = [5, 10, 15, 20]
    probability = 1/32
    num_runs = 10

    # Display top-10 frequent words for each file
    display_top_words("./books/Os_Lusíadas_ENG.txt", top_k=10)
    display_top_words("./books/Os_Lusíadas_PT.txt", top_k=10)

    precision_data = []
    ranking_similarity_data = []
    performance_summary = []

    ks_for_similarity = [
        10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250,
        275, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000,
        1200, 1400, 1600, 1800, 2000, 2250, 2500, 2750, 3000
    ]

    for file in files:
        print()
        print("=" * 85)
        print(f"Processing {file}")
        words = process_file(os.path.join(folder_path, file))

        # -- Exact Counter --
        exact_counter = Counter(words)
        total_words = sum(exact_counter.values())
        unique_words = len(exact_counter)
        print(f"  Total words: {total_words}, unique words: {unique_words}")

        # -- Run fixed probability multiple times, then average --
        averaged_estimates = run_fixed_probability_multiple_times(words,
                                                                  probability=probability,
                                                                  num_runs=num_runs)
        for k in ks_to_test:
            # Exact top/bottom
            exact_top_k = exact_counter.most_common(k)
            bottom_k_ex = get_bottom_k_from_dict(exact_counter, k)

            # Approx top/bottom from *averaged* estimates
            approx_top_k = get_top_k_from_dict(averaged_estimates, k)
            approx_bottom_k = get_bottom_k_from_dict(averaged_estimates, k)

            top_precision = compute_precision(exact_top_k, approx_top_k)
            bottom_precision = compute_precision(bottom_k_ex, approx_bottom_k)

            precision_data.append({
                'file': file,
                'k': k,
                'top_precision': top_precision,
                'bottom_precision': bottom_precision
            })

        # ------------------------------------------------
        # 2) EXACT counting
        # ------------------------------------------------
        exact_result = measure_method(
            "Exact",
            exact_count_method,
            words
        )
        performance_summary.append({
            'method': exact_result['method'],
            'time_sec': exact_result['time_sec'],
            'memory_kb': exact_result['memory_kb'],
            'top_10_sample': exact_result['output']  # If you want to see an example
        })
        
        # ------------------------------------------------
        # 3) FIXED PROBABILITY
        # ------------------------------------------------
        fprob_result = measure_method(
            "FixedProbability",
            fixed_probability_method,
            words,
            probability=1/32
        )
        performance_summary.append({
            'method': fprob_result['method'],
            'time_sec': fprob_result['time_sec'],
            'memory_kb': fprob_result['memory_kb'],
            'top_10_sample': fprob_result['output']
        })
        
        # ------------------------------------------------
        # 4) SPACE SAVING
        # ------------------------------------------------
        ss_result = measure_method(
            "SpaceSaving",
            space_saving_method,
            words,
            k=50
        )
        performance_summary.append({
            'method': ss_result['method'],
            'time_sec': ss_result['time_sec'],
            'memory_kb': ss_result['memory_kb'],
            'top_10_sample': ss_result['output']
        })

        # Print summary
        print("\n----- Performance Summary -----")
        for row in performance_summary[-3:]:  # Print the last three entries
            print(f"{row['method']}: {row['time_sec']:.4f} seconds, {row['memory_kb']:.2f} KB")

        print("=" * 85)

        # -- Ranking Similarity for All k --
        for k in ks_for_similarity:
            comparison = compare_top_k_order_exact_vs_spacesaving(words, k=k)
            ranking_similarity_data.append({
                'file': clean_filename(file),
                'k': k,
                'kendall_tau': comparison['kendall_tau'],
                'strict_position_matches': comparison['strict_position_matches']
            })

    # Turn results into a DataFrame, for easy plotting
    precision_df = pd.DataFrame(precision_data)
    ranking_similarity_df = pd.DataFrame(ranking_similarity_data)
    print("\nSample of precision data:")
    print(precision_df .head())

    print("\nSample of ranking similarity data:")
    print(ranking_similarity_df.head())

    # Example: plot the average top-precision vs. k across all files
    grouped = precision_df.groupby("k").mean(numeric_only=True).reset_index()

    plt.figure(figsize=(8,5))
    plt.plot(grouped['k'], grouped['top_precision'], marker='o', label='Top-k Precision')
    plt.plot(grouped['k'], grouped['bottom_precision'], marker='s', label='Bottom-k Precision')
    plt.ylim(0, 1.05)
    plt.xlabel("k")
    plt.ylabel("Precision (Overlap Ratio)")
    plt.title(f"Fixed Probability (Avg of {num_runs} runs), p=1/32")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -- Plotting Precision vs. k across all files --
    plt.figure(figsize=(10, 6))
    for file in ranking_similarity_df['file'].unique():
        file_data = ranking_similarity_df[ranking_similarity_df['file'] == file]
        plt.plot(file_data['k'], file_data['kendall_tau'], marker='o', label=file)

    plt.xlabel("k")
    plt.ylabel("Kendall's Tau")
    plt.title("Ranking Similarity (Space-Saving vs. Exact Count) Across All Files")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -- Aggregate Kendall's Tau across all files for each k --
    aggregated_tau = ranking_similarity_df.groupby('k')['kendall_tau'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(aggregated_tau['k'], aggregated_tau['kendall_tau'], marker='s', color='b')
    plt.xlabel("k")
    plt.ylabel("Average Kendall's Tau")
    plt.title("Average Ranking Similarity Across All Files")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -- Scatter Plot with Color Coding for Strict Position Matches --
    # Define color based on whether there are strict position matches
    ranking_similarity_df['color'] = ranking_similarity_df['strict_position_matches'].apply(
        lambda x: 'red' if x > 0 else 'blue'
    )

    plt.figure(figsize=(10, 6))
    for file in ranking_similarity_df['file'].unique():
        file_data = ranking_similarity_df[ranking_similarity_df['file'] == file]
        plt.scatter(
            file_data['k'],
            file_data['kendall_tau'],
            c=file_data['color'],
            label=file if file_data['color'].any() else "_nolegend_",
            alpha=0.6
        )

    plt.xlabel("k")
    plt.ylabel("Kendall's Tau")
    plt.title("Ranking Similarity vs. k (Space-Saving vs. Exact Count) Across All Files")
    plt.grid(True)
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Strict Match', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='No Strict Match', markerfacecolor='blue', markersize=10)
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
