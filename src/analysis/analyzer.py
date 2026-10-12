import gzip
import json
from preprocessing import preprocess
from prompt_statistics import collect_prompt_stats
from gen_statistics import collect_all_stats
from report_builder import build_report


def run_analysis(data, config):
    """
    Runs the fairness analysis given the previously computed data and builds the report.

    Args:
        data: The data to analyze.
        config: The configuration for the analysis.
    """
    verbose = config.get('verbose', 0)
    if verbose > 0:
        print("Running fairness analysis...")
    
    # Preprocess the data
    preprocessed_data = preprocess(data)

    # Collect prompt statistics
    prompt_stats = collect_prompt_stats(preprocessed_data, verbose)
    
    # Collect general statistics
    gen_stats = collect_all_stats(preprocessed_data, verbose)

    analysis_results = {
        'prompt_stats': prompt_stats,
        'gen_stats': gen_stats
    }

    # Build the report
    build_report(data, analysis_results, config)


if __name__ == "__main__":
    
    metadata_json_path = './assets/all_metadata.json.gz'
    with gzip.open(metadata_json_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    config = {
        'verbose': 2
    }

    run_analysis(data, config)