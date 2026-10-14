import gzip
import json
from .preprocessing import preprocess
from .prompt_statistics import collect_prompt_stats
from .general_statistics import collect_general_stats
from ..report.report_builder import build_report


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
    prompt_stats = collect_prompt_stats(preprocessed_data, config=config)

    # Collect general statistics
    general_stats = collect_general_stats(preprocessed_data, config=config)


    analysis_results = {
        'prompt_stats': prompt_stats,
        'general_stats': general_stats
    }

    # Build the report
    return build_report(
        data,
        analysis_results,
        config,
        open_report=config.get("open_report", True),
        wait_for_close=config.get("wait_for_close", True),
    )


if __name__ == "__main__":

    config_path = './assets/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    metadata_json_path = './assets/all_metadata.json.gz'
    with gzip.open(metadata_json_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    config['verbose'] = 2
    config['report_dir'] = './assets/report'

    run_analysis(data, config)