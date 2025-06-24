import json
import pandas as pd
from typing import Any, Dict, List, Callable, Any, Union
from collections import Counter 
from llm_eval.utils.logging import get_logger
from llm_eval.analysis import AnalysisReportGenerator, format_report_as_markdown
import importlib

logger = get_logger(name="util", level="INFO")

class EvaluationResult:
    """
    Encapsulates the evaluation output:
    - metrics: Dict[str, float] or more complex structure
    - samples: List[Dict[str, Any]], each representing a data instance with 'prediction', 'reference', etc.
    - info: Dict containing pipeline metadata (model name, dataset, etc.)

    Provides helper methods for introspection, error analysis, 
    and dictionary-like convenience access.
    """
    def __init__(
        self,
        metrics: Dict[str, Any],
        samples: List[Dict[str, Any]],
        info: Dict[str, Any]
    ):
        self.metrics = metrics
        self.samples = samples
        self.info = info

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire result to a dictionary (similar to the old JSON structure).
        """
        return {
            "metrics": self.metrics,
            "samples": self.samples,
            "info": self.info
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame where each row is a sample, with columns:
         - "input", "reference", "prediction"
         - Possibly flattened fields like "evaluation.is_correct"
         - Additional fields if they exist
        """
        df = pd.DataFrame(self.samples)
        if "evaluation" in df.columns:
            # Flatten 'evaluation' dict into separate columns 
            eval_df = df["evaluation"].apply(pd.Series)
            df = pd.concat([df.drop(columns=["evaluation"]), eval_df.add_prefix("eval_")], axis=1)
        return df

    def save_json(self, path: str):
        """
        Save the entire result (metrics, samples, info) to a JSON file.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def benchhub_citation_report(self, output_path: str) -> None:
        """
        Generates a LaTeX citation report for evaluations run with the BenchHub dataset.
        
        This method creates a LaTeX table summarizing the datasets included in the evaluation
        and provides the necessary BibTeX entries for citation. The report is saved to the
        specified output path.

        Args:
            output_path (str): The file path where the LaTeX report will be saved.
        
        Raises:
            ValueError: If this method is called on an EvaluationResult that was not
                        generated from a BenchHub dataset run (i.e., 'benchmark_details' missing in info).
        """
        if "benchmark_details" not in self.info:
            raise ValueError(
                "This report can only be generated for results from a BenchHub dataset run, "
                "as it requires 'benchmark_details' in the 'info' dictionary."
            )

        # 1. Count samples for each benchmark
        benchmark_names = [
            sample["metadata"]["benchmark_name"]
            for sample in self.samples
            if "metadata" in sample and "benchmark_name" in sample["metadata"]
        ]
        sample_counts = Counter(benchmark_names)

        # 2. Build the LaTeX table rows
        table_rows = []
        benchmark_details = self.info["benchmark_details"]
        for benchmark_name, details in benchmark_details.items():
            count = sample_counts.get(benchmark_name, 0)
            citation_key = details.get("citation_key", "UNKNOWN")
            table_rows.append(f"\\cite{{{citation_key}}} & {count} \\\\")
        
        table_content = "\n".join(table_rows)

        # 3. HRET Citation (from README.md)
        hret_citation = """@article{lee2025hret,
  title={HRET: A Self-Evolving LLM Evaluation Toolkit for Korean},
  author={Lee, Hanwool and Kim, Soo Yong and Choi, Dasol and Baek, SangWon and Hong, Seunghyeok and Jeong, Ilgyun and Hwang, Inseon and Lee, Naeun and Son, Guijin},
  journal={arXiv preprint arXiv:2503.22968},
  year={2025}
}"""

        # 4. BenchHub Citation (provided by user)
        benchhub_citation = """@misc{kim2025benchhub,
      title={BenchHub: A Unified Benchmark Suite for Holistic and Customizable LLM Evaluation}, 
      author={Eunsu Kim and Haneul Yoo and Guijin Son and Hitesh Patel and Amit Agarwal and Alice Oh},
      year={2025},
      eprint={2506.00482},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.00482}, 
}"""

        # 5. Build the full LaTeX report string
        report_template = f"""
The evaluation dataset are sampled using BenchHub~\\cite{{kim2025benchhub}}, and the evaluation are conducted using hret~\\cite{{lee2025hret}}.
The individual datasets include in the evaluation set, along with their statistics, are summarized in Table~\\ref{{tab:eval-dataset}}.

% Please add the following required packages to your document preamble:
% \\usepackage{{booktabs}}
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{@{{}}ll@{{}}}}
\\toprule
\\textbf{{Dataset}} & \\textbf{{Number of Samples}} \\\\ \\midrule
{table_content}
\\bottomrule
\\end{{tabular}}
\\caption{{Breakdown of datasets included in the evaluation set.}}
\\label{{tab:eval-dataset}}
\\end{{table}}

% --- BibTeX Entries ---

{hret_citation}

{benchhub_citation}
"""
        # Add citations for each individual benchmark
        for details in benchmark_details.values():
            report_template += f"\n{details.get('citation', '')}\n"

        # 6. Write to file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_template.strip())
            logger.info(f"BenchHub citation report successfully saved to '{output_path}'.")
        except IOError as e:
            logger.error(f"Failed to write citation report to '{output_path}': {e}", exc_info=True)
            raise

    def __repr__(self):
        return f"EvaluationResult(metrics={self.metrics}, info={self.info}, samples=[...])"

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to 'metrics', 'samples', 'info'.

        Example:
            result["metrics"] -> self.metrics
            result["samples"] -> self.samples
            result["info"]    -> self.info
        """
        if key == "metrics":
            return self.metrics
        elif key == "samples":
            return self.samples
        elif key == "info":
            return self.info
        else:
            raise KeyError(f"'{key}' is not a valid key. Use 'metrics', 'samples', or 'info'.")

    def __contains__(self, key: str) -> bool:
        """
        Check if a key is one of the valid fields: 'metrics', 'samples', 'info'.
        """
        return key in ["metrics", "samples", "info"]

    def get(self, key: str, default=None) -> Any:
        """
        Emulate dict.get() behavior. Returns the corresponding field if it exists,
        otherwise returns default.
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            return default
    
    def analysis_report(
        self, top_n: int = 5, output_format: str = 'markdown'
    ) -> Union[str, Dict[str, Any]]:
        """
        Generates an in-depth analysis report of the evaluation results.
        
        This method acts as a facade, delegating the actual analysis logic
        to the `AnalysisReportGenerator` class for better modularity.

        Args:
            top_n (int): The number of top items to show for keywords, patterns, etc.
            output_format (str): The desired output format. Can be 'markdown' or 'dict'.

        Returns:
            Union[str, Dict[str, Any]]: The formatted report as a string or a structured dictionary.
        
        Raises:
            ValueError: If an unsupported output format is requested.
        """
        if not self.samples:
            return "No samples available for analysis."

        try:
            # 1. Prepare data for the analysis module
            df = self.to_dataframe()
            
            # 2. Instantiate the generator and create the report data
            generator = AnalysisReportGenerator(df, self.metrics)
            report_data = generator.generate(top_n=top_n)

            # 3. Return the result in the requested format
            if output_format == 'dict':
                return report_data
            elif output_format == 'markdown':
                # Delegate formatting to the analysis module as well
                return format_report_as_markdown(report_data)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            # Catch potential errors (e.g., spaCy model not found)
            return f"An error occurred while generating the report: {e}"

def _load_function(func_path: str) -> Callable:
    """
    Dynamically load a function given its full path as a string.
    For example: "my_package.my_module.my_function"
    """
    try:
        module_path, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        if not callable(func):
            raise ValueError(f"'{func_path}' is not callable.")
        return func
    except Exception as e:
        logger.error(f"Failed to load function from path '{func_path}': {e}")
        raise e