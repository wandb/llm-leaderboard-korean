import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from typing import Dict, Any, List

# Load the spaCy model once to be reused.
try:
    NLP_MODEL = spacy.load("ko_core_news_sm")
except OSError:
    print("Warning: spaCy Korean model ('ko_core_news_sm') not found. Report analysis will be limited.")
    print("-> Please run: python -m spacy download ko_core_news_sm")
    NLP_MODEL = None


class AnalysisReportGenerator:
    """
    Generates an in-depth analysis report from an evaluation result DataFrame.
    """

    def __init__(self, df: pd.DataFrame, metrics: Dict[str, Any]):
        """
        Initializes the report generator.

        Args:
            df (pd.DataFrame): The DataFrame containing evaluation samples.
            metrics (Dict[str, Any]): A dictionary of overall metrics (e.g., accuracy).
        """
        if NLP_MODEL is None:
            raise RuntimeError("spaCy model not loaded. Cannot generate analysis report.")
        
        self.df = df
        self.metrics = metrics
        self.correct_df = self.df[self.df['eval_is_correct'] == True]
        self.incorrect_df = self.df[self.df['eval_is_correct'] == False]
        self.nlp = NLP_MODEL

    def generate(self, top_n: int = 5) -> Dict[str, Any]:
        """
        Orchestrates the analysis of all components and returns the final report data.

        Args:
            top_n (int): The number of top items to show for keywords, error patterns, etc.

        Returns:
            Dict[str, Any]: A dictionary containing the structured analysis report.
        """
        report_data = {
            "summary": self._analyze_summary(),
            "subsets": self._analyze_subsets(),
            "linguistic_quality": self._analyze_linguistic_quality(),
            "error_analysis": self._analyze_errors(top_n),
            "examples": self._select_examples(),
        }
        return report_data

    def _analyze_summary(self) -> Dict[str, Any]:
        """Analyzes and returns the overall performance summary."""
        total_samples = len(self.df)
        num_correct = len(self.correct_df)
        accuracy = self.metrics.get('accuracy', num_correct / total_samples if total_samples > 0 else 0)
        return {
            "total_samples": total_samples,
            "num_correct": num_correct,
            "num_incorrect": len(self.incorrect_df),
            "accuracy": f"{accuracy:.2%}",
            "other_metrics": {k: v for k, v in self.metrics.items() if k != 'accuracy'}
        }

    def _analyze_subsets(self) -> Dict[str, str]:
        """Analyzes performance for each subset if available."""
        if '_subset_name' in self.df.columns:
            subset_accuracy = self.df.groupby('_subset_name')['eval_is_correct'].mean().to_dict()
            return {k: f"{v:.2%}" for k, v in subset_accuracy.items()}
        return {}
        
    def _analyze_linguistic_quality(self) -> Dict[str, Dict]:
        """Analyzes linguistic features of the generated predictions."""
        return {
            "correct_predictions": self._get_linguistic_features(self.correct_df['prediction']),
            "incorrect_predictions": self._get_linguistic_features(self.incorrect_df['prediction']),
        }

    def _get_linguistic_features(self, text_series: pd.Series) -> Dict:
        """Calculates linguistic features like TTR for a series of texts."""
        if text_series.empty:
            return {"lexical_diversity_ttr": 0.0}
            
        all_tokens = [token.lemma_ for doc in self.nlp.pipe(text_series) for token in doc if token.is_alpha]
        ttr = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
        
        return {"lexical_diversity_ttr": f"{ttr:.4f}"}

    def _analyze_errors(self, top_n: int) -> Dict[str, Any]:
        """Performs an in-depth analysis of incorrect predictions."""
        if self.incorrect_df.empty:
            return {}
        
        # Find the most common nouns in incorrect predictions.
        incorrect_nouns = [
            token.lemma_ for doc in self.nlp.pipe(self.incorrect_df['prediction']) 
            for token in doc if token.pos_ == 'NOUN'
        ]
        common_error_keywords = dict(Counter(incorrect_nouns).most_common(top_n))

        return {
            "top_error_keywords": common_error_keywords,
        }
        
    def _select_examples(self) -> Dict[str, Dict]:
        """Selects representative examples for correct and incorrect cases."""
        examples = {}
        if not self.correct_df.empty:
            best_case = self.correct_df.sample(1).iloc[0].to_dict()
            # Convert numpy types to native python types for JSON serialization
            examples["best_case"] = {k: v.item() if hasattr(v, 'item') else v for k, v in best_case.items()}
        if not self.incorrect_df.empty:
            worst_case = self.incorrect_df.sample(1).iloc[0].to_dict()
            examples["worst_case"] = {k: v.item() if hasattr(v, 'item') else v for k, v in worst_case.items()}
        return examples


def format_report_as_markdown(data: dict) -> str:
    """
    Formats the analysis data dictionary into a human-readable Markdown string.

    Args:
        data (dict): The report data generated by AnalysisReportGenerator.

    Returns:
        str: A Markdown formatted report.
    """
    report_lines = ["# LLM Performance Analysis Report"]

    # --- Summary Section ---
    summary = data.get('summary', {})
    if summary:
        report_lines.append("## 1. Overall Performance")
        for key, value in summary.items():
            if key == "other_metrics" and value:
                report_lines.append(f"- **Other Metrics**: ")
                for mk, mv in value.items():
                    report_lines.append(f"  - {mk}: {mv}")
            elif key != "other_metrics":
                 report_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")


    return "\n".join(report_lines)