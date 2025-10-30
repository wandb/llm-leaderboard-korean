import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import logging
from llm_eval.utils.logging import get_logger

import pandas as pd


@dataclass
class ScoringCoefficients:
    """Coefficients for standardized score calculation"""
    year: int
    common_coefft: float
    selection_coefft: float
    constant: float


@dataclass
class TestResult:
    """Individual test result data"""
    year: int
    raw_score: int
    std_score: int
    percentile: int
    grade: int


@dataclass
class ModelStatistics:
    """Aggregated statistics for a model"""
    model_name: str
    avg_std_score: float
    avg_grade: float
    results_by_year: Dict[int, TestResult]


logger = get_logger(name="Korean SAT grade", level=logging.INFO)


class ScoreCalculator:
    """Handles score calculations for different year ranges"""

    def __init__(self, scoring_information: Dict[str, pd.DataFrame]):
        self.scoring_information = scoring_information
        self.coefficients = self._load_coefficients()

    def calculate_result(self, year, score_result: Dict) -> TestResult:
        """Calculate test result based on year"""

        if int(year) > 2021:
            return self._calculate_modern_score(score_result, year)
        else:
            return self._calculate_legacy_score(score_result, year)

    def _load_coefficients(self) -> Dict[int, ScoringCoefficients]:
        """Load scoring coefficients for years 2022-2024"""
        return {
            2022: ScoringCoefficients(2022, 1.156, 0.913, 37.4),
            2023: ScoringCoefficients(2023, 0.980, 0.866, 35.1),
            2024: ScoringCoefficients(2024, 1.124, 0.923, 38.7)
        }

    def _calculate_modern_score(self, score_result: Dict, year: int) -> TestResult:
        """Calculate score for years 2022-2024 using coefficients"""
        grading_df = self.scoring_information[year]
        coeffs = self.coefficients[year]

        common_score = score_result[f"{year}_common_score"]
        choice_score = score_result[f"{year}_choice_score"]

        raw_score = int(score_result[f"{year}_total_score"])
        std_score = round(
            coeffs.common_coefft * common_score +
            coeffs.selection_coefft * choice_score +
            coeffs.constant
        )

        grading_row = grading_df[grading_df['표준점수'] == std_score]
        percentile = int(grading_row['백분위'].iloc[0])
        grade = int(grading_row['등급'].iloc[0])

        return TestResult(year, raw_score, std_score, percentile, grade)

    def _calculate_legacy_score(self, score_result: Dict, year: int) -> TestResult:
        """Calculate score for years 2021 and earlier using lookup table"""
        grading_df = self.scoring_information[year]
        overall_sum = score_result[f"{year}_total_score"]

        score_row = grading_df[grading_df['원점수'] == overall_sum]

        return TestResult(
            year=year,
            raw_score=int(score_row['원점수'].iloc[0]),
            std_score=int(score_row['표준점수'].iloc[0]),
            percentile=int(score_row['백분위'].iloc[0]),
            grade=int(score_row['등급'].iloc[0])
        )
