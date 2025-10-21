#!/usr/bin/env python3
"""
Script for filtering BFCL JSON data files based on test_case_ids_to_generate.json

This script reads the test_case_ids_to_generate.json file and filters each
corresponding JSON data file to only include records with IDs listed in the
test case IDs file.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any


class BFCLDataFilter:
    """Filter BFCL data files based on specified test case IDs."""

    # Mapping from test_case_ids_to_generate.json keys to actual file names
    FILE_MAPPING = {
        "format_sensitivity": "BFCL_v4_format_sensitivity.json",
        "irrelevance": "BFCL_v4_irrelevance.json",
        "live_irrelevance": "BFCL_v4_live_irrelevance.json",
        "live_multiple": "BFCL_v4_live_multiple.json",
        "live_parallel": "BFCL_v4_live_parallel.json",
        "live_parallel_multiple": "BFCL_v4_live_parallel_multiple.json",
        "live_relevance": "BFCL_v4_live_relevance.json",
        "live_simple": "BFCL_v4_live_simple.json",
        "memory_kv": "BFCL_v4_memory.json",
        "memory_rec_sum": "BFCL_v4_memory.json",
        "memory_vector": "BFCL_v4_memory.json",
        "multi_turn_base": "BFCL_v4_multi_turn_base.json",
        "multi_turn_composite": "BFCL_v4_multi_turn_base.json",
        "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
        "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
        "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
        "multiple": "BFCL_v4_multiple.json",
        "parallel": "BFCL_v4_parallel.json",
        "parallel_multiple": "BFCL_v4_parallel_multiple.json",
        "simple_java": "BFCL_v4_simple_java.json",
        "simple_javascript": "BFCL_v4_simple_javascript.json",
        "simple_python": "BFCL_v4_simple_python.json",
        "web_search_base": "BFCL_v4_web_search.json",
        "web_search_no_snippet": "BFCL_v4_web_search.json",
    }

    def __init__(self, directory: str = "."):
        """
        Initialize the filter with a directory path.

        Args:
            directory: Path to the directory containing JSON files
        """
        self.directory = Path(directory)
        self.test_case_ids_file = self.directory / "test_case_ids_to_generate.json"

        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        if not self.test_case_ids_file.exists():
            raise ValueError(f"Test case IDs file not found: {self.test_case_ids_file}")

    def load_test_case_ids(self) -> Dict[str, List[str]]:
        """
        Load the test case IDs from test_case_ids_to_generate.json

        Returns:
            Dictionary mapping category names to lists of test case IDs
        """
        with open(self.test_case_ids_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def read_jsonl_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Read a JSONL file (one JSON object per line).

        Args:
            filepath: Path to the JSONL file

        Returns:
            List of dictionaries containing the parsed data
        """
        if not filepath.exists():
            print(f"Warning: File does not exist: {filepath}")
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()

            if not content:
                return []

            # Parse JSONL format
            records = []
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {filepath.name}: {e}")

            return records

    def write_jsonl_file(self, filepath: Path, records: List[Dict[str, Any]]):
        """
        Write records to a JSONL file.

        Args:
            filepath: Path to output file
            records: List of records to write
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def filter_records(self, records: List[Dict[str, Any]], allowed_ids: Set[str]) -> List[Dict[str, Any]]:
        """
        Filter records to only include those with IDs in the allowed set.

        Args:
            records: List of all records
            allowed_ids: Set of allowed test case IDs

        Returns:
            Filtered list of records
        """
        filtered = []
        for record in records:
            if "id" in record and record["id"] in allowed_ids:
                filtered.append(record)
        return filtered

    def process_category(self, category: str, test_case_ids: List[str]) -> Dict[str, Any]:
        """
        Process a single category by filtering its data file.

        Args:
            category: Category name from test_case_ids_to_generate.json
            test_case_ids: List of test case IDs to keep

        Returns:
            Dictionary with processing results
        """
        result = {
            "category": category,
            "filename": self.FILE_MAPPING.get(category, f"BFCL_v4_{category}.json"),
            "target_count": len(test_case_ids),
            "original_count": 0,
            "filtered_count": 0,
            # possible_answer mirror stats (if present)
            "answer_original_count": 0,
            "answer_filtered_count": 0,
            "success": False,
            "message": "",
        }

        # Skip if no IDs specified
        if not test_case_ids:
            result["message"] = "No test case IDs specified (empty list)"
            result["success"] = True
            return result

        # Get file path
        filepath = self.directory / result["filename"]

        if not filepath.exists():
            result["message"] = f"File not found: {filepath}"
            return result

        # Read original data
        original_records = self.read_jsonl_file(filepath)
        result["original_count"] = len(original_records)

        if not original_records:
            result["message"] = "Original file is empty"
            return result

        # Filter records
        allowed_ids = set(test_case_ids)
        filtered_records = self.filter_records(original_records, allowed_ids)
        result["filtered_count"] = len(filtered_records)

        # Write filtered data back
        try:
            self.write_jsonl_file(filepath, filtered_records)
            result["success"] = True
            result["message"] = f'Successfully filtered from {result["original_count"]} to {result["filtered_count"]} records'
        except Exception as e:
            result["message"] = f"Error writing file: {e}"

        # Also process possible_answer mirror if it exists
        answers_dir = self.directory / "possible_answer"
        ans_filepath = answers_dir / result["filename"]
        if answers_dir.exists() and ans_filepath.exists():
            answer_records = self.read_jsonl_file(ans_filepath)
            result["answer_original_count"] = len(answer_records)

            if answer_records:
                filtered_answer_records = self.filter_records(answer_records, allowed_ids)
                result["answer_filtered_count"] = len(filtered_answer_records)
                try:
                    self.write_jsonl_file(ans_filepath, filtered_answer_records)
                except Exception as e:
                    # Append message but do not fail overall category
                    result["message"] += f"; Answers write error: {e}"

        return result

    def process_all_categories(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Process all categories from test_case_ids_to_generate.json

        Args:
            dry_run: If True, only report what would be done without modifying files

        Returns:
            Dictionary with overall results
        """
        test_case_ids = self.load_test_case_ids()

        results = {"total_categories": len(test_case_ids), "processed_categories": 0, "total_original_records": 0, "total_filtered_records": 0, "total_target_records": 0, "category_results": []}

        print("=" * 80)
        print("FILTERING BFCL DATA FILES BY TEST CASE IDS")
        print("=" * 80)
        if dry_run:
            print("\n*** DRY RUN MODE - No files will be modified ***\n")

        for category, ids in sorted(test_case_ids.items()):
            print(f"\nProcessing category: {category}")
            print(f"  Target IDs: {len(ids)}")

            if dry_run:
                # In dry run mode, just read and report
                filepath = self.directory / self.FILE_MAPPING.get(category, f"BFCL_v4_{category}.json")
                if filepath.exists():
                    original_records = self.read_jsonl_file(filepath)
                    allowed_ids = set(ids)
                    filtered_records = self.filter_records(original_records, allowed_ids)
                    print(f"  Original records: {len(original_records)}")
                    print(f"  Would filter to: {len(filtered_records)} records")
                else:
                    print(f"  File not found: {filepath}")
                # Also report for possible_answer mirror
                ans_filepath = self.directory / "possible_answer" / self.FILE_MAPPING.get(category, f"BFCL_v4_{category}.json")
                if ans_filepath.exists():
                    answer_records = self.read_jsonl_file(ans_filepath)
                    filtered_answer_records = self.filter_records(answer_records, allowed_ids)
                    print(f"  Answers original: {len(answer_records)}")
                    print(f"  Answers would filter to: {len(filtered_answer_records)} records")
            else:
                # Actually process the category
                result = self.process_category(category, ids)
                results["category_results"].append(result)

                if result["success"]:
                    results["processed_categories"] += 1
                    results["total_original_records"] += result["original_count"]
                    results["total_filtered_records"] += result["filtered_count"]
                    results["total_target_records"] += result["target_count"]

                print(f"  Original records: {result['original_count']}")
                print(f"  Filtered records: {result['filtered_count']}")
                print(f"  Status: {result['message']}")
                # Print possible_answer mirror stats when available
                if result.get("answer_original_count", 0) or result.get("answer_filtered_count", 0):
                    print(f"  Answers original: {result['answer_original_count']}")
                    print(f"  Answers filtered: {result['answer_filtered_count']}")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total categories: {results['total_categories']}")
        print(f"Successfully processed: {results['processed_categories']}")
        print(f"Total original records: {results['total_original_records']}")
        print(f"Total filtered records: {results['total_filtered_records']}")
        print(f"Total target records: {results['total_target_records']}")

        if not dry_run:
            print(f"\nReduction: {results['total_original_records'] - results['total_filtered_records']} records removed")

        return results


def main():
    """Main function to run the filtering script."""
    import argparse

    parser = argparse.ArgumentParser(description="Filter BFCL JSON data files based on test_case_ids_to_generate.json", formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--directory", type=str, default=".", help="Directory containing JSON files (default: current directory)")

    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying files")

    args = parser.parse_args()

    # Initialize filter
    filter_tool = BFCLDataFilter(args.directory)

    # Process all categories
    results = filter_tool.process_all_categories(dry_run=args.dry_run)

    if args.dry_run:
        print("\n*** DRY RUN COMPLETE - No files were modified ***")
    else:
        print("\n*** FILTERING COMPLETE ***")


if __name__ == "__main__":
    main()
