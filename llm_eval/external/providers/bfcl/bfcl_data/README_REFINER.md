# BFCL Data Refiner

A comprehensive Python script for refining and processing JSON data files in the BFCL (Berkeley Function Calling Leaderboard) dataset.

## Features

- **Multi-format Support**: Handles both JSONL (JSON Lines) and regular JSON formats
- **Data Validation**: Validates JSON structure against expected schema
- **Key Normalization**: Rename and normalize keys across different data files
- **Format Conversion**: Convert between JSONL and JSON formats
- **Comprehensive Analysis**: Generate detailed reports on data structure and quality
- **Batch Processing**: Process all JSON files in a directory at once

## Installation

No additional dependencies required. The script uses only Python standard library modules:
- `json`
- `pathlib`
- `typing`
- `collections`
- `argparse`

## Usage

### Basic Analysis

Analyze all JSON files in the current directory:

```bash
python temp.py --analyze
```

Analyze files in a specific directory:

```bash
python temp.py --directory /path/to/data --analyze
```

### Refining Files

Refine all files to JSONL format (one JSON object per line):

```bash
python temp.py --refine
```

Refine to regular JSON format with pretty printing:

```bash
python temp.py --refine --format json
```

Save refined files to a different directory:

```bash
python temp.py --refine --output-dir ./refined_data
```

### Export Summary Report

Generate a detailed JSON report of all data files:

```bash
python temp.py --export-report
```

This creates a `data_summary_report.json` file with:
- Total files and records count
- Empty files list
- Files with validation errors
- Key distribution across all files
- Type distribution for each key
- Detailed information for each file

### Combined Operations

You can combine multiple operations:

```bash
python temp.py --analyze --refine --export-report
```

## Data Structure

### Expected Schema

#### Standard Keys (required in all records)
- `id` (str): Unique identifier for the test case
- `question` (list): List of question objects

#### Optional Keys
- `function` (list): Function definitions (in non-multi-turn files)
- `initial_config` (dict): Initial configuration (in multi-turn files)
- `path` (list): Execution path (in multi-turn files)
- `involved_classes` (list): Classes involved (in multi-turn files)
- `missed_function` (dict): Missed function information
- `ground_truth` (str/list): Expected correct answer
- `execution_result` (str/list/dict): Execution results
- `possible_answer` (str/list): Possible valid answers

### File Types

The script handles several types of BFCL data files:

1. **Simple Function Calling** (`BFCL_v4_simple_*.json`)
   - Single function call tests
   - Keys: `id`, `question`, `function`

2. **Multiple Function Calling** (`BFCL_v4_multiple.json`)
   - Multiple function call tests
   - Keys: `id`, `question`, `function`

3. **Multi-turn Conversations** (`BFCL_v4_multi_turn_*.json`)
   - Multi-turn dialogue tests
   - Keys: `id`, `question`, `initial_config`, `path`, `involved_classes`
   - Note: Does not have `function` key

4. **Live Tests** (`BFCL_v4_live_*.json`)
   - Live API interaction tests
   - Keys: `id`, `question`, `function`

5. **Irrelevance Tests** (`BFCL_v4_irrelevance.json`)
   - Tests for handling irrelevant queries
   - Keys: `id`, `question`, `function`

## Example Output

### Analysis Output

```
================================================================================
ANALYZING BFCL DATA FILES
================================================================================

Total files: 20
Total records: 515

Empty files (7):
  - BFCL_v4_format_sensitivity.json
  - BFCL_v4_live_parallel.json
  ...

Files with errors (4):
  - BFCL_v4_multi_turn_base.json
      Record 0: Missing required key: 'function'
      ...

Key distribution:
  id: 515 occurrences
  question: 515 occurrences
  function: 256 occurrences
  initial_config: 259 occurrences
  ...

Type distribution by key:
  function:
    - list: 256
  id:
    - str: 515
  ...
```

## Programmatic Usage

You can also use the script as a Python module:

```python
from pathlib import Path
from temp import BFCLDataRefiner

# Initialize refiner
refiner = BFCLDataRefiner('/path/to/bfcl_data')

# Analyze files
analysis = refiner.analyze_all_files()
print(f"Total records: {analysis['total_records']}")

# Get all JSON files
json_files = refiner.get_json_files()

# Read a specific file
records = refiner.read_json_file(json_files[0])

# Validate a record
issues = refiner.validate_record(records[0], 'test.json')

# Refine a single file
success = refiner.refine_file(
    json_files[0],
    output_path=Path('output.jsonl'),
    format_type='jsonl'
)

# Normalize keys
key_mapping = {'old_key': 'new_key'}
normalized = refiner.normalize_keys(records[0], key_mapping)

# Export report
refiner.export_summary_report('my_report.json')
```

## Advanced Features

### Custom Key Mapping

To rename keys during refinement, modify the `refine_file` or `refine_all_files` methods:

```python
refiner = BFCLDataRefiner('.')
key_mapping = {
    'old_field_name': 'new_field_name',
    'deprecated_key': 'updated_key'
}
refiner.refine_all_files(
    output_dir='./normalized',
    key_mapping=key_mapping
)
```

### Filtering Files

The script automatically excludes certain files (like `temp.py` and `test_case_ids_to_generate.json`). You can customize this:

```python
json_files = refiner.get_json_files(
    exclude_patterns=['temp', 'backup', 'old']
)
```

## Troubleshooting

### Empty Files

Empty files are reported but not processed. They are listed in the analysis output under "Empty files".

### Validation Errors

Files with missing required keys or incorrect types are flagged. The script reports the first 3 errors per file, with a count of additional errors.

### Format Detection

The script automatically detects whether a file is JSONL or regular JSON:
- Files with multiple lines are parsed as JSONL
- Files with nested structures are parsed as regular JSON
- Special handling for format_sensitivity.json which has a category-based structure

## Contributing

To extend the script with new features:

1. Add new validation rules in `validate_record()`
2. Add new key types in `STANDARD_KEYS` or `OPTIONAL_KEYS`
3. Add new export formats in `refine_file()`
4. Add new analysis metrics in `analyze_all_files()`

## License

This script is part of the BFCL (Berkeley Function Calling Leaderboard) project.


