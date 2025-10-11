from llm_eval.datasets import load_datasets


def test_halluLens_dataset_loading():
    # Load HalluLens dataset
    loader = load_datasets(name="halluLens", split="test")
    all_samples_path = loader.load()

    assert len(all_samples_path) == 4

    # Test subset keys
    expected_keys = ['precise_wikiqa', 'longwiki']
    subset_loader = load_datasets(name="halluLens", subset=expected_keys, split="test")
    subset_samples_path = subset_loader.load()

    assert set(subset_samples_path.keys()) == set(expected_keys)


def test_halluLens_precise_wikiqa_runner():
    loader = load_datasets(name="halluLens", split="test")
    all_samples_path = loader.load()

    from llm_eval.external.providers.hallulens.hallulens_runner import precise_wikiqa_runner
    precise_wikiqa_runner(
        qa_dataset_path=all_samples_path['precise_wikiqa'],
        model="gpt-3.5-turbo",
        inference_method="openai",
        inf_batch_size=8,
    )


def test_halluLens_longwiki_qa_runner():
    loader = load_datasets(name="halluLens", split="test")
    all_samples_path = loader.load()

    from llm_eval.external.providers.hallulens.hallulens_runner import longwiki_runner
    longwiki_runner(benchmark_dataset_path=all_samples_path['longwiki'],
                    model="gpt-3.5-turbo",
                    inference_method="openai",
                    max_workers=10,
                    )


def test_halluLens_mixed_entity_refusal_runner():
    loader = load_datasets(name="halluLens", split="test")
    all_samples_path = loader.load()

    from llm_eval.external.providers.hallulens.hallulens_runner import non_mixed_entity_runner
    non_mixed_entity_runner(
        tested_model='meta-llama/Llama-3.1-8B-Instruct',
        prompt_path=all_samples_path['mixed_entities'],
    )


def test_halluLens_generated_entity_refusal_runner():
    loader = load_datasets(name="halluLens", split="test")
    all_samples_path = loader.load()

    from llm_eval.external.providers.hallulens.hallulens_runner import non_generated_entity_runner
    non_generated_entity_runner(
        prompt_path=all_samples_path['generated_entities'],
    )
