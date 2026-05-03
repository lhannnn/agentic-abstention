"""
Canonical dataset policy for the ACE agentic-abstention benchmark variant.

This fork no longer uses the full upstream AbstentionBench roster. The active
benchmark is the filtered dataset suite used for agentic abstention evaluation
under a Wikimedia-only SEARCH assumption.
"""

AGENTIC_BENCHMARK_DATASET_CONFIG_NAMES = (
    "alcuna",
    "bbq",
    "big_bench_disambiguate",
    "big_bench_known_unknowns",
    "coconot",
    "falseqa",
    "gpqa",
    "gsm8k",
    "kuq",
    "mediq",
    "mmlu_math",
    "moralchoice",
    "qaqa",
    "situated_qa",
    "umwp",
    "worldsense",
)

AGENTIC_BENCHMARK_DATASET_CONFIG_NAMES_SET = frozenset(
    AGENTIC_BENCHMARK_DATASET_CONFIG_NAMES
)

KUQ_ACTIVE_BENCHMARK_CATEGORIES = (
    "ambiguous",
    "controversial",
    "false assumption",
    "future unknown",
    "unsolved problem",
)

KUQ_ACTIVE_BENCHMARK_CATEGORIES_SET = frozenset(KUQ_ACTIVE_BENCHMARK_CATEGORIES)

KUQ_EXCLUDED_BENCHMARK_CATEGORIES = ("counterfactual",)

KUQ_EXCLUDED_BENCHMARK_CATEGORIES_SET = frozenset(KUQ_EXCLUDED_BENCHMARK_CATEGORIES)

COCONOT_INCLUDED_ABSTENTIONBENCH_CATEGORIES = (
    "False presumptions",
    "Humanizing",
    "Incomprehensible",
    "Subjective",
    "Unknowns",
    "Unsupported",
)

COCONOT_INCLUDED_ABSTENTIONBENCH_CATEGORIES_SET = frozenset(
    COCONOT_INCLUDED_ABSTENTIONBENCH_CATEGORIES
)

COCONOT_EXCLUDED_ABSTENTIONBENCH_CATEGORIES = (
    "Temporal",
    "Underspecification",
    "Safety",
)

COCONOT_EXCLUDED_ABSTENTIONBENCH_CATEGORIES_SET = frozenset(
    COCONOT_EXCLUDED_ABSTENTIONBENCH_CATEGORIES
)

AGENTIC_ALLOWED_DATASET_NAME_EXTENDED = (
    "ALCUNADataset",
    "BBQDataset",
    "BigBenchDisambiguateDataset",
    "BigBenchKnownUnknownsDataset",
    "CoCoNotDataset_false_presumptions",
    "CoCoNotDataset_humanizing",
    "CoCoNotDataset_incomprehensible",
    "CoCoNotDataset_subjective",
    "CoCoNotDataset_unknowns",
    "CoCoNotDataset_unsupported",
    "FalseQADataset",
    "GPQA",
    "GSM8K",
    "KUQDataset_ambiguous",
    "KUQDataset_controversial",
    "KUQDataset_false_assumption",
    "KUQDataset_future_unknown",
    "KUQDataset_unsolved_problem",
    "MediQDataset",
    "MMLUMath",
    "MoralChoiceDataset",
    "QAQADataset",
    "SituatedQAGeoDataset",
    "UMWP",
    "WorldSenseDataset",
)

AGENTIC_ALLOWED_DATASET_NAME_EXTENDED_SET = frozenset(
    AGENTIC_ALLOWED_DATASET_NAME_EXTENDED
)
