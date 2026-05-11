"""
// ML (https://github.com/stepanzorin/ml)
// Copyright Text: 2026 Stepan Zorin <stz.hom@gmail.com>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

TaskType = Literal[
    "regression",
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "unsupervised",
]
ColumnKind = Literal["numeric", "categorical", "date", "ignored", "target"]
CategoricalEncoding = Literal["one_hot", "top_k_one_hot"]
NumericMissingStrategy = Literal["error", "mean", "zero"]
DateEncoding = Literal["components", "days_since_epoch"]

MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "-"}
MISSING_CATEGORY = "__MISSING__"
OTHER_CATEGORY = "__OTHER__"

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)


@dataclass(frozen=True)
class Options:
    input_csv: Path
    output_json: Path
    task: TaskType
    target_column: str | None
    target_columns: list[str]
    delimiter: str | None
    encoding: str
    has_header: bool
    categorical_columns: set[str]
    numeric_columns: set[str]
    date_columns: set[str]
    ignored_columns: set[str]
    detect_sample_size: int
    max_one_hot_cardinality: int
    top_k: int
    numeric_missing: NumericMissingStrategy
    date_encoding: DateEncoding
    train_ratio: float | None
    shuffle: bool
    seed: int


@dataclass
class NumericSpec:
    column: str
    missing_strategy: str
    fill_value: float | None


@dataclass
class CategoricalSpec:
    column: str
    encoding: str
    categories: list[str]
    missing_category: str
    other_category: str | None


@dataclass
class DateSpec:
    column: str
    encoding: str
    missing_strategy: str


@dataclass
class FeatureInfo:
    name: str
    source_column: str
    kind: str


RawRow = dict[str, str]
RawRows = list[RawRow]


def split_csv_arg(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def split_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_missing(value: str) -> bool:
    return trim(value).lower() in MISSING_TOKENS


def try_parse_float(value: str) -> float | None:
    value = trim(value)
    if is_missing(value):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def try_parse_date(value: str) -> date | None:
    value = trim(value)
    if is_missing(value):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def sniff_delimiter(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, newline="") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def ensure_unique_columns(columns: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for index, raw_name in enumerate(columns):
        name = trim(raw_name) or f"column_{index}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        result.append(name if count == 0 else f"{name}__{count + 1}")
    return result


def read_csv_rows(path: Path, *, delimiter: str | None, encoding: str, has_header: bool) -> tuple[list[str], RawRows, str]:
    actual_delimiter = delimiter or sniff_delimiter(path, encoding)

    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=actual_delimiter)

        if has_header:
            try:
                raw_header = next(reader)
            except StopIteration as exc:
                raise ValueError("CSV file is empty") from exc
            columns = ensure_unique_columns(raw_header)
            start_line = 2
        else:
            all_rows = [[trim(cell) for cell in row] for row in reader if row and not all(is_missing(cell) for cell in row)]
            if not all_rows:
                raise ValueError("CSV file is empty")
            width = max(len(row) for row in all_rows)
            columns = [f"column_{i}" for i in range(width)]
            rows: RawRows = []
            for raw_row in all_rows:
                padded = raw_row + [""] * (width - len(raw_row))
                rows.append({columns[i]: padded[i] for i in range(width)})
            return columns, rows, actual_delimiter

        rows = []
        for line_number, raw_row in enumerate(reader, start=start_line):
            if not raw_row or all(is_missing(cell) for cell in raw_row):
                continue
            if len(raw_row) > len(columns):
                raise ValueError(f"Row {line_number} has {len(raw_row)} cells, but header has {len(columns)} columns")
            padded = [trim(cell) for cell in raw_row] + [""] * (len(columns) - len(raw_row))
            rows.append({columns[i]: padded[i] for i in range(len(columns))})

    return columns, rows, actual_delimiter


def detect_column_kind(values: Sequence[str], sample_limit: int) -> ColumnKind:
    detected: ColumnKind | None = None
    checked = 0

    for raw in values:
        value = trim(raw)
        if is_missing(value):
            continue

        if try_parse_date(value) is not None:
            current: ColumnKind = "date"
        elif try_parse_float(value) is not None:
            current = "numeric"
        else:
            return "categorical"

        if detected is None:
            detected = current
        elif detected != current:
            return "categorical"

        checked += 1
        if checked >= sample_limit:
            break

    return detected or "categorical"


def infer_kinds(columns: list[str], rows: RawRows, options: Options) -> OrderedDict[str, ColumnKind]:
    target_columns = set()
    if options.target_column:
        target_columns.add(options.target_column)
    target_columns.update(options.target_columns)

    kinds: OrderedDict[str, ColumnKind] = OrderedDict()
    for column in columns:
        if column in target_columns:
            kinds[column] = "target"
        elif column in options.ignored_columns:
            kinds[column] = "ignored"
        elif column in options.numeric_columns:
            kinds[column] = "numeric"
        elif column in options.categorical_columns:
            kinds[column] = "categorical"
        elif column in options.date_columns:
            kinds[column] = "date"
        else:
            kinds[column] = detect_column_kind([row.get(column, "") for row in rows], options.detect_sample_size)
    return kinds


def split_fit_rows(rows: RawRows, options: Options) -> tuple[RawRows, RawRows, RawRows, dict[str, Any]]:
    ordered_rows = list(rows)
    if options.shuffle:
        rng = random.Random(options.seed)
        rng.shuffle(ordered_rows)

    if options.train_ratio is None:
        return ordered_rows, ordered_rows, [], {"has_split": False}

    if not (0.0 < options.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")

    train_count = int(len(ordered_rows) * options.train_ratio)
    if train_count <= 0 or train_count >= len(ordered_rows):
        raise ValueError("--train-ratio creates an empty train or test split")

    train_rows = ordered_rows[:train_count]
    test_rows = ordered_rows[train_count:]
    return train_rows, train_rows, test_rows, {
        "has_split": True,
        "train_ratio": options.train_ratio,
        "shuffle": options.shuffle,
        "seed": options.seed,
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
    }


def fit_numeric_specs(columns: list[str], fit_rows: RawRows, kinds: OrderedDict[str, ColumnKind], options: Options) -> dict[str, NumericSpec]:
    specs: dict[str, NumericSpec] = {}
    for column in columns:
        if kinds[column] != "numeric":
            continue
        values = [try_parse_float(row.get(column, "")) for row in fit_rows]
        present = [v for v in values if v is not None]
        if options.numeric_missing == "mean":
            fill_value = statistics.fmean(present) if present else 0.0
        elif options.numeric_missing == "zero":
            fill_value = 0.0
        else:
            fill_value = None
        specs[column] = NumericSpec(column=column, missing_strategy=options.numeric_missing, fill_value=fill_value)
    return specs


def fit_categorical_specs(columns: list[str], fit_rows: RawRows, kinds: OrderedDict[str, ColumnKind], options: Options) -> dict[str, CategoricalSpec]:
    specs: dict[str, CategoricalSpec] = {}
    for column in columns:
        if kinds[column] != "categorical":
            continue
        values = []
        for row in fit_rows:
            value = trim(row.get(column, ""))
            values.append(MISSING_CATEGORY if is_missing(value) else value)
        counts = Counter(values)
        sorted_categories = sorted(counts, key=lambda x: (-counts[x], x))

        if len(sorted_categories) <= options.max_one_hot_cardinality:
            categories = sorted(sorted_categories)
            encoding: CategoricalEncoding = "one_hot"
            other: str | None = None
        else:
            categories = sorted(sorted_categories[: options.top_k])
            if OTHER_CATEGORY not in categories:
                categories.append(OTHER_CATEGORY)
            encoding = "top_k_one_hot"
            other = OTHER_CATEGORY

        if MISSING_CATEGORY in counts and MISSING_CATEGORY not in categories:
            categories.append(MISSING_CATEGORY)

        specs[column] = CategoricalSpec(
            column=column,
            encoding=encoding,
            categories=categories,
            missing_category=MISSING_CATEGORY,
            other_category=other,
        )
    return specs


def fit_date_specs(columns: list[str], kinds: OrderedDict[str, ColumnKind], options: Options) -> dict[str, DateSpec]:
    return {
        column: DateSpec(column=column, encoding=options.date_encoding, missing_strategy="error")
        for column in columns
        if kinds[column] == "date"
    }


def encode_numeric(column: str, raw_value: str, spec: NumericSpec) -> float:
    parsed = try_parse_float(raw_value)
    if parsed is not None:
        return parsed
    if spec.missing_strategy == "error":
        raise ValueError(f"Missing or invalid numeric value in column '{column}': {raw_value!r}")
    assert spec.fill_value is not None
    return spec.fill_value


def encode_categorical(raw_value: str, spec: CategoricalSpec) -> list[float]:
    value = trim(raw_value)
    value = spec.missing_category if is_missing(value) else value
    if value not in spec.categories:
        if spec.other_category and spec.other_category in spec.categories:
            value = spec.other_category
        else:
            # For normal one-hot without OTHER: unknown category becomes all zeros.
            return [0.0 for _ in spec.categories]
    return [1.0 if category == value else 0.0 for category in spec.categories]


def encode_date(column: str, raw_value: str, spec: DateSpec) -> list[float]:
    parsed = try_parse_date(raw_value)
    if parsed is None:
        raise ValueError(f"Missing or invalid date value in column '{column}': {raw_value!r}")
    if spec.encoding == "days_since_epoch":
        return [float((parsed - date(1970, 1, 1)).days)]
    month_angle = 2.0 * math.pi * (parsed.month - 1) / 12.0
    weekday_angle = 2.0 * math.pi * parsed.weekday() / 7.0
    return [
        float(parsed.year),
        math.sin(month_angle),
        math.cos(month_angle),
        math.sin(weekday_angle),
        math.cos(weekday_angle),
    ]


def build_feature_infos(columns: list[str], kinds: OrderedDict[str, ColumnKind], categorical_specs: dict[str, CategoricalSpec], date_specs: dict[str, DateSpec]) -> list[FeatureInfo]:
    infos: list[FeatureInfo] = []
    for column in columns:
        kind = kinds[column]
        if kind in {"target", "ignored"}:
            continue
        if kind == "numeric":
            infos.append(FeatureInfo(name=column, source_column=column, kind="numeric"))
        elif kind == "categorical":
            for category in categorical_specs[column].categories:
                infos.append(FeatureInfo(name=f"{column}={category}", source_column=column, kind="one_hot"))
        elif kind == "date":
            if date_specs[column].encoding == "days_since_epoch":
                infos.append(FeatureInfo(name=f"{column}:days_since_epoch", source_column=column, kind="date_numeric"))
            else:
                for suffix in ("year", "month_sin", "month_cos", "weekday_sin", "weekday_cos"):
                    infos.append(FeatureInfo(name=f"{column}:{suffix}", source_column=column, kind="date_component"))
    return infos


def encode_features(row: RawRow, columns: list[str], kinds: OrderedDict[str, ColumnKind], numeric_specs: dict[str, NumericSpec], categorical_specs: dict[str, CategoricalSpec], date_specs: dict[str, DateSpec]) -> list[float]:
    features: list[float] = []
    for column in columns:
        kind = kinds[column]
        if kind in {"target", "ignored"}:
            continue
        raw_value = row.get(column, "")
        if kind == "numeric":
            features.append(encode_numeric(column, raw_value, numeric_specs[column]))
        elif kind == "categorical":
            features.extend(encode_categorical(raw_value, categorical_specs[column]))
        elif kind == "date":
            features.extend(encode_date(column, raw_value, date_specs[column]))
        else:
            raise ValueError(f"Unsupported column kind for '{column}': {kind}")
    return features


def build_class_mapping(values: list[str], task: TaskType) -> tuple[dict[str, int], list[str]]:
    normalized = [trim(v) for v in values if not is_missing(trim(v))]
    if not normalized:
        raise ValueError("Cannot build class mapping from empty target values")

    unique = sorted(set(normalized))

    if task == "binary_classification":
        if set(unique).issubset({"0", "1"}):
            id_to_class = ["0", "1"]
            return {"0": 0, "1": 1}, id_to_class
        if len(unique) != 2:
            raise ValueError(f"Binary classification target must have exactly 2 classes, got {len(unique)}: {unique}")

    class_to_id = {label: i for i, label in enumerate(unique)}
    return class_to_id, unique


def build_target_info(options: Options, fit_rows: RawRows) -> tuple[dict[str, Any], Any]:
    if options.task == "unsupervised":
        return {"mode": "none"}, None

    if options.task == "regression":
        assert options.target_column is not None
        return {"mode": "numeric", "column": options.target_column}, None

    if options.task in {"binary_classification", "multiclass_classification"}:
        assert options.target_column is not None
        class_to_id, id_to_class = build_class_mapping([row.get(options.target_column, "") for row in fit_rows], options.task)
        return {
            "mode": "class_id",
            "column": options.target_column,
            "class_to_id": class_to_id,
            "id_to_class": id_to_class,
        }, class_to_id

    if options.task == "multilabel_classification":
        return {"mode": "multilabel", "columns": options.target_columns}, None

    raise ValueError(f"Unsupported task: {options.task}")


def encode_target(row: RawRow, options: Options, target_info: dict[str, Any], target_aux: Any) -> dict[str, Any] | None:
    if options.task == "unsupervised":
        return None

    if options.task == "regression":
        assert options.target_column is not None
        value = try_parse_float(row.get(options.target_column, ""))
        if value is None:
            raise ValueError(f"Missing or invalid numeric target in column '{options.target_column}'")
        return {"numeric": value}

    if options.task in {"binary_classification", "multiclass_classification"}:
        assert options.target_column is not None
        raw = trim(row.get(options.target_column, ""))
        if is_missing(raw):
            raise ValueError(f"Missing class target in column '{options.target_column}'")
        class_to_id: dict[str, int] = target_aux
        if raw not in class_to_id:
            raise ValueError(f"Unknown target class {raw!r} in column '{options.target_column}'")
        return {"class_id": class_to_id[raw]}

    if options.task == "multilabel_classification":
        labels: list[float] = []
        for column in options.target_columns:
            value = try_parse_float(row.get(column, ""))
            if value is None:
                raise ValueError(f"Missing or invalid multilabel target in column '{column}'")
            if value not in (0.0, 1.0):
                raise ValueError(f"Multilabel target column '{column}' must contain 0/1 values, got {value}")
            labels.append(value)
        return {"labels": labels}

    raise ValueError(f"Unsupported task: {options.task}")


def transform_rows(rows: RawRows, columns: list[str], kinds: OrderedDict[str, ColumnKind], numeric_specs: dict[str, NumericSpec], categorical_specs: dict[str, CategoricalSpec], date_specs: dict[str, DateSpec], options: Options, target_info: dict[str, Any], target_aux: Any) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        sample = {"features": encode_features(row, columns, kinds, numeric_specs, categorical_specs, date_specs)}
        target = encode_target(row, options, target_info, target_aux)
        if target is not None:
            sample["target"] = target
        samples.append(sample)
    return samples


def preprocess(options: Options) -> dict[str, Any]:
    columns, rows, delimiter = read_csv_rows(options.input_csv, delimiter=options.delimiter, encoding=options.encoding, has_header=options.has_header)

    if not rows:
        raise ValueError("CSV has no data rows")

    if options.task != "unsupervised":
        if options.task == "multilabel_classification":
            missing = [name for name in options.target_columns if name not in columns]
            if missing:
                raise ValueError(f"Target columns not found: {missing}")
        else:
            if options.target_column not in columns:
                raise ValueError(f"Target column not found: {options.target_column!r}")

    train_fit_rows, train_rows, test_rows, split = split_fit_rows(rows, options)
    kinds = infer_kinds(columns, train_fit_rows, options)
    numeric_specs = fit_numeric_specs(columns, train_fit_rows, kinds, options)
    categorical_specs = fit_categorical_specs(columns, train_fit_rows, kinds, options)
    date_specs = fit_date_specs(columns, kinds, options)
    feature_infos = build_feature_infos(columns, kinds, categorical_specs, date_specs)
    target_info, target_aux = build_target_info(options, train_fit_rows)

    all_samples = transform_rows(train_rows + test_rows, columns, kinds, numeric_specs, categorical_specs, date_specs, options, target_info, target_aux)
    train_samples = transform_rows(train_rows, columns, kinds, numeric_specs, categorical_specs, date_specs, options, target_info, target_aux) if split.get("has_split") else []
    test_samples = transform_rows(test_rows, columns, kinds, numeric_specs, categorical_specs, date_specs, options, target_info, target_aux) if split.get("has_split") else []

    result: dict[str, Any] = {
        "format_version": 3,
        "task_type": options.task,
        "source_csv": str(options.input_csv),
        "target": target_info,
        "row_count": len(all_samples),
        "feature_count": len(feature_infos),
        "columns": columns,
        "column_kinds": dict(kinds),
        "feature_names": [info.name for info in feature_infos],
        "features": [asdict(info) for info in feature_infos],
        "preprocessing": {
            "delimiter": delimiter,
            "numeric": {name: asdict(spec) for name, spec in numeric_specs.items()},
            "categorical": {name: asdict(spec) for name, spec in categorical_specs.items()},
            "date": {name: asdict(spec) for name, spec in date_specs.items()},
            "missing_tokens": sorted(MISSING_TOKENS),
        },
        "samples": all_samples,
        "split": split,
    }

    if split.get("has_split"):
        result["train_samples"] = train_samples
        result["test_samples"] = test_samples

    return result


def parse_args() -> Options:
    parser = argparse.ArgumentParser(description="Preprocess a CSV file into ML-ready JSON for C++ tabular ML code.")
    parser.add_argument("input_csv", type=Path, help="Absolute or relative path to input CSV file")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSON file path")
    parser.add_argument("--task", choices=("regression", "binary_classification", "multiclass_classification", "multilabel_classification", "unsupervised"), default="regression")
    parser.add_argument("--target", default=None, help="Target column for regression/binary/multiclass tasks")
    parser.add_argument("--targets", default="", help="Comma-separated target columns for multilabel_classification")
    parser.add_argument("--delimiter", default=None, help="CSV delimiter. If omitted, the tool tries to auto-detect it")
    parser.add_argument("--encoding", default="utf-8", help="Input CSV encoding, default: utf-8")
    parser.add_argument("--no-header", action="store_true", help="Use generated column names: column_0, column_1, ...")
    parser.add_argument("--categorical", default="", help="Comma-separated columns forced to categorical")
    parser.add_argument("--numeric", default="", help="Comma-separated columns forced to numeric")
    parser.add_argument("--date", default="", help="Comma-separated columns forced to date")
    parser.add_argument("--ignore", default="", help="Comma-separated columns ignored from features")
    parser.add_argument("--detect-sample-size", type=int, default=50, help="Non-empty values checked per column for type detection")
    parser.add_argument("--max-one-hot-cardinality", type=int, default=20, help="If unique categories <= this value, normal one-hot is used")
    parser.add_argument("--top-k", type=int, default=20, help="Top K categories kept for high-cardinality columns")
    parser.add_argument("--numeric-missing", choices=("error", "mean", "zero"), default="error")
    parser.add_argument("--date-encoding", choices=("components", "days_since_epoch"), default="components")
    parser.add_argument("--train-ratio", type=float, default=None, help="Optional train/test split ratio, e.g. 0.8")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows before optional split/output")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    output_json = args.output.expanduser().resolve() if args.output else input_csv.with_suffix(".ml.json")

    if args.detect_sample_size <= 0:
        raise ValueError("--detect-sample-size must be positive")
    if args.max_one_hot_cardinality <= 0:
        raise ValueError("--max-one-hot-cardinality must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.train_ratio is not None and not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")

    target_columns = split_csv_list(args.targets)
    target_column = args.target

    if args.task in {"regression", "binary_classification", "multiclass_classification"} and not target_column:
        raise ValueError(f"--target is required for task {args.task}")
    if args.task == "multilabel_classification" and not target_columns:
        raise ValueError("--targets is required for multilabel_classification")
    if args.task == "unsupervised":
        target_column = None
        target_columns = []

    return Options(
        input_csv=input_csv,
        output_json=output_json,
        task=args.task,
        target_column=target_column,
        target_columns=target_columns,
        delimiter=args.delimiter,
        encoding=args.encoding,
        has_header=not args.no_header,
        categorical_columns=split_csv_arg(args.categorical),
        numeric_columns=split_csv_arg(args.numeric),
        date_columns=split_csv_arg(args.date),
        ignored_columns=split_csv_arg(args.ignore),
        detect_sample_size=args.detect_sample_size,
        max_one_hot_cardinality=args.max_one_hot_cardinality,
        top_k=args.top_k,
        numeric_missing=args.numeric_missing,
        date_encoding=args.date_encoding,
        train_ratio=args.train_ratio,
        shuffle=args.shuffle,
        seed=args.seed,
    )


def main() -> None:
    options = parse_args()
    result = preprocess(options)

    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    with options.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote: {options.output_json}")
    print(f"Task: {result['task_type']}")
    print(f"Rows: {result['row_count']}")
    print(f"Features: {result['feature_count']}")
    if result.get("split", {}).get("has_split"):
        print(f"Train rows: {result['split']['train_row_count']}")
        print(f"Test rows: {result['split']['test_row_count']}")


if __name__ == "__main__":
    main()
