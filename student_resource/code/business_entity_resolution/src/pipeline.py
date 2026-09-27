"""Offline, reproducible business entity-resolution pipeline.

All commands operate exclusively on the challenge TSVs.  Indexes, models and
reports belong in ``artifacts/`` (which is intentionally ignored by Git).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

DELIM = "\t"
FIELDS = ("entity_id", "business_name", "business_address", "country")
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "llp", "ltd", "limited", "corp",
    "corporation", "company", "co", "pvt", "private", "plc", "gmbh",
    "sa", "sas", "sarl", "bv", "lp",
}
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)\d[\d\-\/]{0,11}(?!\w)")


@dataclass(frozen=True)
class Record:
    entity_id: str
    name: str
    address: str
    country: str


@dataclass
class Candidate:
    entity_id: str
    routes: set[str]
    best_rank: int


def norm(text: str) -> str:
    """Conservative text view; raw text is never discarded."""
    text = unicodedata.normalize("NFKC", text or "").casefold().replace("&", " and ")
    return " ".join(TOKEN_RE.findall(text))


def tokens(text: str) -> list[str]:
    return [x for x in norm(text).split() if len(x) > 1]


def suffix_free_name(name: str) -> str:
    # One-character names/initials are weak retrieval keys but meaningful text;
    # retain them in the reversible normalisation view.
    return " ".join(x for x in norm(name).split() if x not in LEGAL_SUFFIXES)


def chars(text: str, n: int = 3, limit: int = 8) -> list[str]:
    value = f"  {norm(text).replace(' ', '_')}  "
    grams = sorted({value[i:i + n] for i in range(max(0, len(value) - n + 1))})
    if len(grams) <= limit:
        return grams
    # Deterministic spread avoids indexing every gram of long addresses.
    return [grams[round(i * (len(grams) - 1) / (limit - 1))] for i in range(limit)]


def numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(norm(text))[:3]


def prefix(text: str) -> str:
    value = suffix_free_name(text).replace(" ", "")
    return value[:5] if len(value) >= 5 else ""


def read_records(path: Path) -> Iterator[Record]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=DELIM)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: expected {FIELDS}, got {reader.fieldnames}")
        for row in reader:
            yield Record(*(row[k].strip() for k in FIELDS))


def read_truth(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=DELIM)
        if tuple(reader.fieldnames or ()) != ("source1_entity_id", "matched_entity_ids"):
            raise ValueError(f"{path}: invalid ground-truth header")
        for row in reader:
            result[row["source1_entity_id"]] = {
                x for x in row["matched_entity_ids"].split(",") if x
            }
    return result


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def macro_f05(truth: dict[str, set[str]], predicted: dict[str, set[str]]) -> float:
    """Organizer-defined per-S1 macro F0.5, including singleton convention."""
    scores = []
    for s1, actual in truth.items():
        guessed = predicted.get(s1, set())
        if not actual:
            scores.append(1.0 if not guessed else 0.0)
            continue
        tp = len(actual & guessed)
        if not guessed or not tp:
            scores.append(0.0)
            continue
        precision, recall = tp / len(guessed), tp / len(actual)
        scores.append(1.25 * precision * recall / (0.25 * precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def profile_file(path: Path) -> dict[str, object]:
    rows = missing_name = missing_address = 0
    countries, name_lengths, address_lengths = Counter(), [], []
    for record in read_records(path):
        rows += 1
        countries[record.country] += 1
        if record.name:
            name_lengths.append(len(record.name))
        else:
            missing_name += 1
        if record.address:
            address_lengths.append(len(record.address))
        else:
            missing_address += 1
    return {
        "file": str(path), "rows": rows, "countries": dict(countries),
        "missing_name": missing_name, "missing_address": missing_address,
        "mean_name_length": statistics.mean(name_lengths) if name_lengths else 0,
        "mean_address_length": statistics.mean(address_lengths) if address_lengths else 0,
    }


def cmd_eda(args: argparse.Namespace) -> None:
    root = Path(args.data_dir)
    report = {"files": []}
    for split in ("train", "test"):
        for source in ("source1", "source2", "source3"):
            report["files"].append(profile_file(root / split / f"{split}_{source}.tsv"))
    truth = read_truth(root / "train" / "train_ground_truth.tsv")
    counts = Counter(map(len, truth.values()))
    report["ground_truth"] = {"entities": len(truth), "match_count": dict(counts),
                              "singletons": counts[0]}
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))


def cmd_split(args: argparse.Namespace) -> None:
    """Deterministic three-fold assignment stratified by country and label shape."""
    data = Path(args.data_dir)
    truth = read_truth(data / "train" / "train_ground_truth.tsv")
    countries = {r.entity_id: r.country for r in read_records(data / "train" / "train_source1.tsv")}
    buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
    for entity_id, matches in truth.items():
        bucket = 0 if not matches else min(len(matches), 6)
        buckets[(countries[entity_id], bucket)].append(entity_id)
    folds: dict[str, int] = {}
    for key, ids in sorted(buckets.items()):
        ids.sort(key=lambda value: hashlib.sha256(f"{args.seed}:{value}".encode()).hexdigest())
        for i, entity_id in enumerate(ids):
            folds[entity_id] = i % args.folds
    write_json(Path(args.out), {"seed": args.seed, "folds": args.folds, "assignment": folds})
    print(f"wrote {len(folds)} S1 assignments to {args.out}")


def key_rows(record: Record) -> Iterator[tuple[str, str]]:
    name_tokens = [x for x in tokens(record.name) if len(x) >= 3][:6]
    address_tokens = [x for x in tokens(record.address) if len(x) >= 3][:6]
    for token in name_tokens:
        yield "name_token", token
    for token in address_tokens:
        yield "address_token", token
    for gram in chars(record.name):
        yield "name_char3", gram
    if (value := prefix(record.name)):
        yield "name_prefix", value
    for number in numbers(record.address):
        for token in name_tokens[:2]:
            yield "number_name", f"{number}|{token}"


def create_index(db_path: Path, paths: Iterable[Path], batch_size: int = 10000) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        con.executescript("""
        PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
        CREATE TABLE records(entity_id TEXT PRIMARY KEY, source TEXT, country TEXT,
          name TEXT, address TEXT, name_norm TEXT, address_norm TEXT);
        CREATE TABLE candidate_keys(kind TEXT, key TEXT, country TEXT, entity_id TEXT);
        CREATE INDEX keys_lookup ON candidate_keys(kind, key, country);
        CREATE TABLE key_stats(kind TEXT, key TEXT, country TEXT, n INTEGER,
          PRIMARY KEY(kind, key, country));
        """)
        record_batch, key_batch = [], []
        for path in paths:
            source = "S2" if "source2" in path.name else "S3"
            for record in read_records(path):
                record_batch.append((record.entity_id, source, record.country, record.name,
                                     record.address, norm(record.name), norm(record.address)))
                key_batch.extend((kind, key, record.country, record.entity_id) for kind, key in key_rows(record))
                if len(record_batch) >= batch_size:
                    con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", record_batch)
                    con.executemany("INSERT INTO candidate_keys VALUES (?,?,?,?)", key_batch)
                    con.commit(); record_batch.clear(); key_batch.clear()
        if record_batch:
            con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", record_batch)
            con.executemany("INSERT INTO candidate_keys VALUES (?,?,?,?)", key_batch)
            con.commit()
        con.execute("INSERT INTO key_stats SELECT kind,key,country,COUNT(*) FROM candidate_keys GROUP BY kind,key,country")
        con.execute("CREATE INDEX stats_lookup ON key_stats(kind,key,country)")
        con.commit()
    finally:
        con.close()


def retrieve(con: sqlite3.Connection, record: Record, max_block: int, budget: int) -> dict[str, Candidate]:
    found: dict[str, Candidate] = {}
    for kind, key in key_rows(record):
        stat = con.execute("SELECT n FROM key_stats WHERE kind=? AND key=? AND country=?", (kind, key, record.country)).fetchone()
        if stat is None or stat[0] > max_block:
            continue
        rows = con.execute("SELECT entity_id FROM candidate_keys WHERE kind=? AND key=? AND country=? LIMIT ?", (kind, key, record.country, max_block)).fetchall()
        for rank, (entity_id,) in enumerate(rows, start=1):
            candidate = found.setdefault(entity_id, Candidate(entity_id, set(), rank))
            candidate.routes.add(kind)
            candidate.best_rank = min(candidate.best_rank, rank)
    # More independent routes and smaller within-block rank form a cheap retrieval score.
    # Sparse/missing records are objectively more ambiguous. Grant only those
    # rows a bounded rescue budget; normal rows retain the swept base budget.
    effective_budget = min(100, budget + 20) if len(tokens(record.name)) < 2 or not record.address else budget
    ranked = sorted(found.values(), key=lambda c: (-len(c.routes), c.best_rank, c.entity_id))[:effective_budget]
    return {candidate.entity_id: candidate for candidate in ranked}


def candidate_stats(truth: dict[str, set[str]], candidates: dict[str, set[str]]) -> dict[str, object]:
    sizes = sorted(len(value) for value in candidates.values())
    linked = sum(len(actual & candidates.get(s1, set())) for s1, actual in truth.items())
    total = sum(map(len, truth.values()))
    entities_all = sum(actual <= candidates.get(s1, set()) for s1, actual in truth.items())
    def percentile(p: float) -> float:
        return sizes[min(len(sizes) - 1, math.floor((len(sizes) - 1) * p))] if sizes else 0
    return {"candidate_link_recall": linked / total if total else 1.0,
            "entity_all_links_recall": entities_all / len(truth) if truth else 1.0,
            "total_pairs": sum(sizes), "mean_candidates": statistics.mean(sizes) if sizes else 0,
            "median_candidates": percentile(.5), "p95_candidates": percentile(.95),
            "p99_candidates": percentile(.99), "max_candidates": max(sizes, default=0)}


def write_candidates(path: Path, candidates: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=DELIM, lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1 in sorted(candidates):
            writer.writerow([s1, ",".join(sorted(candidates[s1]))])


def cmd_candidates(args: argparse.Namespace) -> None:
    data = Path(args.data_dir); split = args.split
    target_paths = [data / split / f"{split}_source2.tsv", data / split / f"{split}_source3.tsv"]
    db = Path(args.index)
    if args.rebuild or not db.exists():
        create_index(db, target_paths)
    con = sqlite3.connect(db)
    all_candidates: dict[str, set[str]] = {}
    meta_path = Path(args.meta); meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as meta:
        for i, record in enumerate(read_records(data / split / f"{split}_source1.tsv"), start=1):
            candidates = retrieve(con, record, args.max_block, args.budget)
            all_candidates[record.entity_id] = set(candidates)
            meta.write(json.dumps({"source1_entity_id": record.entity_id,
                                   "candidates": [{"entity_id": c.entity_id, "routes": sorted(c.routes), "best_rank": c.best_rank} for c in candidates.values()]}) + "\n")
            if i % 100000 == 0: print(f"retrieved {i:,}", file=sys.stderr)
    con.close(); write_candidates(Path(args.out), all_candidates)
    if split == "train":
        metrics = candidate_stats(read_truth(data / "train" / "train_ground_truth.tsv"), all_candidates)
        write_json(Path(args.report), metrics); print(json.dumps(metrics, indent=2))


def record_map(db_path: Path, ids: Iterable[str]) -> dict[str, Record]:
    con = sqlite3.connect(db_path)
    result: dict[str, Record] = {}
    try:
        for entity_id in ids:
            row = con.execute("SELECT entity_id,name,address,country FROM records WHERE entity_id=?", (entity_id,)).fetchone()
            if row: result[entity_id] = Record(*row)
    finally: con.close()
    return result


def fetch_records(con: sqlite3.Connection, ids: list[str]) -> dict[str, Record]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(
        f"SELECT entity_id,name,address,country FROM records WHERE entity_id IN ({placeholders})", ids
    ).fetchall()
    return {row[0]: Record(*row) for row in rows}


def similarity(a: str, b: str) -> float:
    from rapidfuzz.fuzz import ratio
    return ratio(norm(a), norm(b)) / 100.0


def feature_row(left: Record, right: Record, route_count: int = 0, best_rank: int = 999) -> list[float]:
    left_name, right_name = set(tokens(left.name)), set(tokens(right.name))
    left_addr, right_addr = set(tokens(left.address)), set(tokens(right.address))
    def jaccard(a: set[str], b: set[str]) -> float: return len(a & b) / len(a | b) if a | b else 0.0
    n1, n2 = set(numbers(left.address)), set(numbers(right.address))
    return [similarity(left.name, right.name), similarity(left.address, right.address),
            jaccard(left_name, right_name), jaccard(left_addr, right_addr),
            float(bool(n1 & n2)), float(bool(n1 and n2 and not (n1 & n2))),
            float(left.country == right.country), float(bool(left.address and right.address)),
            float(right.entity_id.startswith("S2-")), route_count, 1.0 / (1 + best_rank)]


def load_meta(path: Path) -> Iterator[tuple[str, list[dict[str, object]]]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line); yield row["source1_entity_id"], row["candidates"]


def load_folds(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    values = json.loads(Path(path).read_text(encoding="utf-8"))["assignment"]
    return {entity_id: int(fold) for entity_id, fold in values.items()}


def cmd_train(args: argparse.Namespace) -> None:
    try:
        import joblib
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before train: " + str(exc))
    data = Path(args.data_dir); truth = read_truth(data / "train" / "train_ground_truth.tsv")
    folds = load_folds(args.folds)
    left = {r.entity_id: r for r in read_records(data / "train" / "train_source1.tsv")}
    con = sqlite3.connect(args.index)
    x, y = [], []
    try:
        for s1, candidates in load_meta(Path(args.meta)):
            if folds and folds.get(s1) == args.exclude_fold:
                continue
            sample_value = int(hashlib.sha256(s1.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if sample_value > args.sample_fraction:
                continue
            ids = [str(candidate["entity_id"]) for candidate in candidates]
            right = fetch_records(con, ids)
            negatives = 0
            for candidate in candidates:
                cid = str(candidate["entity_id"]); positive = cid in truth[s1]
                if not positive and negatives >= args.max_negatives:
                    continue
                if cid not in right:
                    continue
                x.append(feature_row(left[s1], right[cid], len(candidate["routes"]), int(candidate["best_rank"])))
                y.append(int(positive))
                if not positive: negatives += 1
    finally:
        con.close()
    model = LGBMClassifier(n_estimators=args.trees, learning_rate=.05, num_leaves=31,
                           subsample=.8, colsample_bytree=.8, class_weight="balanced", n_jobs=-1)
    model.fit(x, y)
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    write_json(Path(args.report), {"pairs": len(y), "positives": int(sum(y)), "features": 11,
                                   "model": args.model, "excluded_fold": args.exclude_fold,
                                   "sample_fraction": args.sample_fraction,
                                   "max_negatives_per_s1": args.max_negatives})


def cmd_predict(args: argparse.Namespace) -> None:
    try: import joblib
    except ImportError as exc: raise SystemExit("Install requirements.txt before predict: " + str(exc))
    data = Path(args.data_dir); split = args.split; model = joblib.load(args.model)
    folds = load_folds(args.folds)
    left = {r.entity_id: r for r in read_records(data / split / f"{split}_source1.tsv")}
    con = sqlite3.connect(args.index)
    predictions: dict[str, set[str]] = {entity_id: set() for entity_id in left}
    try:
        for s1, candidates in load_meta(Path(args.meta)):
            if folds and args.only_fold is not None and folds.get(s1) != args.only_fold:
                continue
            ids = [str(candidate["entity_id"]) for candidate in candidates]
            right = fetch_records(con, ids); rows, kept_ids = [], []
            for candidate in candidates:
                cid = str(candidate["entity_id"])
                if cid in right:
                    rows.append(feature_row(left[s1], right[cid], len(candidate["routes"]), int(candidate["best_rank"])))
                    kept_ids.append(cid)
            for cid, probability in zip(kept_ids, model.predict_proba(rows)[:, 1] if rows else []):
                if probability >= args.threshold: predictions[s1].add(cid)
    finally:
        con.close()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=DELIM, lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1 in sorted(predictions): writer.writerow([s1, ",".join(sorted(predictions[s1]))])
    if split == "train":
        truth = read_truth(data / "train" / "train_ground_truth.tsv")
        if folds and args.only_fold is not None:
            truth = {s1: matches for s1, matches in truth.items() if folds.get(s1) == args.only_fold}
            predictions = {s1: predictions[s1] for s1 in truth}
        score = macro_f05(truth, predictions)
        print(json.dumps({"macro_f05": score, "threshold": args.threshold}, indent=2))


def read_results(path: Path, value_column: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=DELIM)
        expected = ("source1_entity_id", value_column)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path}: expected {expected}")
        for row in reader:
            if row["source1_entity_id"] in result:
                raise ValueError(f"{path}: duplicate Source-1 ID")
            result[row["source1_entity_id"]] = {x for x in row[value_column].split(",") if x}
    return result


def cmd_evaluate(args: argparse.Namespace) -> None:
    truth = read_truth(Path(args.truth))
    predicted = read_results(Path(args.matching), "matched_entity_ids")
    score = macro_f05(truth, predicted)
    singleton = {s1: ids for s1, ids in truth.items() if not ids}
    metrics = {"macro_f05": score,
               "singleton_accuracy": macro_f05(singleton, predicted),
               "entities": len(truth)}
    write_json(Path(args.out), metrics)
    print(json.dumps(metrics, indent=2))


def cmd_ablate(args: argparse.Namespace) -> None:
    """Evaluate candidate budgets against an already-built train index."""
    report: dict[str, object] = {"budgets": {}}
    data = Path(args.data_dir); truth = read_truth(data / "train" / "train_ground_truth.tsv")
    con = sqlite3.connect(args.index)
    try:
        source = data / "train" / "train_source1.tsv"
        for budget in args.budgets:
            candidates = {record.entity_id: set(retrieve(con, record, args.max_block, budget))
                          for record in read_records(source)}
            report["budgets"][str(budget)] = candidate_stats(truth, candidates)
    finally:
        con.close()
    write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))


def cmd_package(args: argparse.Namespace) -> None:
    """Create the required submission structure without copying prohibited data."""
    import shutil
    import zipfile
    root = Path(args.root); output = root / "student_resource" / "output"
    required = [output / "matching_results.tsv", output / "candidate_pairs.tsv",
                root / "student_resource" / "Documentation_template.md"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise SystemExit("Cannot package; missing: " + ", ".join(missing))
    package = Path(args.out); package.parent.mkdir(parents=True, exist_ok=True)
    code_root = root / "student_resource" / "code" / "business_entity_resolution"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in required:
            zf.write(path, path.relative_to(root / "student_resource"))
        for path in code_root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, Path("code") / "business_entity_resolution" / path.relative_to(code_root))
    print(f"created {package}")


def cmd_full(args: argparse.Namespace) -> None:
    """Run the reproducible baseline path; use individual commands for sweeps."""
    root = Path(args.root)
    data_dir = str(root / "student_resource" / "dataset")
    artifacts = root / "artifacts"
    cmd_eda(argparse.Namespace(data_dir=data_dir, out=str(artifacts / "reports" / "eda.json")))
    cmd_split(argparse.Namespace(data_dir=data_dir, out=str(artifacts / "splits" / "folds.json"), folds=3, seed=20260927))
    train_index = artifacts / "cache" / "train_targets.sqlite"
    train_meta = artifacts / "candidates" / "train.jsonl"
    cmd_candidates(argparse.Namespace(data_dir=data_dir, split="train", index=str(train_index),
        out=str(artifacts / "candidates" / "train.tsv"), meta=str(train_meta),
        report=str(artifacts / "reports" / "train_candidates.json"), budget=args.budget,
        max_block=args.max_block, rebuild=args.rebuild))
    model = artifacts / "models" / "matcher.joblib"
    cmd_train(argparse.Namespace(data_dir=data_dir, index=str(train_index), meta=str(train_meta),
        model=str(model), report=str(artifacts / "reports" / "train.json"), trees=args.trees,
        folds=None, exclude_fold=-1, sample_fraction=.25, max_negatives=10))
    test_index = artifacts / "cache" / "test_targets.sqlite"
    test_meta = artifacts / "candidates" / "test.jsonl"
    output = root / "student_resource" / "output"
    cmd_candidates(argparse.Namespace(data_dir=data_dir, split="test", index=str(test_index),
        out=str(output / "candidate_pairs.tsv"), meta=str(test_meta),
        report=str(artifacts / "reports" / "test_candidates.json"), budget=args.budget,
        max_block=args.max_block, rebuild=args.rebuild))
    cmd_predict(argparse.Namespace(data_dir=data_dir, split="test", index=str(test_index),
        meta=str(test_meta), model=str(model), out=str(output / "matching_results.tsv"), threshold=args.threshold,
        folds=None, only_fold=None))
    cmd_validate(argparse.Namespace(matching=str(output / "matching_results.tsv"),
        candidate=str(output / "candidate_pairs.tsv"), test_dir=str(Path(data_dir) / "test"),
        validator=str(root / "student_resource" / "utils" / "validate_submission.py"), check_ids=False))


def cmd_validate(args: argparse.Namespace) -> None:
    # Reuse the supplied official implementation instead of duplicating format rules.
    validator = Path(args.validator)
    namespace: dict[str, object] = {"__name__": "validator_module"}
    exec(compile(validator.read_text(encoding="utf-8"), str(validator), "exec"), namespace)
    errors, warnings = namespace["validate"](args.matching, args.candidate, args.test_dir, args.check_ids)
    for warning in warnings: print("WARNING:", warning)
    if errors:
        for error in errors: print("ERROR:", error)
        raise SystemExit(1)
    print("PASS")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="student_resource/dataset")
    p.add_argument("--config", default="student_resource/code/business_entity_resolution/config/default.json")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("eda"); q.add_argument("--out", default="artifacts/reports/eda.json"); q.set_defaults(func=cmd_eda)
    q = sub.add_parser("split"); q.add_argument("--out", default="artifacts/splits/folds.json"); q.add_argument("--folds", type=int, default=3); q.add_argument("--seed", type=int, default=20260927); q.set_defaults(func=cmd_split)
    q = sub.add_parser("candidates"); q.add_argument("--split", choices=["train", "test"], required=True); q.add_argument("--index", required=True); q.add_argument("--out", required=True); q.add_argument("--meta", required=True); q.add_argument("--report", default="artifacts/reports/candidates.json"); q.add_argument("--budget", type=int, default=40); q.add_argument("--max-block", type=int, default=200); q.add_argument("--rebuild", action="store_true"); q.set_defaults(func=cmd_candidates)
    q = sub.add_parser("train"); q.add_argument("--index", required=True); q.add_argument("--meta", required=True); q.add_argument("--model", default="artifacts/models/matcher.joblib"); q.add_argument("--report", default="artifacts/reports/train.json"); q.add_argument("--trees", type=int, default=300); q.add_argument("--folds"); q.add_argument("--exclude-fold", type=int, default=-1); q.add_argument("--sample-fraction", type=float, default=.25); q.add_argument("--max-negatives", type=int, default=10); q.set_defaults(func=cmd_train)
    q = sub.add_parser("predict"); q.add_argument("--split", choices=["train", "test"], required=True); q.add_argument("--index", required=True); q.add_argument("--meta", required=True); q.add_argument("--model", required=True); q.add_argument("--out", required=True); q.add_argument("--threshold", type=float, default=.8); q.add_argument("--folds"); q.add_argument("--only-fold", type=int); q.set_defaults(func=cmd_predict)
    q = sub.add_parser("evaluate"); q.add_argument("--truth", default="student_resource/dataset/train/train_ground_truth.tsv"); q.add_argument("--matching", required=True); q.add_argument("--out", default="artifacts/reports/evaluation.json"); q.set_defaults(func=cmd_evaluate)
    q = sub.add_parser("ablate"); q.add_argument("--index", required=True); q.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 30, 40, 60, 100]); q.add_argument("--max-block", type=int, default=200); q.add_argument("--out", default="artifacts/reports/candidate_ablation.json"); q.set_defaults(func=cmd_ablate)
    q = sub.add_parser("validate"); q.add_argument("--matching", required=True); q.add_argument("--candidate", required=True); q.add_argument("--test-dir", default="student_resource/dataset/test"); q.add_argument("--validator", default="student_resource/utils/validate_submission.py"); q.add_argument("--check-ids", action="store_true"); q.set_defaults(func=cmd_validate)
    q = sub.add_parser("package"); q.add_argument("--root", default="."); q.add_argument("--out", default="artifacts/submission/team_submission.zip"); q.set_defaults(func=cmd_package)
    q = sub.add_parser("full"); q.add_argument("--root", default="."); q.add_argument("--budget", type=int, default=40); q.add_argument("--max-block", type=int, default=200); q.add_argument("--trees", type=int, default=300); q.add_argument("--threshold", type=float, default=.8); q.add_argument("--rebuild", action="store_true"); q.set_defaults(func=cmd_full)
    return p


def main() -> None:
    args = parser().parse_args()
    config_path = Path(args.config)
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "--data-dir" not in sys.argv[1:] and config.get("data_dir"):
            args.data_dir = config["data_dir"]
        for key, value in config.get(args.command, {}).items():
            flag = "--" + key.replace("_", "-")
            if flag not in sys.argv[1:] and hasattr(args, key):
                setattr(args, key, value)
    resolved = {key: value for key, value in vars(args).items() if key != "func"}
    write_json(Path("artifacts/configs") / f"{args.command}_latest.json", resolved)
    args.func(args)


if __name__ == "__main__": main()
