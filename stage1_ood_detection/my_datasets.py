"""Dataset helpers used by Stage-1 and Stage-2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _read_json_records(path: Path) -> List[Dict]:
    """Load newline-delimited JSON files into memory."""
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class TinghuaallData:
    """Binary ID/OOD split built from the processed Tinghuaall corpus."""

    name = "Tinghuaall"

    def __init__(self):
        self.nli = False
        self.num_out = 1
        self.num_iter = 30
        self.seed = 42
        self._root = Path(__file__).resolve().parent / "0_data_process_factory"
        self.train_file = self._root / "processed_train.json"
        self.valid_file = self._root / "processed_valid.json"
        self.test_file = self._root / "processed_test.json"
        self.id_labels = {"mail", "music", "youku", "taobao"}
        self.ood_labels = {"weixin", "weibo"}
        self.test_records: List[Dict] = []

    def _load_train_records(self) -> List[Dict]:
        train_records = _read_json_records(self.train_file)
        train_records.extend(_read_json_records(self.valid_file))
        return train_records

    def load(self) -> Tuple[List[str], List[str], List[int], List[int], Dict[str, int]]:
        train_records = self._load_train_records()
        test_records = _read_json_records(self.test_file)
        self.test_records = test_records

        X_train = [record["text"] for record in train_records]
        y_train = [0] * len(X_train)  # training set is purely ID traffic

        X_test = [record["text"] for record in test_records]
        y_test = [
            0 if record["label"] in self.id_labels else 1
            for record in test_records
        ]

        # Provide a deterministic mapping for compatibility with legacy callers
        mapping = {text: idx for idx, text in enumerate(X_train)}
        return X_train, X_test, y_train, y_test, mapping

    def iter_test_records(self) -> Sequence[Dict]:
        if not self.test_records:
            self.load()
        return self.test_records
