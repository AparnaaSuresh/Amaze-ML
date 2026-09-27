import tempfile
import unittest
from pathlib import Path

from pipeline import Candidate, Record, create_index, macro_f05, norm, retrieve, suffix_free_name


class MetricTests(unittest.TestCase):
    def test_singleton(self):
        truth = {"S1-1": set()}
        self.assertEqual(macro_f05(truth, {"S1-1": set()}), 1.0)
        self.assertEqual(macro_f05(truth, {"S1-1": {"S2-1"}}), 0.0)

    def test_multi_match(self):
        truth = {"S1-1": {"S2-1", "S3-1"}}
        self.assertAlmostEqual(macro_f05(truth, {"S1-1": {"S2-1", "S2-x", "S3-1"}}), 5 / 7)

    def test_normalization_preserves_meaningful_tokens(self):
        self.assertEqual(norm("A&B, Pvt. Ltd."), "a and b pvt ltd")
        self.assertEqual(suffix_free_name("A&B, Pvt. Ltd."), "a and b")

    def test_sqlite_candidate_retrieval_deduplicates_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tsv = root / "train_source2.tsv"
            tsv.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                           "S2-1\tAcme Pvt Ltd\t12 Main Road\tIndia\n"
                           "S2-2\tOther Shop\t99 Elsewhere\tIndia\n", encoding="utf-8")
            index = root / "target.sqlite"
            create_index(index, [tsv], batch_size=1)
            import sqlite3
            con = sqlite3.connect(index)
            candidates = retrieve(con, Record("S1-1", "Acme Limited", "12 Main Rd", "India"), 50, 10)
            con.close()
            self.assertEqual(set(candidates), {"S2-1"})
            self.assertGreaterEqual(len(candidates["S2-1"].routes), 1)


if __name__ == "__main__": unittest.main()
