"""Verification for the SupportPilot knowledge layer fixes."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from repositories.faq_repository import FAQRepository, FAQ_PATH, VECTORSTORE_PATH


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def main() -> None:
    # --- Fresh ingest of all FAQs on the real store -----------------
    banner("1. FRESH INGEST")
    if VECTORSTORE_PATH.exists():
        shutil.rmtree(VECTORSTORE_PATH)
        print(f"Cleared existing vectorstore at {VECTORSTORE_PATH}")

    expected = len(json.loads(FAQ_PATH.read_text(encoding="utf-8")))
    repo = FAQRepository()
    count = repo.collection.count()
    print(f"FAQs in JSON:        {expected}")
    print(f"Documents in store:  {count}")
    print(f"Distance metric:     {repo.collection.metadata.get('hnsw:space')}")
    assert count == expected, f"expected {expected} docs, got {count}"
    print(f"PASS: all {count} FAQs ingested fresh")

    # --- Two real test queries --------------------------------------
    banner("2. TEST QUERIES")
    for query in (
        "How do I track my order?",
        "I need to cancel my grooming appointment",
    ):
        record = repo.search(query, top_k=1)[0]
        print(f"\nquery:    {query!r}")
        print(f"title:    {record['title']}")
        print(f"category: {record['category']}")
        print(f"distance: {record['distance']:.3f}")
        # Confirm the neutral schema is exactly the agreed key set.
        assert set(record) == {"id", "title", "content", "category", "distance"}

    # --- Stale detection on an isolated temp store ------------------
    banner("3. STALE DETECTION")
    # ignore_cleanup_errors: ChromaDB keeps file handles open on Windows,
    # which would otherwise make temp-dir teardown raise.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        faq_copy = tmp / "faq.json"
        store = tmp / "vectorstore"
        faqs = json.loads(FAQ_PATH.read_text(encoding="utf-8"))
        faq_copy.write_text(json.dumps(faqs), encoding="utf-8")

        repo1 = FAQRepository(faq_path=faq_copy, vectorstore_path=store)
        hash1 = repo1.collection.metadata.get("faq_hash")
        print(f"initial ingest:     {repo1.collection.count()} docs, hash={hash1[:12]}...")

        # (a) edit an answer -> same count, different hash -> re-ingest
        faqs[0]["answer"] = faqs[0]["answer"] + " (edited)"
        faq_copy.write_text(json.dumps(faqs), encoding="utf-8")
        repo2 = FAQRepository(faq_path=faq_copy, vectorstore_path=store)
        hash2 = repo2.collection.metadata.get("faq_hash")
        edited = repo2.search(faqs[0]["question"], top_k=1)[0]
        print(f"after edit:         {repo2.collection.count()} docs, hash={hash2[:12]}...")
        assert hash2 != hash1, "hash should change after editing faq.json"
        assert edited["content"].endswith("(edited)"), "edited answer not served"
        print("PASS: edited faq.json triggered re-ingest (hash mismatch); fresh answer served")

        # (b) remove an entry -> count mismatch -> re-ingest
        count_before = repo2.collection.count()
        faqs.pop()
        faq_copy.write_text(json.dumps(faqs), encoding="utf-8")
        repo3 = FAQRepository(faq_path=faq_copy, vectorstore_path=store)
        print(f"after removing one: {repo3.collection.count()} docs (was {count_before})")
        assert repo3.collection.count() == len(faqs)
        print("PASS: count change triggered re-ingest")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
