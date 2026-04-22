from pathlib import Path

import scripts.build_plate_ocr_independent_eval_set as eval_script


def test_select_independent_entries_balances_by_subset_and_excludes_protected() -> None:
    entries = [
        Path("test/ccpd_challenge__a.jpg"),
        Path("test/ccpd_challenge__b.jpg"),
        Path("test/ccpd_blur__c.jpg"),
        Path("test/ccpd_blur__d.jpg"),
        Path("test/ccpd_db__e.jpg"),
    ]
    protected = {Path("test/ccpd_challenge__b.jpg")}

    selected, summary = eval_script.select_independent_entries(
        entries,
        include_subsets=("ccpd_challenge", "ccpd_blur", "ccpd_db"),
        protected_entries=protected,
        per_subset_limit=1,
        seed=7,
    )

    assert len(selected) == 3
    assert Path("test/ccpd_challenge__b.jpg") not in selected
    assert {eval_script.infer_source_subset(path) for path in selected} == {"ccpd_challenge", "ccpd_blur", "ccpd_db"}
    assert summary["protected_excluded_count"] == 1
    assert summary["selected_by_subset"] == {"ccpd_challenge": 1, "ccpd_blur": 1, "ccpd_db": 1}
