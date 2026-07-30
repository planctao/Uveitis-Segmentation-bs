import json

from scripts.summarize_innovation_runs import load_run, markdown


def test_load_run_compares_fixed_reference(tmp_path) -> None:
    run = tmp_path / "innovation"
    run.mkdir()
    summary = run / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold": "f1",
                        "best_epoch": 7,
                        "best_paper_macro_dice": 0.80,
                        "best_paper_dice_1": 0.81,
                        "best_paper_dice_2": 0.79,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    row = load_run(summary, reference_macro=0.77)

    assert row["run"] == "innovation"
    assert row["beats_reference"] is True
    assert abs(row["gain"] - 0.03) < 1e-9


def test_markdown_marks_failed_candidate() -> None:
    report = markdown(
        [
            {
                "run": "candidate",
                "folds": ["f1"],
                "mean_dice_1": 0.7,
                "mean_dice_2": 0.7,
                "mean_macro_dice": 0.7,
                "gain": -0.07,
                "beats_reference": False,
            }
        ],
        "baseline",
        0.77,
    )

    assert "| no |" in report
