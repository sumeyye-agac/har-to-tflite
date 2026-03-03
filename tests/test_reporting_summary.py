from pathlib import Path

from h2t.reporting.report import write_summary
from h2t.utils.jsonio import write_json


def test_summary_includes_status_and_artifact_flags(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    artifacts.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    write_json(
        artifacts / "data_summary.json",
        {
            "source": "synthetic",
            "x_train_shape": [8, 128, 9],
            "x_test_shape": [4, 128, 9],
        },
    )
    write_json(
        artifacts / "train_metrics.json",
        {
            "backend": "numpy_stub",
            "eval_accuracy": 0.5,
            "student_model_path": "artifacts/models/student_stub.npz",
        },
    )
    write_json(
        artifacts / "export_manifest.json",
        {"variants": {"fp32": {"status": "skipped", "size_bytes": 0, "reason": "no_tf"}}},
    )

    (results / "bench_host.csv").write_text(
        "variant,status,mean_ms,p50_ms,p90_ms,num_runs,threads,reason\n"
        "fp32,ok,1.2,1.2,1.3,3,1,\n",
        encoding="utf-8",
    )
    (results / "bench_android.csv").write_text(
        "variant,status,mean_ms,p50_ms,p90_ms,runs,threads,use_nnapi,reason\n"
        "fp32,stub,0,0,0,0,0,false,adb_missing\n",
        encoding="utf-8",
    )

    cfg = {"paths": {"artifacts_dir": str(artifacts), "results_dir": str(results)}}
    summary_path, leaderboard_path = write_summary(cfg)

    content = summary_path.read_text(encoding="utf-8")
    assert "## Run status" in content
    assert "complete_artifacts_present: True" in content
    assert "best_host_ms: 1.2" in content
    assert leaderboard_path.exists()
