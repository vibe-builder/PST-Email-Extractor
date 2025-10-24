from __future__ import annotations

from typer.testing import CliRunner

from pst_email_extractor.cli.app import app

runner = CliRunner()


def test_cli_version_option():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_cli_requires_paths(tmp_path):
    result = runner.invoke(app, [
        "extract",
        "--pst",
        str(tmp_path / "missing.pst"),
        "--output",
        str(tmp_path),
        "--json",
    ])
    assert result.exit_code != 0


def test_cli_address_mode_validates_formats(tmp_path):
    result = runner.invoke(app, [
        "addresses",
        "--pst",
        str(tmp_path / "missing.pst"),
        "--output",
        str(tmp_path),
        "--format",
        "json",
        "--format",
        "csv",
    ])
    # Should fail before path validation for missing file
    assert result.exit_code != 0
