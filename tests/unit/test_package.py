from pst_email_extractor import ExtractionConfig, __version__


def test_version_available():
    assert isinstance(__version__, str) and __version__


def test_config_defaults(tmp_path):
    cfg = ExtractionConfig(pst_path=tmp_path / 'sample.pst', output_dir=tmp_path, formats=['json'])
    assert cfg.mode == 'extract'
