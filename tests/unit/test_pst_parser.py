from __future__ import annotations

import pytest

from pst_email_extractor import pst_parser


def test_is_pypff_available_returns_bool(monkeypatch):
    monkeypatch.setattr(pst_parser, "_PYPFF_AVAILABLE", False, raising=False)
    assert pst_parser.is_pypff_available() is False


def test_require_pypff_raises(monkeypatch):
    monkeypatch.setattr(pst_parser, "_PYPFF_AVAILABLE", False, raising=False)
    with pytest.raises(pst_parser.DependencyError):
        pst_parser.require_pypff()


def test_require_pypff_no_raise(monkeypatch):
    monkeypatch.setattr(pst_parser, "_PYPFF_AVAILABLE", True, raising=False)
    pst_parser.require_pypff()


def test_iter_emails_requires_pypff(monkeypatch):
    monkeypatch.setattr(pst_parser, "_PYPFF_AVAILABLE", False, raising=False)
    with pytest.raises(pst_parser.DependencyError):
        list(pst_parser.iter_emails("dummy.pst"))
