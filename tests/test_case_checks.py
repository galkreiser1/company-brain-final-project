"""The Webex scoring rules (evals/case_checks.py) pass their own offline selftest."""

from evals.case_checks import selftest


def test_case_checks_selftest():
    selftest()
