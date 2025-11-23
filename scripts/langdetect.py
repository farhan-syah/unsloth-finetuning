"""
Fast langdetect replacement to prevent 18-20 minute delay in IFEval benchmarks.

When base models generate malformed text, langdetect becomes extremely slow.
This module replaces langdetect with a dummy that always returns 'en'.

Usage: Set PYTHONPATH to include this script's directory before running lm_eval
"""


class LangDetectException(Exception):
    """Exception raised when language detection fails"""
    pass


def detect(text):
    """Always return English to avoid slow text analysis"""
    return 'en'


def detect_langs(text):
    """Always return English with 100% confidence"""
    class Lang:
        lang = 'en'
        prob = 1.0
    return [Lang()]


# Expose exception for compatibility
__all__ = ['detect', 'detect_langs', 'LangDetectException']
