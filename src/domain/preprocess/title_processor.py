from __future__ import annotations

import hashlib
import re

from shared.utils import get_logger

logger = get_logger(__name__)


class TitleProcessor:
    """Processor for cleaning and normalizing title text."""

    @staticmethod
    def trim_and_case(text: str) -> str:
        """Trim whitespace and normalize case."""
        return text.strip().title()

    @staticmethod
    def whitespace_collapse(text: str) -> str:
        """Collapse multiple whitespaces into single space."""
        return re.sub(r'\s+', ' ', text)

    @staticmethod
    def bracket_stripping(text: str) -> str:
        """Remove content within brackets and the brackets themselves."""
        # Remove content within various types of brackets
        text = re.sub(r'\[.*?\]', '', text)  # Square brackets
        text = re.sub(r'\(.*?\)', '', text)  # Round brackets
        text = re.sub(r'\{.*?\}', '', text)  # Curly brackets
        text = re.sub(r'<.*?>', '', text)    # Angle brackets
        return text

    @staticmethod
    def delimiter_normalise(text: str) -> str:
        """Normalize delimiters to standard format."""
        # Replace various delimiters with standard ones
        text = re.sub(r'[–—―]', '-', text)    # Em/en dashes to hyphen
        text = re.sub(r'[''‚‛]', "'", text)   # Various apostrophes
        text = re.sub(r'[""„‟]', '"', text)   # Various quotes
        text = re.sub(r'[…]', '...', text)    # Ellipsis
        return text

    @staticmethod
    def prefix_suffix_rules(text: str) -> str:
        """Apply prefix/suffix normalization rules."""
        # Remove common prefixes
        prefixes = [r'^(the|a|an)\s+', r'^(project|task|work)\s+']
        for prefix in prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)

        # Remove common suffixes
        suffixes = [r'\s+(inc|ltd|corp|llc)\.?$', r'\s+(project|task|work)$']
        for suffix in suffixes:
            text = re.sub(suffix, '', text, flags=re.IGNORECASE)

        return text

    @classmethod
    def process_title(cls, title: str) -> str:
        """Apply all title processing techniques."""
        if title is None or title == '':
            return ''

        # Convert to string if not already
        title = str(title)

        # Apply processing steps in order
        title = cls.trim_and_case(title)
        title = cls.whitespace_collapse(title)
        title = cls.bracket_stripping(title)
        title = cls.delimiter_normalise(title)
        title = cls.prefix_suffix_rules(title)
        title = cls.whitespace_collapse(title)  # Final cleanup
        title = title.strip()

        return title

    @classmethod
    def generate_title_hash(cls, title: str) -> str:
        """Generate hash for processed title."""
        processed_title = cls.process_title(title)
        return hashlib.md5(processed_title.lower().encode()).hexdigest()[:8]
