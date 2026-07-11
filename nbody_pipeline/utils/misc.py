"""Miscellaneous utility functions"""


def can_convert_to_float(s: str) -> bool:
    """
    Check if a string can be converted to a float.

    Args:
        s: String to check

    Returns:
        True if the string can be converted to float, False otherwise
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
