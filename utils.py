def safe_float(val):
    """Safely convert a string to float. Returns np.nan if conversion fails."""
    try:
        return float(val.strip())
    except (ValueError, TypeError, AttributeError):
        return np.nan