__all__ = ["normalise_labels"]


def normalise_labels(ll):
    """
    
    Args:
        ll:

    Returns:

    """
    if "walk" in ll:
        return "walk"
    elif "elevator" in ll:
        return "stand"
    elif ll in {"lie", "sleep"}:
        return "lie"
    elif ll in {"vacuum", "iron", "laundry", "clean"}:
        return "chores"
    elif ll in {"run", "jump", "rope_jump", "soccer", "cycle"}:
        return "sport"
    return ll
