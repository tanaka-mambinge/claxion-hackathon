def clipper(amount: float, min: float, max: float) -> float:
    if amount < min:
        return min
    elif amount > max:
        return max
    else:
        return amount


standardize_columns = lambda x: x.lower().strip().replace(" ", "_")
