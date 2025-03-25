def fully_flatten(l): # from tinygrad repo
  if hasattr(l, "__len__") and hasattr(l, "__getitem__") and not isinstance(l, str):
    flattened = []
    for li in l: flattened.extend(fully_flatten(li))
    return flattened
  return [l]