"""Use only the exact frozen M8 final-register development features."""
from large_token_data import load_all as load_token_streams


def load_all():
    frame, base, tokens, digest = load_token_streams()
    return frame, {**base, 'side_tokens': tokens['registers']}, digest
