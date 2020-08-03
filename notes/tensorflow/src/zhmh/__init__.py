import os


def make_cache(cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
