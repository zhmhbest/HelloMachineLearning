import torch


def check_device():
    if 0 == torch.cuda.device_count():
        print("None cuda device.")
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.get_device_name()}.")
    else:
        print(f"Using cpu.")


check_device()
