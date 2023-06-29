import os
from huggingface_hub import hf_hub_download


def download_file(destination_folder: str, repo_id: str, filename: str):
    local_path = os.path.abspath(destination_folder)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_path,
        local_dir_use_symlinks=True,
    )


if __name__ == "__main__":
    """full url: https://huggingface.co/abacaj/Replit-v2-CodeInstruct-3B-ggml/blob/main/replit-v2-codeinstruct-3b.q4_1.bin"""

    destination_folder = "models"
    repo_id = "abacaj/Replit-v2-CodeInstruct-3B-ggml"
    model_filename = "replit-v2-codeinstruct-3b.q4_1.bin"
    config_filename = "config.json"

    download_file(destination_folder, repo_id, model_filename)
    download_file(destination_folder, repo_id, config_filename)
