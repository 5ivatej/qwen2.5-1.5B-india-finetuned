# scripts/push_merged.py
from huggingface_hub import HfApi, create_repo, upload_folder
api = HfApi()

# change this to your HF username/org + repo name
repo_id = "5ivatej/qwen2.5-1.5B-india-finetuned"

create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

upload_folder(
    repo_id=repo_id,
    folder_path="qwen25-1p5b-india-merged",  # your merged dir
    repo_type="model",
    commit_message="Initial upload: merged LoRA into Qwen2.5-1.5B",
    ignore_patterns=["*.ipynb_checkpoints/*", "data/*", "adapters/*"]
)
print("Uploaded to:", repo_id)
