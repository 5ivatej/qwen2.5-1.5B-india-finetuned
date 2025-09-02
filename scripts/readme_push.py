from huggingface_hub import HfApi

api = HfApi()
repo_id = "5ivatej/qwen2.5-1.5B-india-finetuned"   # your model repo
api.upload_file(
    path_or_fileobj="README.md",    # local file
    path_in_repo="README.md",       # remote path
    repo_id=repo_id,
    repo_type="model",
    commit_message="Update model card"
)
