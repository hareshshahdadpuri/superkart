import os
from huggingface_hub import HfApi

def deploy():
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print('⚠️ HF_TOKEN not set. Skipping deployment.')
        return
    api = HfApi()
    api.create_repo(
        repo_id='shahdadpuri/superkart-sales-dashboard',
        repo_type='space', space_sdk='docker', exist_ok=True, token=hf_token
    )
    api.upload_folder(
        folder_path='app', path_in_repo='app',
        repo_id='shahdadpuri/superkart-sales-dashboard',
        repo_type='space', token=hf_token
    )
    api.upload_file(
        path_or_fileobj='Dockerfile', path_in_repo='Dockerfile',
        repo_id='shahdadpuri/superkart-sales-dashboard',
        repo_type='space', token=hf_token
    )
    print('✅ App deployed to HF Spaces: shahdadpuri/superkart-sales-dashboard')

if __name__ == '__main__':
    deploy()
