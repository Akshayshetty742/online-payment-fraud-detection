"""
Module to download datasets from remote sources (e.g. Google Drive).
"""

import os
import re

DEFAULT_FILE_ID = "127JqP3WGjBVihR-ZcUR86T3wwy3_g63v"
DEFAULT_DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")
FALLBACK_FILENAME = "new_data.csv"

def get_confirm_token_and_url(response_text: str):
    """Parse confirmation URL and parameters from Google Drive virus scan warning page."""
    action_match = re.search(r'action="([^"]+)"', response_text)
    action_url = action_match.group(1) if action_match else "https://drive.usercontent.google.com/download"
    
    params = {}
    for match in re.finditer(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', response_text):
        params[match.group(1)] = match.group(2)
        
    return action_url, params

def download_dataset(file_id: str = DEFAULT_FILE_ID, destination_dir: str = DEFAULT_DEST_DIR) -> str:
    """
    Downloads the dataset from Google Drive handling large-file confirmation tokens.
    
    Args:
        file_id: Google Drive file ID
        destination_dir: Directory where the file will be saved
        
    Returns:
        str: Absolute path to the downloaded file
    """
    import requests
    from tqdm import tqdm

    os.makedirs(destination_dir, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    base_url = "https://drive.google.com/uc?export=download"
    print(f"Connecting to Google Drive for File ID: {file_id}...")
    response = session.get(base_url, params={'id': file_id, 'export': 'download'}, stream=True)
    
    content_type = response.headers.get('Content-Type', '')
    dest_path = os.path.join(destination_dir, FALLBACK_FILENAME)
    
    # Check if virus scan warning page is returned (HTML response)
    if 'text/html' in content_type:
        html = response.text
        fn_match = re.search(r'<span class="uc-name-size"><a[^>]*>([^<]+)</a>', html)
        filename = fn_match.group(1) if fn_match else FALLBACK_FILENAME
        dest_path = os.path.join(destination_dir, filename)
        
        action_url, params = get_confirm_token_and_url(html)
        if not params:
            params = {'id': file_id, 'export': 'download', 'confirm': 't'}
            
        print(f"Detected large file '{filename}'. Confirming download via: {action_url}")
        response = session.get(action_url, params=params, stream=True)
    else:
        cd = response.headers.get('Content-Disposition', '')
        if 'filename=' in cd:
            filename = cd.split('filename=')[-1].strip('"\'')
            dest_path = os.path.join(destination_dir, filename)
            
    if response.status_code != 200:
        raise RuntimeError(f"Download failed with HTTP status {response.status_code}: {response.text[:200]}")
        
    total_size = int(response.headers.get('Content-Length', 0))
    print(f"Downloading to: {dest_path} (Expected size: {total_size / (1024*1024):.1f} MB if known)")
    
    chunk_size = 1024 * 1024  # 1MB
    with open(dest_path, "wb") as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
                
    file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"Successfully downloaded: {dest_path} ({file_size_mb:.2f} MB)")
    return dest_path
