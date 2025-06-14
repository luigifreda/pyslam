#!/usr/bin/env python3
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define files to download
files = {
    "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth": "https://huggingface.co/Zhenggang/MV-DUSt3R/resolve/main/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
    "MVD.pth": "https://huggingface.co/Zhenggang/MV-DUSt3R/resolve/main/checkpoints/MVD.pth",
    "MVDp_s1.pth": "https://huggingface.co/Zhenggang/MV-DUSt3R/resolve/main/checkpoints/MVDp_s1.pth",
    "MVDp_s2.pth": "https://huggingface.co/Zhenggang/MV-DUSt3R/resolve/main/checkpoints/MVDp_s2.pth",  # ✅ fixed link
}

def download_with_progress(filename, url):
    if os.path.exists(filename):
        return f"[✓] Already exists: {filename}"

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(filename, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True, desc=filename, ncols=80
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        return f"[✓] Finished: {filename}"
    except Exception as e:
        return f"[✗] Failed: {filename} — {e}"

# Parallel download with status
if __name__ == "__main__":
    print("[INFO] Starting parallel downloads...\n")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_with_progress, fn, url) for fn, url in files.items()]
        for future in as_completed(futures):
            print(future.result())
