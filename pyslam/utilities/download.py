import os
import tempfile
import shutil
import requests
import gdown
from tqdm import tqdm
import multiprocessing as mp
import hashlib
import errno


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, "rb") as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath)


def gdrive_download(url, output, position=0):
    # Check if output folder exists or create it
    output_folder = os.path.dirname(output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output):
        print(f"downloading {url} to {output}")
        gdown.download(url, output)
    else:
        print(f"file already exists: {output}")


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
        print(f"Saved to: {os.path.abspath(filename)}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def http_download(url, output, position=0):
    # Create a temporary file in the same directory as the destination output
    temp_dir = tempfile.mkdtemp()
    temp_output = os.path.join(temp_dir, os.path.basename(output))

    # Check if output folder exists or create it
    output_folder = os.path.dirname(output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output):
        print(f" Downloading {position}: url: {url}, temporary location: {temp_output}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 KiB

            # Use tqdm with dynamic_ncols to adjust the progress bar size dynamically
            progress_bar = tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,  # position=position, # TODO: position does not seem to work as expected
                leave=False,  # Prevent the progress bar from leaving a line after finishing
                ncols=100,  # Set a fixed width for the progress bar
                dynamic_ncols=True,  # Allow dynamic width adjustment
            )

            with open(temp_output, "wb") as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)

            progress_bar.close()

            # Move the downloaded file to the destination output
            shutil.move(temp_output, output)
            print(f" Download {position} complete, file moved to {output}")

        except requests.exceptions.RequestException as e:
            print(f" Error downloading {url}: {e}")

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    else:
        print(f" File already exists: {output}")


class DataDownloader:
    def __init__(self, download_json):
        self.download_json = download_json

    def download_process(self, url, path, type, position=0):
        p = None
        if type == "http":
            p = mp.Process(target=http_download, args=(url, path, position))
        elif type == "gdrive":
            p = mp.Process(target=gdrive_download, args=(url, path, position))
        else:
            raise NotImplementedError(f"Download type '{type}' is not implemented")
        return p

    def start(self):
        processes = []
        position = 0  # Start progress bars at position 1
        for pth, (url, type) in self.download_json.items():
            p = self.download_process(url=url, path=pth, type=type, position=position)
            position += 1
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
