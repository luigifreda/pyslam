"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""


# From torchvision.uitls
import smtplib, socket, os, os.path, hashlib, errno
import __main__ as main
from email.mime.text import MIMEText
from os import path, mkdir
import multiprocessing
import gdown

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
                print("Failed download. Trying https -> http instead." " Downloading " + url + " to " + fpath)
                urllib.request.urlretrieve(url, fpath)


def check_dir(dir):
    if not path.isdir(dir):
        mkdir(dir)
        
        
def list_dir(root, prefix=False):
    ### List all directories at a given root
    # root (str): Path to directory whose folders need to be listed
    # prefix (bool, optional): If true, prepends the path to each result, otherwise only returns the name of the directories found
    root = os.path.expanduser(root)
    directories = list(filter(lambda p: os.path.isdir(os.path.join(root, p)), os.listdir(root)))

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root, suffix, prefix=False):
    ### List all files ending with a suffix at a given root
    # root (str): Path to directory whose folders need to be listed
    # suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png'). It uses the Python "str.endswith" method and is passed directly
    # prefix (bool, optional): If true, prepends the path to each result, otherwise only returns the name of the files found
    root = os.path.expanduser(root)
    files = list(filter(lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix), os.listdir(root)))

    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def make_dirs(dir, exist_ok=True):
    # helper function, adds exist_ok to python2
    if exist_ok:
        try:
            os.makedirs(dir)
        except:
            pass
    else:
        os.makedirs(dir)
        
        
import smtplib, socket, os
from email.mime.text import MIMEText

# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
# with open(textfile) as fp:
# Create a text/plain message
# msg = MIMEText(fp.read())

def send_email(recipient, ignore_host="", login_gmail="", login_password=""):
    msg = MIMEText("")

    if socket.gethostname() == ignore_host:
        return
    msg["Subject"] = socket.gethostname() + " just finished running a job " + os.path.basename(main.__file__)
    msg["From"] = "pyslam@gmail.com"
    msg["To"] = recipient

    s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    s.ehlo()
    s.login(login_gmail, login_password)
    s.sendmail(login_gmail, recipient, msg.as_string())
    s.quit()


# download file from google drive
# need to pass as input the 'url' and output 'path'
def gdrive_download_lambda(*args, **kwargs):
    url = kwargs["url"]
    output = kwargs["path"]
    # check if outfolder exists or create it
    output_folder = os.path.dirname(output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output):
        print(f'downloading {url} to {output}')
        gdown.download(url, output)
    else: 
        print(f'file already exists: {output}')