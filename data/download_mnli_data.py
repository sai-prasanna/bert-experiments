''' Script for downloading MNLI data.

This is a fork of https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py

'''

import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='glue_data')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    url = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce'

    print("Downloading and extracting MNLI...")
    data_file = "MNLI.zip"
    urllib.request.urlretrieve(url, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(args.data_dir)
    os.remove(data_file)
    print("\tCompleted!")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))