import os
import sys
import shutil
import subprocess

from argparse import ArgumentParser

DIR = os.path.dirname(os.path.abspath(__file__))
FTP_URL = "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D"
OBJECT_NET = "ObjectNet3D"
OBJECT_NET_PATH = f"{DIR}/{OBJECT_NET}"
CONTENT = {
    "Image_sets" : "ObjectNet3D_image_sets.zip",
    "CAD" : "ObjectNet3D_cads.zip",
    "Annotations" : "ObjectNet3D_annotations.zip",
    "Images" : "ObjectNet3D_images.zip"
}

def main(args):

    if not os.path.exists(OBJECT_NET_PATH):
        os.mkdir(OBJECT_NET_PATH)
    
    for content_type, zip_file in CONTENT.items():
        if os.path.exists(f"{OBJECT_NET_PATH}/{content_type}"):
            print(f"{OBJECT_NET}/{content_type} Already Exists")
            continue
        
        subprocess.run(["wget", "-O", f"{OBJECT_NET_PATH}/{zip_file}", f"{FTP_URL}/{zip_file}"])
        subprocess.run(["unzip", "-q", "-d", f"{OBJECT_NET_PATH}/{content_type}_temp", f"{OBJECT_NET_PATH}/{zip_file}"])
        subprocess.run(["rm", f"{OBJECT_NET_PATH}/{zip_file}"])
        subprocess.run(" ".join(
            ["mv", f"{OBJECT_NET_PATH}/{content_type}_temp/{OBJECT_NET}/*", f"{OBJECT_NET_PATH}/"]
            ), shell=True)
        subprocess.run(["rm", "-r", f"{OBJECT_NET_PATH}/{content_type}_temp"])

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
