import os
import sys
import shutil
import subprocess
import argparse

ROOT_ELEVATION = 2
DIR = os.path.abspath(__file__)
for i in range(ROOT_ELEVATION):
    DIR = os.path.dirname(DIR)

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", default=10, type=int,
                    help="ModelNet Version (10, 40)")

MODEL_NET_10_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
MODEL_NET_40_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"

MODEL_NET_10_PATH = f"{DIR}/ModelNet10"
MODEL_NET_40_PATH = f"{DIR}/ModelNet40"

def main(args):

    if args.version == 10:
        MODEL_NET_URL = MODEL_NET_10_URL
        MODEL_NET_PATH = MODEL_NET_10_PATH
    elif args.version == 40:
        MODEL_NET_URL = MODEL_NET_40_URL
        MODEL_NET_PATH = MODEL_NET_40_PATH
    else:
        raise ValueError()

    if not os.path.exists(MODEL_NET_PATH):
        os.mkdir(MODEL_NET_PATH)
        
        subprocess.run(["wget", "-O", f"{MODEL_NET_PATH}/ModelNet{args.version}.zip", f"{MODEL_NET_URL}"])
        subprocess.run(["unzip", "-q", "-d", f"{MODEL_NET_PATH}/ModelNet{args.version}_temp", f"{MODEL_NET_PATH}/ModelNet{args.version}.zip"])
        subprocess.run(["rm", f"{MODEL_NET_PATH}/ModelNet{args.version}.zip"])
        subprocess.run(" ".join(
            ["mv", f"{MODEL_NET_PATH}/ModelNet10_temp/ModelNet{args.version}/*", f"{MODEL_NET_PATH}/"]
            ), shell=True)
        subprocess.run(["rm", "-r", f"{MODEL_NET_PATH}/ModelNet{args.version}_temp"])
    else:
        print(f"ModelNet{args.version} Already Downloaded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
