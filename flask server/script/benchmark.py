import sys
from argparse import ArgumentParser

def local_benchmark(path):
    f = open(path, mode='r')
    cnt = 0
    sum = 0
    for i in f.readlines():
        # print(float(i))
        sum = sum + float(i)
        cnt = cnt + 1
    result = sum/float(cnt)
    print(result)
    return result

parser = ArgumentParser()
parser.add_argument('--local', help='local data path', type=str, default=None)
parser.add_argument('--gcp', help='GCP data path', type=str, default=None)
args = parser.parse_args()
if (args.local is None):
    print("You have to load the local by --local")
    sys.exit()
if (args.gcp is None):
    print("You have to load the gcp by --gcp")
    sys.exit()
local_average = local_benchmark(args.local)
gcp_average = local_benchmark(args.gcp)