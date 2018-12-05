import hashlib
import sys

def main(argv=None):

    if argv is None:
        argv = sys.argv

    with open(argv[0], 'rb') as file:
        x = file.read()

    print(hashlib.sha3_256(x).hexdigest())
    return 0

if __name__ == '__main__':
    sys.exit(main())
