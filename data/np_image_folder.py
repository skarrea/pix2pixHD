'''
Helper function to get a numpy array dataset from selected folder.
'''
import os

NP_EXTENSIONS = [
	'.npy'
]


def is_numpy_array(filename):
    return any(filename.endswith(extension) for extension in NP_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_numpy_array(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images