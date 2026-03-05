import os
import random
import shutil

def create_subset(src_dir, dst_dir, n):

    os.makedirs(dst_dir, exist_ok=True)

    images = os.listdir(src_dir)
    selected = random.sample(images, n)

    for img in selected:

        src = os.path.join(src_dir, img)
        dst = os.path.join(dst_dir, img)

        shutil.copy(src, dst)


def main():

    create_subset("../data/train/REAL", "../data/train_small/real", 5000)
    create_subset("../data/train/FAKE", "../data/train_small/fake", 5000)

    create_subset("../data/test/REAL", "../data/test_small/real", 2000)
    create_subset("../data/test/FAKE", "../data/test_small/fake", 2000)


if __name__ == "__main__":
    main()