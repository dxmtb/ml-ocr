from mnist import read_mnist
from sklearn.linear_model import LogisticRegression
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')


if __name__ == '__main__':
    train_image, train_label, test_image, test_label = read_mnist()
    clf = LogisticRegression()
    logging.info("Fiting")
    clf.fit(train_image, train_label)
    logging.info("Scoring")
    logging.info("Score: %lf\n", clf.score(test_image, test_label))
