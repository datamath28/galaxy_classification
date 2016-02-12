import numpy as np
import os
from scipy import misc
from constants import max_num_images, image_size, num_channels, num_classes, batch_size


def randomize(dataset, labels):
      permutation = np.random.permutation(labels.shape[0])
      shuffled_dataset = dataset[permutation,:,:,:]
      shuffled_labels = labels[permutation, :]
      return shuffled_dataset, shuffled_labels

# Object to provide the feed dictionary
# This provides polymorphism.  Later we can replace with a fancy class that uses threads and queues
# and does rotations/reflection and possibly other image distortions.


class Feeder(object):

    def __init__(self):
        print "Loading data into memory..."
        self.offset = {'train':0, 'valid':0, 'test':0}

        # Metadata and labels
        metadata = np.loadtxt('../../data/gal_pos_label.txt', delimiter=',') # columns are: dr7objid,ra,dec,spiral,elliptical,uncertain
        gal_id = np.genfromtxt('../../data/gal_pos_label.txt', delimiter=',', dtype=int, usecols=[0])

        dataset = np.ndarray(
            shape=(max_num_images, image_size, image_size, num_channels), dtype=np.float32)
        labels = np.ndarray(shape=(max_num_images, num_classes), dtype=np.int32)

        k = 0
        # Loop over galaxy ids, if we have the file then add the galaxy to our dataset
        # The file may be missing or corrupt in which case we can skip
        for i in range(gal_id.shape[0]):
            filename = '../../images/img_{}.png'.format(gal_id[i])
            if os.path.isfile(filename):
                img = np.expand_dims(misc.imread(filename, flatten=1), axis=2)
                dataset[k] = img[50:150, 50:150, :]
                labels[k] = metadata[i, 3:6]
                k += 1
            if k == max_num_images:
                break

        # We might not have filled up the data structure.  Trim if needed so we don't have zeros at the end
        dataset = dataset[0:k]
        labels = labels[0:k]

        # Randomize the dataset
        np.random.seed(133)

        randomized_dataset, randomized_labels = randomize(dataset, labels)

        # Split into train, validation, test
        n = dataset.shape[0]
        train_cutoff = int(n * 0.8)
        valid_cutoff = int(n * 0.9)

        self.dataset = dict()
        self.labels = dict()

        self.dataset['train'] = randomized_dataset[0:train_cutoff, :, :, :]
        self.labels['train'] = randomized_labels[0:train_cutoff, :]
        self.dataset['valid'] = randomized_dataset[train_cutoff:valid_cutoff, :, :, :]
        self.labels['valid'] = randomized_labels[train_cutoff:valid_cutoff, :]
        self.dataset['test'] = randomized_dataset[valid_cutoff:, :, :, :]
        self.labels['test'] = randomized_labels[valid_cutoff:, :]

    def get_batch(self, set='train'):
        '''
        Get the next batch of data of size batch size.
        :param set: 'test', 'train' or 'valid'
        :return: a chunk of data
        '''
        offset = self.offset[set]
        i = range(offset, offset + batch_size)
        self.offset[set] = offset + batch_size
        return self.dataset[set].take(i, mode='wrap', axis=0), self.labels[set].take(i, mode='wrap', axis=0)

    def get_data_set(self, set):
        '''
        Get a whole dataset.  Discouraged.  Use get_batch instead.
        :param set: 'test', 'train' or 'valid'
        :return: a dataset
        '''
        return self.dataset[set], self.labels[set]

    def epoch(self, set):
        '''
        Iterate through the dataset exactly once, in batches
        :param set: 'test', 'train' or 'valid'
        :return: A generator
        '''
        offset = 0
        while offset < self.dataset[set].shape[0]:
            i = range(offset, offset + batch_size)
            offset += batch_size
            yield self.dataset[set].take(i, mode='clip', axis=0), self.labels[set].take(i, mode='clip', axis=0)