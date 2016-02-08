import numpy as np
import requests
import shutil
from scipy import misc
import matplotlib.pyplot as plt
import os
import sys

'''code for getting scraped images from sdss skyserver'''


def read_galaxy_table(filename):
    '''read in table containing galaxy positions'''

    gal_id = np.genfromtxt(filename, delimiter=',', dtype=int, usecols=[0])    # loadtxt doesn't read in large number correctly 
    f = np.loadtxt(filename, delimiter=',')
    ra = f[:,1]
    dec = f[:,2]
    return gal_id, ra, dec


def create_url(ra, dec):
    '''create url string given ra and dec of galaxy'''

    url = "http://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx?ra={0}&dec={1}&width=200&height=200".format(ra, dec)
    return url


def create_outfile(gal_id, imgdir):
    '''create name of outfile to save image to'''

    outfile = imgdir + "img_{0}.png".format(gal_id)
    return outfile


def get_img(url, outfile):
    '''get img from url'''
    if not os.path.isfile(outfile):
        response = requests.get(url, stream=True)
        with open(outfile, 'wb') as outfile:
            shutil.copyfileobj(response.raw, outfile)
        return True
    return False
	

def plot_img(img):
    '''plot one image'''

    g = misc.imread(img)
    plt.imshow(g)
    plt.show()


def maybe_download_indices(a, b, imgdir="../images/"):
    print "Maybe downloading images files indexed {0} to {1}".format(a, b)
    gal_id, ra, dec = read_galaxy_table('../data/gal_pos_label.txt')
    # get images in the range 32000 to 35000 -- can make this an argument
    downloaded = 0
    for i in range(a, b):
        url = create_url(ra[i], dec[i])
        outfile = create_outfile(gal_id[i], imgdir)
        downloaded += get_img(url, outfile)
        if i % 1000 == 0:
            print "{0} / {1} complete".format(i, b-a)
    print "{0} / {1} complete".format(b-a, b-a)
    print "Finished maybe download.  {0} downloaded, the rest already exist on disk.".format(downloaded)


if __name__ == '__main__':
    '''
    First argument is the number of images to ensure we have locally,
    second argument is the directory to download them to.
    '''
    n = int(sys.argv[1])
    if len(sys.argv) > 2:
        imgdir = sys.argv[2]
        maybe_download_indices(0, n, imgdir)
    else:
        maybe_download_indices(0, n)