import collections
import math
import numpy as np
import copy

def count_kmers(k, r, data):
    """Count the number of times a little sequences appear in a big sequence"""
    r += 1

    d_1 = collections.defaultdict(int)
    d_2 = collections.defaultdict(int)
    d_3 = collections.defaultdict(int)
    for i in range(len(data) - (k - 2)):
        oligo_1 = data[i:i + k]
        oligo_2 = data[i] + data[i + 2:i + k + 1]
        oligo_3 = data[i] + data[i + 3:i + k + 2]
        d_1[oligo_1] += 1  # A la seq que hem definit com k-oligo em sumes un
        d_2[oligo_2] += 1
        d_3[oligo_3] += 1
    d_copy_1 = copy.copy(d_1)
    d_copy_2 = copy.copy(d_2)
    d_copy_3 = copy.copy(d_3)

    # Eliminate random nucleotids that might be present in the DNA sequence, denoted as N
    for key in d_1.keys():
        if "N" in key:
            del d_copy_1[key]
    for key in d_2.keys():
        if "N" in key:
            del d_copy_2[key]
    for key in d_3.keys():
        if "N" in key:
            del d_copy_3[key]

    return d_copy_1, d_copy_2, d_copy_3, r


def probabilities(kmer_count, k, data):
    """calculate the probability of a little sequence to be present"""
    probabilities = collections.defaultdict(float)
    N = len(data)
    for key, value in kmer_count.items():
        probabilities[key] = float(value) / (N - k + 1)  # N-K+1 possible words
    return probabilities


def chaos_game_representation(probabilities, k):
    """Returns the value in a matrix (which simulates the image of the dna)"""
    array_size = int(math.sqrt(4 ** k))
    chaos = []
    for i in range(array_size):
        chaos.append([0] * array_size)

    maxx = array_size
    maxy = array_size
    posx = 1
    posy = 1
    for key, value in probabilities.items():
        for char in key:
            if char == "T":
                posx += maxx / 2
            elif char == "C":
                posy += maxy / 2
            elif char == "G":
                posx += maxx / 2
                posy += maxy / 2
            maxx = maxx / 2
            maxy /= 2
        chaos[int(posy - 1)][int(posx - 1)] = value
        maxx = array_size
        maxy = array_size
        posx = 1
        posy = 1

    return chaos


def create_images(dataset, k):
    """Create a matrix with all the DNA images"""
    r = 0
    # Bucle to generate all the chaos game images
    input_image = []
    for i in range(len(dataset)):
        channels = []
        data_signal = dataset[i]
        f_1, f_2, f_3, r = count_kmers(k, r, data_signal)
        f_prob_1 = probabilities(f_1, k, data_signal)
        f_prob_2 = probabilities(f_2, k, data_signal)
        f_prob_3 = probabilities(f_3, k, data_signal)
        chaos_k_1 = chaos_game_representation(f_prob_1, k)
        chaos_k_2 = chaos_game_representation(f_prob_2, k)
        chaos_k_3 = chaos_game_representation(f_prob_3, k)

        channels.append(np.array(chaos_k_1))
        channels.append(np.array(chaos_k_2))
        channels.append(np.array(chaos_k_3))
        input_image.append(np.array(channels))

    input_image = np.array(input_image)
    print(np.shape(input_image))
    return input_image
