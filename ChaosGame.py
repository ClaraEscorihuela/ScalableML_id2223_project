import numpy as np
import random

def listToString(s):
    """ List to String conversion """
    str1 = ""
    return (str1.join(s))


def list_in(a, b):
    """ Check string inside another string"""
    return any(map(lambda x: b[x:x + len(a)] == a, range(len(b) - len(a) + 1)))


def create_dummy_junction(dummy_junction, random_elements):
    """ modified dummy_junction creation"""
    size = len(dummy_junction)

    num_random_nucleotids = 0  # number of random nucleotid sequences
    if random_elements == 'one':
        num_random_nucleotids = 1
    elif random_elements == 'multiple':
        num_random_nucleotids = random.randint(0, 1)

    if num_random_nucleotids > 0:  # If I want to add nucleotids
        dummy_junction_modified = dummy_junction

        for i in range(num_random_nucleotids):
            bases = 'ATGC'
            if random_elements == 'one':
                num_random_nucleotids = 1
                print(num_random_nucleotids)
            elif random_elements == 'multiple':
                num_nucseq = random.randint(1, 2)  # Number of nucleotids to include in the sequence of new nucleotids
            position = random.randint(1, size)  # Position to iclude the new sequences of nucleotids
            nucleotid_random = random.choices(bases, k=num_nucseq)  # Random bases to include
            n_r_string = listToString(nucleotid_random)  # Convert the list into a string
            dummy_junction_modified = dummy_junction_modified[:position] + n_r_string + dummy_junction_modified[
                                                                                        position:]  # Include the new sequences in the previous one
            print('dummy_junction', dummy_junction, 'pos', position, 'num_nuc_seq', num_nucseq, 'n', nucleotid_random,
                  'd_j_m', dummy_junction_modified)

        print("The sequences has been modified from {} to {} ".format(dummy_junction, dummy_junction_modified))

        return dummy_junction_modified

        # if a do not want to add nucleotids
    else:
        return dummy_junction



def elongate_seq(seq, desired_seq_len):
    """ Sequences elongation by random base addition"""

    original_len = len(seq)
    pad_len = desired_seq_len - original_len

    if pad_len <= 0:
        print("Initial sequence is larger than the desired length")
        final_seq = seq
    else:
        bases = 'ATGC'
        pad_seq = random.choices(bases, k=pad_len)
        pad_seq = ''.join(pad_seq)
        i = random.randint(1, pad_len - 1)
        prev = pad_seq[:i]
        post = pad_seq[i:]
        final_seq = prev + seq + post

    return final_seq


def create_dataset(dummy_junction, dummy_no_junction, length, num_of_seq, random_elements='no'):
    """ Dataset creation"""

    if random_elements != 'no' and random_elements != 'multiple' and random_elements != 'one':
        print('Random parameter can only be set to: no, one or multiple. Otherwise, value *no will be used as default')

    fake_seq = []
    fake_lbl = []

    for i in range(round(num_of_seq / 2)):
        d_j = create_dummy_junction(dummy_junction, random)
        new_seq = elongate_seq(d_j, length)
        fake_seq.append(new_seq)
        fake_lbl.append(1)

    error = 0
    for i in range(round(num_of_seq / 2)):
        new_seq = elongate_seq(dummy_no_junction, length)
        if list_in(dummy_junction, new_seq):
            error = error + 1
            fake_lbl.append(1)
        else:
            fake_lbl.append(0)
        fake_seq.append(new_seq)

    # Check error
    error2 = 0
    for i in range(round(num_of_seq / 2)):
        if list_in(dummy_junction, new_seq):
            error2 = error2 + 1
            print("error2")

    fake_lbl = np.array(fake_lbl)

    return fake_seq, fake_lbl, error2