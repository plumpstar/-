from math import floor
import random


def next_prime():
    '''的到下一个值'''
    def is_prime(num):
        "判断是不是需要的值"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True

    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2


def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def halton_sequence(size, dim):
    '''Halton序列'''
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return seq

def halton_sample(i, dim):
    '''第i个'''
    seqs = halton_sequence(i, dim)

    sample = []
    for seq in seqs:
        sample += [seq[-1]]

    return sample

def random_sample(dim):
    '''均匀随机样本'''
    sample = []
    for i in range(dim):
        sample += [random.random()]

    return sample


def get_sample(i, method, types=[], domains=[], dims=[1]):
    '''获取样本并使其适应领域'''

    assert len(types) == len(domains)

    samples = []
    norm_samples = []

    for k in range(len(dims)):

        d = dims[k]

        # 采样
        if method == 'random':
            sample = random_sample(d)
        if method == 'halton':
            sample = halton_sample(i, d)

        typ = types[k]
        domain = domains[k]
        norm_sample = []

        for j in range(d):

            sample[j] = sample[j]*(domain[1]-domain[0])+domain[0]

            if typ == 'int':
                sample[j] = int(sample[j])
            elif typ == 'float':
                sample[j] = float(sample[j])

            norm_sample.append((sample[j]-domain[0])/float(domain[1]-domain[0]))

        samples += [sample]
        norm_samples += [norm_sample]

    return samples, norm_samples
