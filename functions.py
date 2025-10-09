import numpy as np
import itertools as it
from math import *
from numba import njit

## G-COBG ##

@njit(cache=True)
def prob_k(p_arr, subset_array, T):
    total_prob = 0.0
    num_subsets = subset_array.shape[0]
    subset_size = subset_array.shape[1]
    for s in range(num_subsets):
        subset = subset_array[s]
        prob = 0.0
        for j in range(1, 1 << subset_size):
            p_sum = 0.0
            bits_set = 0
            for i in range(subset_size):
                if (j >> i) & 1:
                    p_sum += p_arr[subset[i]]
                    bits_set += 1
            sign = (-1) ** (subset_size - bits_set)
            prob += sign * (p_sum ** T)
        total_prob += prob
    return total_prob

def N(p_arr, T):
    """
    Computes the exact distribution of the number of distinct elements in L i.i.d. draws from a categorical distribution with arbitrary probabilities.
    """
    p_arr = np.array(p_arr, dtype=np.float64)
    m = len(p_arr)
    max_k = min(m, T)
    dist = np.zeros(max_k, dtype=np.float64)
    for k in range(1, max_k + 1):
        subsets = list(it.combinations(range(m), k))
        subset_array = np.array(subsets, dtype=np.int32)
        dist[k - 1] = prob_k(p_arr, subset_array, T)
    return dist

def N_equiprobable(m, T):
    """
    Computes the exact distribution of number of distinct elements in L draws from N equiprobable elements.
    """
    max_k = min(m, T)
    dist = np.zeros(max_k, dtype=np.float64)
    for k in range(1, max_k + 1):
        incl_excl_sum = 0
        for j in range(0, k + 1):
            sign = (-1) ** j
            term = comb(k, j) * ((k - j) ** T)
            incl_excl_sum += sign * term
        prob = (comb(m, k) * incl_excl_sum) / (m ** T)
        dist[k - 1] = prob
    return dist

## U-COBG ##

@njit(cache=True)
def leaf_fn(path, f, m):
    fees = np.zeros(m + 1)
    for idx in range(len(path) - 1):
        val = path[idx]
        fees[val] += f[idx]
    return fees, 1.0

@njit(cache=True)
def aggregate_fn(v_mat, p_arr, lambda_arr, g, m):
    l = len(p_arr)
    if l == m + 1:
        l_arr = np.roll(lambda_arr, -1)
    else:
        l_arr = lambda_arr[1:]
    v_arr = np.diag(v_mat)
    c = v_arr > g
    p = 0.0
    for i in range(l):
        p += l_arr[i] * c[i] * p_arr[i]
    v = np.zeros(m)
    for i in range(m):
        temp_sum = 0.0
        for j in range(l):
            temp_sum += v_mat[j, i] * l_arr[j] * c[j]
        v[i] = temp_sum + g * (l_arr[i] * (1 - c[i]))
    return v, p

@njit(cache=True)
def recurse(path, current_depth, T, lambda_arr, f, g, m):
    if current_depth == T + 1:
        return leaf_fn(path, f, m)
    elif current_depth == T:
        child_v = np.zeros((m + 1, m + 1))
        child_p = np.zeros(m + 1)
        for i in range(m + 1):
            new_path = path.copy()
            new_path[current_depth] = i
            v, p = recurse(new_path, current_depth + 1, T, lambda_arr, f, g, m)
            child_v[i, :] = v
            child_p[i] = p
        aggr = aggregate_fn(child_v, child_p, lambda_arr, g, m)
        return aggr
    else:
        child_v = np.zeros((m, m))
        child_p = np.zeros(m)
        for i in range(m):
            new_path = path.copy()
            new_path[current_depth] = i
            v, p = recurse(new_path, current_depth + 1, T, lambda_arr, f, g, m)
            child_v[i, :] = v
            child_p[i] = p
        aggr = aggregate_fn(child_v, child_p, lambda_arr, g, m)
        return aggr

def probability(lambda_arr, f, g):
    """
    Computes the censoring probability of a transaction giving fee g by a transaction with fee vector f, given hashrate distribution lambda_arr.
    """
    m = len(lambda_arr) - 1
    T = len(f) - 1
    path = np.zeros(T + 2, dtype=np.int64)
    _, result_p = recurse(path, 0, T, lambda_arr, f, g, m)
    return result_p

def random_fees(T, F2):
    breaks = np.sort(np.random.uniform(0, F2, T))
    points = np.concatenate(([0], breaks, [F2]))
    return np.diff(points)

def optimal_probability(lambda_arr, T, k, R = 1000, logs = True):
    """
    Tries to find a fee vector f with given bribe budget F2=k*f1 which yields the highest possible censoring probability,
    for a given hashrate distribution lambda_arr. Does so by trying R random fee vectors.
    """
    if logs:
        max_f = np.ones(T + 1) * k / (T + 1)
        max_prob = probability(lambda_arr, max_f, 1)
        print("ctr    prob    | array")
        print(f"{0:5d} {f"{max_prob:.3f}":8} {"|"} {' '.join(f'{x:8}' for x in [f"{x:.3f}" for x in max_f])}")

        for i in range(1, R + 1):
            f = random_fees(T, k)
            prob = probability(lambda_arr, f, 1)
            if prob >= max_prob:
                max_f = f
                max_prob = prob
                print(f"{i:5d} {f"{max_prob:.3f}":8} {"|"} {' '.join(f'{x:8}' for x in [f"{x:.3f}" for x in max_f])}")
    else:
        max_f = np.ones(T + 1) * k / (T + 1)
        max_prob = probability(lambda_arr, max_f, 1)

        for i in range(1, R + 1):
            f = random_fees(T, k)
            prob = probability(lambda_arr, f, 1)
            if prob >= max_prob:
                max_f = f
                max_prob = prob
        return max_prob
    
def rho(lambda_arr, f2, f1):
    """
    Compute the values of rho as in Theorem 4.6, for given hashrate distribution lambda_arr, when trying to censor a transaction with fee f1 using
    a transaction with fee f2.
    """
    def a(lambda_arr, j):
        return np.log(np.sum(lambda_arr[j:]))

    def b(f1, f2, lambda_arr, j):
        return np.log(f1 / f2 / lambda_arr[j])

    def ell(f1, f2, lambda_arr):
        return np.sum(lambda_arr[1:] < f1 / f2)
    
    m = len(lambda_arr) - 1
    rho_arr = np.zeros(m + 1)
    l = ell(f1, f2, lambda_arr)
    s = 0
    for j in range(l + 1, m + 1):
        rho_arr[j] = np.ceil(rho_arr[j - 1]) + (b(f1, f2, lambda_arr, j) - s) / a(lambda_arr, j)
        s += (np.ceil(rho_arr[j] - np.ceil(rho_arr[j - 1]))) * a(lambda_arr, j)
    return np.ceil(rho_arr)
