    norm_A = np.max(np.abs(A).sum(axis=1))
    norm_inv_A = np.max(np.abs(inv(A)).sum(axis=1))