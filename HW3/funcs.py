import numpy as np
from scipy.sparse import diags
from scipy.linalg import block_diag

def block_tridiag_old(a, b, c, M, N): 
    # builds a (N x M) x (M x N) block tridiagonal matrix
    # this algorithm took me an unbelievable amount of time (approx. 5 hours) to figure out... 
    # it's useless because I misunderstood the matrix setup for 2D C-N
    # but I've grown attached to it through the struggle and now I care for it too much to say goodbye..
    Q = np.zeros([M*N, M*N])

    A = diags([0, a, 0], [-1,0,1], shape=(N, M)).toarray()
    B = diags([b, c, b], [-1,0,1], shape=(N, M)).toarray()
    C = diags([0, a, 0], [-1,0,1], shape=(N, M)).toarray()
    
    if M < N:
        A = block_diag(*([A]*M))
        C = block_diag(*([C]*(M-1)))
    elif M > N:   
        A = block_diag(*([A]*(N-1)))
        C = block_diag(*([C]*N))  
    else: 
        A = block_diag(*([A]*(M-1)))
        C = block_diag(*([C]*(M-1)))

    B = block_diag(*([B]*(min(M, N))))
    Q[0:B.shape[0], 0:B.shape[1]] += B
    Q[0:A.shape[0], M:M+A.shape[1]] += A
    Q[N:N+C.shape[0], 0:C.shape[1]] += C

    return Q

def block_tridiag(a, b, c, M, N):
    # builds the correct (M x N) x (M x N) block diagonal matrix. it's really a one-liner...
    return diags([a, b, c, b, a], [-M, -1, 0, 1, M], shape=(M*N,M*N)).toarray()

def tridiag(a, b, N):
    # builds the (N x N) tridiagonal matrix
    return diags([a, b, a], [-1, 0, 1], shape=(N, N)).toarray()

def visualize_boundary(B_0, M):
    for i in range(len(B_0)):
        val = "{0:.2f}".format(B_0[i])
        print("%4s" % val, end=" ")
        if (i+1) % M == 0: # right boundary
            print('')

def TDMAsolver(a, b, c, d): # https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = d.size # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc