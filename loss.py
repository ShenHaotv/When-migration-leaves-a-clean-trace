import torch

def loss_wrapper(z, adj, X, Y, C1, C2, n,deg,lamb):
    """
    Compute loss and gradient for L-BFGS optimization (non-zero elemtents only).
    
    Args:
        z: 1D numpy array of non-zero elements
        adj: adjacency matrix
        X: (n, n) matrix
        Y: (n, n) matrix
        C1: (o1,n) row selection matrix
        C2: (o2,n) column selection matrix
        t: Scalar (time)
        n: Dimension of the matrix
        deg:degree of each node
        lamb: hyperparameter
    
    Returns:
        Tuple of (loss_value, grad_flat) where:
            loss_value: Python float of the loss
            grad_flat: Flattened gradient of non-zero elements
    """
    # Reconstruct theta matrix 
    torch.set_printoptions(precision=8) 
    z_torch = torch.tensor(z, requires_grad=True, dtype=torch.float64)
    m_torch= torch.nn.functional.softplus(z_torch)
    g=torch.log(m_torch)                                                                    
    deg_torch=torch.tensor(deg, dtype=torch.float64)
    C1_torch=torch.tensor(C1, dtype=torch.float64)
    C2_torch=torch.tensor(C2, dtype=torch.float64)

    # Convert to PyTorch tensors
    X_torch = torch.tensor(X, dtype=torch.float64)
    Y_torch = torch.tensor(Y, dtype=torch.float64)
    
    # Get indices of non-zero entries in adj (sparse-friendly)
    rows, cols = torch.where(torch.tensor(adj != 0, dtype=torch.bool))
    
    # Construct W directly from theta_flat (exp(theta) for non-zero entries)
    W = torch.zeros((n, n), dtype=torch.float64)
    W[rows, cols] =m_torch  # Only non-zero entries are used
    
    G=torch.zeros((n, n), dtype=torch.float64)
    G[rows, cols] =g
    Gsquare=G**2

    a= torch.sum(Gsquare + Gsquare.T, dim=1)
    b= torch.sum(G + G.T, dim=1)
    pen = a@ (1/deg_torch) - (b/deg_torch)@(b/deg_torch)
    
 
    # Construct Laplacian
    D = torch.diag(W.sum(dim=1))
    L = D - W
     
    # Compute loss
    exp= torch.matrix_exp(-L)
    relative_error=(exp@X_torch-Y_torch)/Y_torch
    residual=C1_torch@relative_error@C2_torch.T
    
    torch.set_printoptions(precision=8) 
    loss = torch.norm(residual, p='fro')**2+lamb*pen
   
    # Backpropagate
    loss.backward()
    
    # Extract only off-diagonal gradients
    grad_z=z_torch.grad.numpy()
    
   
    return loss.item(), grad_z