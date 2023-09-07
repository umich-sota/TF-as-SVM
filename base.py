import torch
import torch.nn as nn
import cvxpy as cp


class SelfAttn(nn.Module):
    
    def __init__(self, input_size, hidden_size=None, univariate=True):
        super(SelfAttn, self).__init__()
        
        self.univariate = univariate
        if hidden_size is None:
            hidden_size = input_size
        if univariate:
            self.W = nn.Linear(input_size, input_size, bias=False)
        else:
            self.query = nn.Linear(input_size, hidden_size, bias=False)
            self.key = nn.Linear(input_size, hidden_size, bias=False)
        self.value = nn.Identity()
        self.v = nn.Parameter(torch.randn(input_size) * 0.01)


    def forward(self, input_seq, cross_input=None, attn_idx=-1, cross_attn=False):
        if cross_attn and cross_input is None:
            raise ValueError('Set cross_attn==True but no cross input provided.')
        
        n, _ , d = input_seq.shape
        if not cross_attn:
            cross_input = input_seq[:, attn_idx]
        cross_input = cross_input.reshape(n,1,d)
        if self.univariate:
            out = torch.softmax(cross_input @ self.W(input_seq).transpose(-2, -1), dim=-1)
        else:
            Q = self.query(cross_input)
            K = self.key(input_seq)
            out = torch.softmax(Q @ K.transpose(-2, -1), dim=-1)
        self.sfx_out = out
        out = out @ self.value(input_seq)
        return out @ self.v
        


def W_svm_solver(X, ids, cross_input=None, attn_idx=-1, cross_attn=False, fro=False):
    if cross_attn and cross_input is None:
        raise ValueError('Set cross_attn==True but no cross input provided.')
    mode = 'fro' if fro else 'nuc'
    n, T, d = X.shape
    W = cp.Variable((d,d))

    Z = cross_input
    if not cross_attn:
        Z = X[:, attn_idx]

    constraints = []
    for i in range(n):
        t = 0
        for j in range(T):
            if j == ids[i]:
                continue
            constraints += [((X[i,j] - X[i,ids[i]]).reshape(1,-1) @ W @ (Z[i]).reshape(-1,1)) <= -1]
            t += 1

    prob = cp.Problem(cp.Minimize(cp.norm(W, mode)), constraints)
    prob.solve()

    return W.value