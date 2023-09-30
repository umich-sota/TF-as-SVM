import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp


class SelfAttn(nn.Module):
    
    def __init__(self, input_size, hidden_size=None, univariate=True, mode='relu'):
        super(SelfAttn, self).__init__()
        
        self.univariate = univariate
        self.mode = mode
        if hidden_size is None:
            hidden_size = input_size
        if univariate:
            self.W = nn.Linear(input_size, input_size, bias=False)
        else:
            self.query = nn.Linear(input_size, hidden_size, bias=False)
            self.key = nn.Linear(input_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.randn(input_size) * 0.01)
        if mode == 'relu':
            self.value = nn.Linear(input_size, hidden_size, bias=False)
            self.act_func = nn.ReLU()
        else:
            self.value = nn.Identity()
            self.act_func = nn.Identity()


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
        if self.mode == 'lambda':
            if not hasattr(self, 'lbd'):
                raise ValueError('Set mode==lambda but no lambda value provided.')
            input_seq_ = self.act_func(out @ self.value(input_seq))
            out = input_seq_ @ self.v - self.lbd * input_seq_.norm(dim=-1)**2
        else:
            out = out @ self.value(input_seq)
            out = self.act_func(out) @ self.v
        return out 
        

def W_svm_solver(X, sfx_out, W_grad, cross_input=None, attn_idx=-1, cross_attn=False, fro=False, THRED=1e-3):
    if cross_attn and cross_input is None:
        raise ValueError('Set cross_attn==True but no cross input provided.')
    mode = 'fro' if fro else 'nuc'
    n, T, d = X.shape
    
    Z = cross_input
    if not cross_attn:
        Z = X[:, attn_idx]
    
    sfx_out = sfx_out.reshape(n,T)
    ids = sfx_out >= THRED

    W_fin = cp.Variable((d,d))
    W_dir = cp.Variable((d,d))

    constraints_fin = []
    constraints_dir = []

    for i in range(n):
        for t1 in range(T):
            for t2 in range(t1+1,T):
                if ids[i,t1] and ids[i,t2]:
                    constraints_fin += [((X[i,t2] - X[i,t1]).reshape(1,-1) @ W_fin @ (Z[i]).reshape(-1,1)) == np.log(sfx_out[i,t2]/sfx_out[i,t1])]
                    constraints_dir += [((X[i,t1] - X[i,t2]).reshape(1,-1) @ W_dir @ (Z[i]).reshape(-1,1)) == 0]
                elif ids[i,t1] and not ids[i,t2]:
                    constraints_dir += [((X[i,t2] - X[i,t1]).reshape(1,-1) @ W_dir @ (Z[i]).reshape(-1,1)) <= -1]
                elif not ids[i,t1] and ids[i,t2]:
                    constraints_dir += [((X[i,t1] - X[i,t2]).reshape(1,-1) @ W_dir @ (Z[i]).reshape(-1,1)) <= -1]
    
    prob_fin = cp.Problem(cp.Minimize(cp.norm(W_fin, mode)), constraints_fin)
    prob_fin.solve()
    prob_dir = cp.Problem(cp.Minimize(cp.norm(W_dir, mode)), constraints_dir)
    prob_dir.solve()    

    R = cp.Variable(1)
    W_fin = np.array(W_fin.value)
    W_dir = np.array(W_dir.value)
    prob_R = cp.Problem(cp.Minimize(cp.norm(W_fin + R * W_dir-W_grad, 'fro')))
    prob_R.solve()

    return W_dir, W_fin + R.value * W_dir