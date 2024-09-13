from inc.diffus import *
from inc.nn import *
from inc.test import *
import pdb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, help = 'dataset name')
    parser.add_argument('--seed', type = int, help = 'random seed')
    parser.add_argument('--data_dir', type = str, help = 'dataset folder')
    parser.add_argument('--output', type = str, help = 'output file name')
    parser.add_argument('--device', type = torch.device, help = 'torch device')
    parser.add_argument('--b_pI0', type = float, help = 'initial infection rate in diffusion parameter estimation')
    parser.add_argument('--b_pR0', type = float, help = 'initial recovery rate in diffusion parameter estimation')
    parser.add_argument('--b_steps', type = int, help = 'optimization steps in diffusion parameter estimation')
    parser.add_argument('--b_lr', type = float, help = 'learning rate in diffusion parameter estimation')
    parser.add_argument('--q_steps', type = int, help = 'training steps for the proposal model')
    parser.add_argument('--q_lr', type = float, help = 'learning rate for the proposal model')
    parser.add_argument('--q_hid', type = int, help = 'hidden size of the proposal model')
    parser.add_argument('--q_gnn', type = int, help = 'number of layers of the GNN in the proposal model')
    parser.add_argument('--q_mlp', type = int, help = 'number of layers of the MLP in the proposal model')
    parser.add_argument('--q_samples', type = int, help = 'sample size to estimate the loss function of the proposal model')
    parser.add_argument('--q_zlim', type = int, help = 'a hyperparameter to stablize gradient')
    parser.add_argument('--p_coef', type = float, help = 'the coefficient gamma in the initial distribution P[y_0]')
    parser.add_argument('--t_samples', type = int, help = 'MCMC sample size')
    parser.add_argument('--t_steps', type = int, help = 'MCMC steps')
    parser.add_argument('--t_keep', type = float, help = 'moving average in MCMC')
    parser.add_argument('--debug', action='store_true', help='enable debug mode to skip training and only sample')
    args = parser.parse_args()
    return args

class QNet(nn.Module):
    @classmethod
    def make(cls, data, args):
        n_obs = len(data.obs)
        print(f"n_obs in make: {n_obs}")
        return cls(
            eidx=data.edge_index,
            T=data.T.item(),
            hid=args.q_hid,
            gnn=args.q_gnn,
            mlp=args.q_mlp,
            n_nodes=data.num_nodes,
            zlim=args.q_zlim,
            n_obs=n_obs  # Added parameter
        ).to(args.device)

    def __init__(self, eidx, T, hid, gnn, mlp, n_nodes, zlim, n_obs):  # Added parameter
        super().__init__()
        self.eidx = eidx
        self.device = self.eidx.device
        self.n_nodes = n_nodes
        self.n_inf = self.n_nodes + 2
        self.n_edges = self.eidx.size(dim=1)
        self.zlim = zlim
        self.T = T
        self.hid = int(hid)
        self.gnn_dep = int(gnn)
        self.mlp_dep = int(mlp)
        self.n_obs = n_obs  # Added parameter
        self.w = nn.Parameter(data=torch.randn((self.n_edges, self.hid), dtype=torch.float32, device=self.device), requires_grad=True)
        self.gnn = GNN(v_in=n_obs, e_in=self.hid, hid=self.hid, dep=self.gnn_dep)  # Modified
        self.mlp = MLP([self.hid] * self.mlp_dep + [2 * self.T * n_nodes])
        self.rem = (pyg.utils.degree(self.eidx[1], num_nodes=self.n_nodes).long().unsqueeze(dim=1) + 1).detach().clone()  # (nodes, 1)
        self.neighbs = [[] for u in range(self.n_nodes)]
        for i in range(self.n_edges):
            self.neighbs[self.eidx[0, i].item()].append(self.eidx[1, i].item())
        for u in range(self.n_nodes):
            self.neighbs[u] = torch.tensor(self.neighbs[u], dtype=torch.long, device=self.device)
        self.zero = torch.tensor(0., dtype=torch.float, device=self.device)
        print(f"MLP structure: {self.mlp.units_list}")

    def clamp_z(self, z):
        return z.clamp(-self.zlim, self.zlim)
    
    def forward(self, y, orig=False):
        print(f"Input y shape: {y.shape}")
        print(f"self.n_obs: {self.n_obs}")

        n_samples, n_nodes, _ = y.size()
        y = y.transpose(0, 1).reshape((-1,self.n_obs))

        eidx = self.eidx.repeat(1, n_samples)
        eidx[0] += torch.arange(n_samples, device=y.device).repeat_interleave(self.eidx.size(1)) * n_nodes
        eidx[1] += torch.arange(n_samples, device=y.device).repeat_interleave(self.eidx.size(1)) * n_nodes
        print(f"eidx shape: {eidx.shape}")
        
        w = self.w.repeat(n_samples, 1)  # (samples*edges, hid)
        print(f"w shape: {w.shape}")
        
        
        z, e = self.gnn(y.float(), eidx, w)
        pdb.set_trace()
        z = self.mlp(z)  # (samples*nodes, 2*T)
        print(f"MLP output shape: {z.shape}")

        z = z.T.reshape((2 * self.T, n_samples, -1))

        print(f"z shape after reshape and permute: {z.shape}")
        
        zI, zR = z[: self.T], z[self.T :] # (T, samples, nodes)
        print(f"Final output shapes - zI: {zI.shape}, zR: {zR.shape}")
        
        if orig:
            return zI, zR, self.clamp_z(zI), self.clamp_z(zR)
        else:
            return self.clamp_z(zI), self.clamp_z(zR)

    def compute_transition_mask(self, t, Y, rem):
        """
        A helper function for lik, to compute the transitoin mask:
        rem_{t, i} > 1 && for all j in N(i): rem_{t, j} > 1
        parameters:
        t: current time step
        Y: shape (T+1, n_nodes, n_samples)
        rem: remaining infection tensor
        """
        n_nodes, n_samples = Y.shape[1], Y.shape[2]
    
        transition_mask = torch.zeros((n_nodes, n_samples), dtype=torch.bool, device=self.device)
        
        infected = (Y[t] == SIR_STATES.I)
        
        # Set cover
        uncovered = infected.clone()
        while uncovered.any():
            coverage_count = torch.zeros((n_nodes, n_samples), dtype=torch.long, device=self.device)
            for i in range(n_nodes):
                if len(self.neighbs[i]) > 0:
                    coverage_count[i] = (uncovered[self.neighbs[i]] & (rem[t, self.neighbs[i]] > 1)).sum(dim=0)
            
            best_coverage, best_node = coverage_count.max(dim=0)
            
            if best_coverage.max() == 0:
                break
            
            for i in range(n_samples):
                if best_coverage[i] > 0:
                    node = best_node[i]
                    transition_mask[node, i] = True
                    uncovered[self.neighbs[node], i] &= ~(rem[t, self.neighbs[node], i] > 1)
        
        # include nodes that can transition based on remaining infections
        transition_mask |= (rem[t] > 1)
        
        return transition_mask

    def lik(self, Y, obs): # Y: (T+1, nodes, samples)
        n_samples = Y.size(dim=2)
        n_obs = len(self.obs)
        print(f"n samples: {n_samples.shape}")
        print(f"Y shape in lik: {Y.shape}")
        # [11,1000,10]
        # Y[0] is the number of time steps, Y[1] is the number of nodes, Y[2] is the number of samples

        zI0, zR0, zI, zR = self.forward(Y[self.obs], orig = True) # (T, nodes, samples)

        zI = zI.clone().detach().requires_grad_(True); zI.retain_grad()
        zR = zR.clone().detach().requires_grad_(True); zR.retain_grad()
        # R->I
        qR = torch.sigmoid(zR) # (T, nodes, samples) # prob of R->I
        lR1 = torch_log(qR) # (T, nodes, samples)
        lR0 = torch_log(1. - qR) # (T, nodes, samples)
        with torch.no_grad():
            mskR = (Y[self.obs[-1]] == SIR_STATES.R) # (T, nodes, samples)
            trsR = (Y[self.obs[:-1]] != SIR_STATES.R) # (T, nodes, samples)
        # I->S
        zI_, uid = zI.sort(dim = 1, descending = True) # (T, nodes, samples)
        qI = torch.sigmoid(zI_) # (T, nodes, samples) # prob of I->S
        lI1 = torch_log(qI) # (T, nodes, samples)
        lI0 = torch_log(1. - qI) # (T, nodes, samples)
        with torch.no_grad():
            mskI = ((Y[1 :] >= SIR_STATES.I) & (Y[: -1] <= SIR_STATES.I)).flatten() # (T * nodes * samples)
            trsI = ((Y[: -1] != SIR_STATES.I)).flatten() # (T * nodes * samples)
            rem = torch.where(mskI, self.rem.expand(self.T, -1, n_samples).flatten(), self.n_inf) # (T * nodes * samples)
            ptr = torch.arange(self.T, dtype = torch.long, device = self.device).unsqueeze(dim = 1) * self.n_nodes # (T, 1)
            for i in range(uid.size(dim = 1)): # I-S 转移的概率
                # 需要判断 *1 还是 qI, 如果不是概率1转移，就是qI， T之后就是1-qI
                uidi = (ptr + uid[:, i]).flatten() * n_samples # (T * samples)
                mski = mskI[uidi] # (T * samples)
                if mski.max():
                    trsi = trsI[uidi] # (T * samples)
                    vids, degi = [], [0]
                    for t in range(self.T):
                        for j in range(n_samples):
                            u = uid[t, i, j]
                            vids.append((t * self.n_nodes + u.unsqueeze(dim = 0)) * n_samples)
                            vid = self.neighbs[u.item()]
                            vids.append((t * self.n_nodes + vid) * n_samples)
                            degi.append(vid.size(dim = 0) + 1)
                    vids = torch.cat(vids, dim = 0) # (T * sum neighbs)
                    degi = torch.tensor(degi, dtype = torch.long, device = self.device) # (1 + T * samples)
                    indptr = degi.cumsum(dim = 0) # (1 + T * samples)
                    degi = degi[1 :] # (T * samples)
                    rems = rem.flatten()[vids] # (T * sum neighbs)
                    opti = (pysc.segment_min_csr(src = rems, indptr = indptr)[0] > 1) # (T * samples)
                    rem.flatten()[vids] = torch.where(mski.repeat_interleave(repeats = degi), torch.where(trsi.repeat_interleave(repeats = degi), rems - 1, self.n_inf), rems) # (T * sum neighbs)
                    mskI[uidi] &= opti # (T * samples)
        # likR + likI
        lik_forward = (torch.where(mskR, torch.where(trsR, lR1, lR0), self.zero).view(-1, n_samples) + torch.where(mskI.view(-1, n_samples), torch.where(trsI.view(-1, n_samples), lI1.view(-1, n_samples), lI0.view(-1, n_samples)), self.zero)).sum(dim = 0) # (samples,)
        
        # backward sampling likelihood
        lik_backward = self.zero

        for t in range(self.T - 1, -1, -1):
            transition_mask = self.compute_transition_mask(t, Y, rem)
            
            # R -> I or R -> S
            mask_R = (Y[t+1] == SIR_STATES.R)
            lik_backward += torch.where(mask_R & (Y[t] == SIR_STATES.I), torch.log(qR[t]), self.zero).sum(dim=0)
            lik_backward += torch.where(mask_R & (Y[t] == SIR_STATES.S), 
                                        torch.log(qR[t] * qI[t] * transition_mask), 
                                        self.zero).sum(dim=0)
            
            # I -> S or I -> I
            mask_I = (Y[t+1] == SIR_STATES.I)
            lik_backward += torch.where(mask_I & (Y[t] == SIR_STATES.S), 
                                        torch.log(qI[t] * transition_mask), 
                                        self.zero).sum(dim=0)
            lik_backward += torch.where(mask_I & (Y[t] == SIR_STATES.I), 
                                        torch.log(1 - qI[t] * transition_mask), 
                                        self.zero).sum(dim=0)
            
            # Update rem for the next iteration
            trs = (Y[t] != SIR_STATES.I)
            rem[t-1] = torch.where((Y[t] == SIR_STATES.I), 
                                torch.where(trs, rem[t] - 1, self.n_inf), rem[t])
    
        lik = lik_forward + lik_backward
        return lik, zI0, zR0, zI, zR # (samples,)
    
    @torch.no_grad()
    def clamp_grad(self, z0, grad):
        return torch.where(z0 < self.zlim, torch.where(z0 > -self.zlim, grad, F.relu(grad)), -F.relu(-grad))
    def backward(self, loss, zI0, zR0, zI, zR):
        loss.backward()
        z0 = torch.stack([zI0, zR0], dim = 0)
        z0.backward(torch.stack([self.clamp_grad(zI0, zI.grad), self.clamp_grad(zR0, zR.grad)], dim = 0))



    @torch.no_grad()
    def samp(self, y, zI, zR, n_samples, data, compute_lik=False):
        zI, uid = zI.sort(dim=1, descending=True)
        uid = uid.squeeze(dim=2)

        qI = torch.sigmoid(zI)
        xI = SIR_STATES.I - qI.expand(-1, -1, n_samples).bernoulli().long()
        lI = torch_log(torch.where(xI != SIR_STATES.I, qI, 1. - qI))

        qR = torch.sigmoid(zR)
        xR = SIR_STATES.R - qR.expand(-1, -1, n_samples).bernoulli().long()
        lR = torch_log(torch.where(xR != SIR_STATES.R, qR, 1. - qR))

        all_Y = torch.empty(self.T+1, self.n_nodes, n_samples, dtype=y.dtype, device=y.device)
        all_Y[data.obs] = data.y[:, data.obs].T.unsqueeze(dim = -1)

        if compute_lik:
            lik = self.zero

        for i, s in enumerate(data.obs):
            y = all_Y[s].unsqueeze(dim=1).expand(-1, n_samples)

            if i == 0:
                for t in range(s - 1, -1, -1):
                    msk = (y == SIR_STATES.R)
                    y = torch.where(msk, xR[t], y)
                    if compute_lik:
                        lik = lik + torch.where(msk, lR[t], self.zero).sum(dim=0)

                    msk = (y == SIR_STATES.I)
                    rem = torch.where(msk, self.rem, self.n_inf)
                    for idx, u in enumerate(uid[t]):
                        if msk[u].max():
                            vid = self.neighbs[u.item()]
                            opt = (rem[u] > 1) & (rem[vid].min(dim=0).values > 1)
                            msk_opt = msk[u] & opt
                            y[u] = torch.where(msk_opt, xI[t, idx], y[u])
                            trs = (y[u] != SIR_STATES.I)
                            rem[u] = torch.where(msk[u], torch.where(trs, rem[u] - 1, self.n_inf), rem[u])
                            rem[vid] = torch.where(msk[u].unsqueeze(dim=0), torch.where(trs.unsqueeze(dim=0), rem[vid] - 1, self.n_inf), rem[vid])
                            msk[u] = msk_opt
                    all_Y[t] = y
                    if compute_lik:
                        lik = lik + torch.where(msk[uid[t]], lI[t], self.zero).sum(dim=0)
            else:
                # set cover for the observed nodes
                prev_s = data.obs[i - 1]
                for t in range(prev_s + 1, s):
                    msk = (y == SIR_STATES.R)
                    y = torch.where(msk, xR[t], y)

                    msk = (y == SIR_STATES.I)
                    rem = torch.where(msk, self.rem, self.n_inf)

                    reachable = data.reachable[t - prev_s]

                    for idx, u in enumerate(uid[t]):
                        if msk[u].max():
                            vid = torch.nonzero(reachable[u], as_tuple=False).squeeze()
                            if vid.numel() == 0:
                                continue
                            opt = (rem[u] > 1) & (rem[vid].min(dim=0).values > 1)
                            msk_opt = msk[u] & opt
                            y[u] = torch.where(msk_opt, xI[t, idx], y[u])
                            trs = (y[u] != SIR_STATES.I)
                            rem[u] = torch.where(msk[u], torch.where(trs, rem[u] - 1, self.n_inf), rem[u])
                            rem[vid] = torch.where(msk[u].unsqueeze(dim=0), torch.where(trs.unsqueeze(dim=0), rem[vid] - 1, self.n_inf), rem[vid])
                            msk[u] = msk_opt
                    all_Y[t] = y
                    if compute_lik:
                        lik = lik + torch.where(msk[uid[t]], lI[t], self.zero).sum(dim=0)

        if compute_lik:
            return all_Y, lik
        else:
            return all_Y

def q_loss(q_net, data, I0, bpar, n_samples):
    """
    Eq 25
    Return the negative mean log-likelihood for proposal distribution.
    """
    print(f"I0: {I0}")
    print(f"n_samples: {n_samples}")
    print(f"data.y shape: {data.y.shape}")
    print(f"data.edge_index shape: {data.edge_index.shape}")
    T = data.T.item()
    n_nodes = data.num_nodes
    Y = diffus_gen(T = T, n_nodes = n_nodes, edge_index = data.edge_index, I0 = I0, n_samples = n_samples, pI = bpar.pI, pR = bpar.pR) # (T+1, nodes, samples)
    print(f"Y shape: {Y.shape}")
    
    q_liks, zI0, zR0, zI, zR = q_net.lik(Y = Y, obs = data.obs) # (samples,)

    print(f"q_liks shape: {q_liks.shape}")
    print(f"zI0 shape: {zI0.shape}")
    print(f"zR0 shape: {zR0.shape}")
    print(f"zI shape: {zI.shape}")
    print(f"zR shape: {zR.shape}")

    return -q_liks.mean(), zI0, zR0, zI, zR

def q_train(data, bpar, args):
    I0 = (data.y[:, 0] == 1).long().sum().item()
    q_net = QNet.make(data, args)
    q_net.train()
    opt = optim.AdamW(q_net.parameters(), lr = args.q_lr)
    pbar = trange(1, args.q_steps + 1)
    for step in pbar:
        opt.zero_grad()
        loss, zI0, zR0, zI, zR = q_loss(q_net, data, I0, bpar, args.q_samples)
        pbar.set_description(f'[step={step}] loss={loss.item():.4f}')
        q_net.backward(loss, zI0, zR0, zI, zR)
        opt.step()
        # Log the loss to wandb
        #wandb.log({"loss": loss.item(), "step": step})
    q_net.eval()
    return q_net

@torch.no_grad()
def t_mcmc(data, bpar, q_net, args, keepdim = True):
    I0 = (data.y[:, 0] == 1).long().sum().item()
    zI, zR = q_net(data.y[:, data.obs]) # (T, nodes, 1)
    #X, lqX = q_net.samp(data.y[:, -1], zI, zR, args.t_samples, data=data, compute_lik = True) # (T, nodes, samples)
    X, lqX = q_net.samp(data.y[:, -1], 0, 0, args.t_samples, data=data, compute_lik = True) 
    pdb.set_trace()
    lpX = diffus_liks(Y = X, edge_index = data.edge_index, I0 = I0, coef = args.p_coef, pI = bpar.pI, pR = bpar.pR) # (samples,)
    tI_avg = data_make_t(X, SIR_STATES.I, dim = 0).float().mean(dim = 1, keepdim = keepdim) # (nodes, 1)
    tR_avg = data_make_t(X, SIR_STATES.R, dim = 0).float().mean(dim = 1, keepdim = keepdim) # (nodes, 1)
    pbar = trange(1, args.t_steps + 1)
    for step in pbar:
        Y, lqY = q_net.samp(data.y[:, -1], zI, zR, args.t_samples, data=data, compute_lik = True) # (T, nodes, samples)
        lpY = diffus_liks(Y = Y, edge_index = data.edge_index, I0 = I0, coef = args.p_coef, pI = bpar.pI, pR = bpar.pR) # (samples,)
        a = torch.rand(args.t_samples, device = args.device) <= torch.exp(lpY + lqX - lpX - lqY) # (samples,) # Hastings MCMC
        X = torch.where(a, Y, X) # (T, nodes, samples)
        lqX = torch.where(a, lqY, lqX) # (samples,)
        lpX = torch.where(a, lpY, lpX) # (samples,)
        tI = data_make_t(X, SIR_STATES.I, dim = 0).float().mean(dim = 1, keepdim = keepdim) # (nodes, 1)
        tR = data_make_t(X, SIR_STATES.R, dim = 0).float().mean(dim = 1, keepdim = keepdim) # (nodes, 1)
        tI_avg = args.t_keep * tI_avg + (1. - args.t_keep) * tI # (nodes, 1)
        tR_avg = args.t_keep * tR_avg + (1. - args.t_keep) * tR # (nodes, 1)
    return tI_avg, tR_avg # (nodes, 1)


def simple_samp(y, zI, zR, data):
    zI, uid = zI.sort(dim=1, descending=True)
    # Remove this line: uid = uid.squeeze(dim=2)
    qI = torch.sigmoid(zI)
    xI = SIR_STATES.I - qI.bernoulli().long()
    qR = torch.sigmoid(zR)
    xR = SIR_STATES.R - qR.bernoulli().long()

    rem = (pyg.utils.degree(data.edge_index[1], num_nodes=data.num_nodes).long() + 1).detach().clone()
    n_inf = data.num_nodes + 2

    neighbs = [[] for _ in range(data.num_nodes)]
    for i in range(data.edge_index.size(1)):
        neighbs[data.edge_index[0, i].item()].append(data.edge_index[1, i].item())
    for u in range(data.num_nodes):
        neighbs[u] = torch.tensor(neighbs[u], dtype=torch.long, device=y.device)

    all_Y = torch.empty(data.T + 1, data.num_nodes, dtype=y.dtype, device=y.device)
    all_Y[data.obs] = data.y[:, data.obs].T

    y = y.clone()  # Make sure y is a copy to avoid modifying the original data

    for i, s in enumerate(data.obs):
        if i == 0:
            for t in range(s - 1, -1, -1):
                msk = (y == SIR_STATES.R)
                y = torch.where(msk, xR[t], y)
                msk = (y == SIR_STATES.I)
                rem = torch.where(msk, rem, n_inf)
                for idx in range(uid.size(1)):
                    u = uid[t, idx].item()
                    if msk[u]:
                        vid = neighbs[u]
                        opt = (rem[u] > 1) & (rem[vid].min() > 1)
                        if opt:
                            y[u] = xI[t, idx]
                            trs = (y[u] != SIR_STATES.I)
                            rem[u] = rem[u] - 1 if trs else n_inf
                            rem[vid] = torch.where(trs, rem[vid] - 1, n_inf)
                all_Y[t] = y
        else:
            y = all_Y[data.obs[i - 1]].clone()
            for t in range(data.obs[i - 1] + 1, s):
                reachable_nodes = data.reachable[t - data.obs[i - 1]].any(dim=0)
                msk = (y == SIR_STATES.R) & reachable_nodes
                y = torch.where(msk, xR[t], y)
                msk = (y == SIR_STATES.I) & reachable_nodes
                rem = torch.where(msk, rem, n_inf)
                for idx in range(uid.size(1)):
                    u = uid[t, idx].item()
                    if msk[u]:
                        vid = neighbs[u]
                        opt = (rem[u] > 1) & (rem[vid].min() > 1)
                        if opt:
                            y[u] = xI[t, idx]
                            trs = (y[u] != SIR_STATES.I)
                            rem[u] = rem[u] - 1 if trs else n_inf
                            rem[vid] = torch.where(trs, rem[vid] - 1, n_inf)
                all_Y[t] = y

    return all_Y


def generate_random_z(T, n_nodes, device):
    zI = torch.randn((T, n_nodes), device=device)
    zR = torch.randn((T, n_nodes), device=device)
    return zI, zR

def generate_and_check_sample(data, bpar, args):
    with torch.no_grad():
        T = data.T.item()
        n_nodes = data.num_nodes
        
        # Generate random zI and zR
        zI, zR = generate_random_z(T, n_nodes, args.device)
        
        # Generate sample history
        sample_history = simple_samp(data.y[:, -1], zI, zR, data)
        
        # Check consistency with observed snapshots
        consistent = True
        for obs in data.obs:
            if not torch.equal(sample_history[obs], data.y[:, obs]):
                consistent = False
                break
        
        if not consistent:
            print("Generated history is NOT consistent with observed snapshots.")
            return None
        
        print("Generated history is consistent with observed snapshots.")
        
        # Compute likelihood under SIR model
        I0 = (data.y[:, 0] == 1).long().sum().item()
        lpX = diffus_liks(Y=sample_history.unsqueeze(-1), 
                          edge_index=data.edge_index, 
                          I0=I0, 
                          coef=args.p_coef, 
                          pI=bpar.pI, 
                          pR=bpar.pR)
        
        print("Likelihood under SIR model:", lpX.item())
        
        # You can set a threshold for what you consider "valid"
        if lpX.item() > -1000:  # This threshold is arbitrary and should be adjusted based on your specific needs
            print("Generated history is valid under the SIR model.")
        else:
            print("Generated history may not be valid under the SIR model.")
        
        # Convert sample_history to y_pred format
        y_pred = sample_history.transpose(0, 1)  # (nodes, T+1)
        return y_pred

def plot_diffusion(X, sample_index=0):
    T, n_nodes, n_samples = X.shape
    sample_history = X[:, :, sample_index].cpu().numpy()  # Select a sample and move it to CPU

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(sample_history, aspect='auto', cmap='viridis')

    ax.set_xlabel('Nodes')
    ax.set_ylabel('Time Steps')
    ax.set_title('Diffusion History for Sample {}'.format(sample_index))
    fig.colorbar(im, ax=ax)
    plt.show()



def main(data):

    # estimate diffusion parameters
    bpar = b_estim(data, args)
    print(f'[est] pI={bpar.pI:.4f}, pR={bpar.pR:.4f}', flush = True)

    # bpar = BPar(pI = 1.0, pR = 0, device = data.y.device)
    # Generate and check a sample history
    # y_pred = generate_and_check_sample(data, bpar, args)

    # You can further analyze or visualize the sample_history here
    # if y_pred is not None:
    #     return y_pred
    # train a proposal network

    q_net = QNet.make(data, args)
    #if not args.debug:
    #pdb.set_trace()
    q_net = q_train(data, bpar, args)
    #else:
    #    q_net.eval()

    # estimate transition times
    tI, tR = t_mcmc(data, bpar, q_net, args, keepdim = True) # (nodes, 1)
    T = data.T.item()
    tI = tI.round().long()
    tR = tR.round().long()
    # compose a history
    with torch.no_grad():
        y_pred = torch.zeros_like(data.y) # (nodes, T+1)
        y_pred.scatter_(dim = 1, index = torch.minimum(tI, data.T), src = torch.full_like(tI, 1))
        y_pred.scatter_(dim = 1, index = torch.minimum(tR, data.T), src = torch.full_like(tR, 2))
        y_pred = y_pred[:, : data.T.item()].cummax(dim = 1).values
        return y_pred

args = get_args()

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="ditto",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate":args.q_lr,
#     "dataset": args.dataset,
#     }
# )


tester = Tester(args.data_dir, args.device, main)
tester.test([args.dataset], seed = args.seed, rep = 1)
tester.save(args.output)

# Finish the wandb run
# wandb.finish()
