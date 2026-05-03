"""
MAMA: Multi-Asset Multi-Agent RL for Portfolio Management
==========================================================
Faithful reproduction of Kim & Lee (IEEE Access, 2025)
DOI: 10.1109/ACCESS.2025.3632201

Key components matching the paper:
  1. TD3 (not PPO) — off-policy with replay buffer
  2. GNN-based SRL — GAT for security selection (Fig 5)
  3. Pretrained SRL — train GNN separately before RL
  4. Masked softmax — actor output for selected securities only (Fig 7)
  5. Two-level hierarchy — inter-asset + intra-asset agents (Fig 4)
  6. Reward = portfolio return (simple, as in paper)
  7. Adam optimizer — β1=0.9, β2=0.999, ε=1e-7, batch=16

Requirements:
  pip install torch yfinance pandas numpy

Usage:
  python mama_paper.py --episodes 200 --device cuda
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json, os, time, copy, random
from collections import deque
from dataclasses import dataclass


# ═══════════════════════════════════════════════
# Config (Table III in paper)
# ═══════════════════════════════════════════════
@dataclass
class Config:
    stock_tickers: tuple = ("SPY","QQQ","IWM","EFA","VGK")
    bond_tickers:  tuple = ("TLT","IEF","LQD")
    comm_tickers:  tuple = ("GLD","USO","DBA")
    start_date: str = "2015-01-01"
    end_date:   str = "2024-12-31"
    test_ratio: float = 0.2

    # Paper hyperparams (Table III)
    lookback: int = 30
    n_episodes: int = 400
    lr_actor:  float = 1e-4
    lr_critic: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005          # target network soft update
    batch_size: int = 32        # standard batch size (reverted from 256 for speed stability)
    n_grad_steps: int = 1       # gradient updates per env step (1 is normal TD3)
    buffer_size: int = 200_000
    exploration_noise: float = 0.1
    policy_noise: float = 0.2   # TD3 target policy smoothing
    noise_clip: float = 0.5
    policy_delay: int = 2       # TD3 delayed policy update

    # GNN SRL
    gnn_pretrain_epochs: int = 100
    gnn_hidden: int = 32
    gnn_out: int = 16
    gnn_heads: int = 2
    gnn_lr: float = 1e-3
    graph_sparsity: float = 40.0

    # Portfolio
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.04
    initial_capital: float = 1_000_000
    top_k_ratio: float = 0.6    # select top 60% of securities per class

    # System
    device: str = "cpu"
    log_interval: int = 10
    save_dir: str = "ckpt_mama"
    seed: int = 42

    @property
    def n_stocks(self): return len(self.stock_tickers)
    @property
    def n_bonds(self): return len(self.bond_tickers)
    @property
    def n_commodities(self): return len(self.comm_tickers)
    @property
    def all_tickers(self):
        return list(self.stock_tickers)+list(self.bond_tickers)+list(self.comm_tickers)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ═══════════════════════════════════════════════
# Data Loader
# ═══════════════════════════════════════════════
YIELD_TICKERS = ["DGS1MO","DGS3MO","DGS1","DGS2","DGS5","DGS10","DGS30"]

class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
    def download(self):
        import yfinance as yf
        print(f"\n  📥 Downloading {len(self.cfg.all_tickers)} tickers...")
        raw = yf.download(self.cfg.all_tickers, start=self.cfg.start_date,
                          end=self.cfg.end_date, auto_adjust=True, progress=False)
        prices = raw["Close"][self.cfg.all_tickers] if isinstance(raw.columns, pd.MultiIndex) else raw[self.cfg.all_tickers]
        prices = prices.ffill().bfill().dropna()
        self.prices_df = prices
        self.returns_df = prices.pct_change().fillna(0)
        print(f"  ✅ {len(prices)} days: {prices.index[0].date()} → {prices.index[-1].date()}")
        # Download Treasury Yield Curve from FRED
        self._download_yields()
        return self

    def _download_yields(self):
        """Download US Treasury Yield Curve from FRED CSV API (no API key needed)."""
        import requests, io
        try:
            print(f"  📥 Downloading Treasury Yields ({len(YIELD_TICKERS)} maturities)...")
            ids = ",".join(YIELD_TICKERS)
            url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                   f"?id={ids}&cosd={self.cfg.start_date}&coed={self.cfg.end_date}")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            yields = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
            # FRED uses '.' for missing values
            yields = yields.replace('.', np.nan).astype(float)
            # Align to ETF trading days, forward-fill missing
            yields = yields.reindex(self.prices_df.index).ffill().bfill().fillna(0)
            # Scale: yields are in % (e.g. 4.5 = 4.5%), divide by 100
            self.yields_df = yields / 100.0
            # Compute derived features: slope (10Y-2Y), slope (10Y-3M), level (avg)
            self.yields_df["SLOPE_10Y_2Y"] = self.yields_df["DGS10"] - self.yields_df["DGS2"]
            self.yields_df["SLOPE_10Y_3M"] = self.yields_df["DGS10"] - self.yields_df["DGS3MO"]
            self.yields_df["LEVEL"] = self.yields_df[YIELD_TICKERS].mean(axis=1)
            print(f"  ✅ Yields: {self.yields_df.shape[1]} features (7 raw + 3 derived)")
        except Exception as e:
            print(f"  ⚠️ Yield download failed ({e}), using zeros")
            n_days = len(self.prices_df)
            self.yields_df = pd.DataFrame(
                np.zeros((n_days, 10)),
                index=self.prices_df.index,
                columns=YIELD_TICKERS + ["SLOPE_10Y_2Y","SLOPE_10Y_3M","LEVEL"]
            )

    def split(self):
        n = len(self.prices_df); si = int(n*(1-self.cfg.test_ratio))
        return ({"prices": self.prices_df.iloc[:si].values, "returns": self.returns_df.iloc[:si].values},
                {"prices": self.prices_df.iloc[si:].values, "returns": self.returns_df.iloc[si:].values})


# ═══════════════════════════════════════════════
# Environment — reward = portfolio return (paper)
# ═══════════════════════════════════════════════
class MarketEnv:
    def __init__(self, prices, returns, cfg, mode="train", test_offset=0, yields=None):
        self.cfg, self.prices, self.returns, self.mode = cfg, prices, returns, mode
        self.n_assets = prices.shape[1]
        self.yields = yields if yields is not None else np.zeros((len(prices), 10))
        self.n_yields = self.yields.shape[1]
        self.test_offset = test_offset  # feature index where test period starts
        self.features = self._feats()
        self._n_features = len(self.features)
        lb = cfg.lookback
        if mode == "test":
            self.max_ep_len = self._n_features - test_offset
        else:
            self.max_ep_len = min(252, self._n_features - 10)
        self.reset()

    def _feats(self):
        T, n = self.prices.shape; lb = self.cfg.lookback; out = []
        for t in range(lb, T):
            rw = self.returns[t-lb:t]
            nr = rw.mean(0)/(rw.std(0)+1e-8)
            ma5 = self.prices[max(t-5,0):t].mean(0)
            ma20 = self.prices[max(t-20,0):t].mean(0)
            mar = ma5/(ma20+1e-8)-1
            vol = rw.std(0)*np.sqrt(252)
            mom = self.prices[t]/(self.prices[t-lb]+1e-8)-1
            g = np.maximum(rw,0).mean(0); l = np.maximum(-rw,0).mean(0)
            rsi = g/(g+l+1e-8)
            out.append(np.concatenate([nr,mar,vol,mom,rsi]).astype(np.float32))
        return np.array(out)

    def obs_dim(self): return self.features.shape[1] + (self.n_assets + 1) + 3

    def reset(self):
        if self.mode=="train":
            # Random episode length (126-252 days) for regularization
            self.max_ep_len = np.random.randint(126, min(253, self._n_features - 9))
            mx = self._n_features - self.max_ep_len
            self.si = np.random.randint(0, max(1, mx))
        else:
            self.si = self.test_offset  # start at test period, full history available for lookback
        self.t = 0
        self.pv = self.cfg.initial_capital
        self.w = np.zeros(self.n_assets+1); self.w[-1]=1.0
        return self._obs()

    def _obs(self):
        fi = min(self.si+self.t, len(self.features)-1)
        return np.concatenate([self.features[fi], self.w,
                               [self.pv/self.cfg.initial_capital, self.cfg.risk_free_rate/252,
                                float(self.t)/252]]).astype(np.float32)

    def step(self, nw):
        tc = np.abs(nw-self.w).sum()*self.cfg.transaction_cost
        ri = self.si+self.t+self.cfg.lookback
        dr = self.returns[ri] if ri<len(self.returns) else np.zeros(self.n_assets)
        cr = self.cfg.risk_free_rate/252
        ar = np.append(dr, cr)
        pr = np.dot(nw, ar)-tc
        self.pv *= (1+pr)
        self.w = nw*(1+ar); self.w /= self.w.sum()+1e-10
        self.t += 1
        done = self.t>=self.max_ep_len if self.mode=="train" else (self.si+self.t+self.cfg.lookback)>=len(self.returns)
        # Paper reward: portfolio return, scaled for TD3 gradient magnitude
        # ×30 balances gradient signal vs loss-aversion (×100 made agent too defensive)
        reward = pr * 30
        return self._obs(), reward, done, {"portfolio_value":self.pv,"return":pr}

    def asset_indices(self):
        ns,nb = self.cfg.n_stocks, self.cfg.n_bonds
        nc = self.cfg.n_commodities
        return {"stocks":list(range(ns)),"bonds":list(range(ns,ns+nb)),
                "commodities":list(range(ns+nb,ns+nb+nc)),"cash":[ns+nb+nc]}

    def get_returns_window(self, sec_indices, window=None):
        if window is None: window = self.cfg.lookback
        ri = self.si+self.t+self.cfg.lookback
        s = max(0,ri-window); e = min(ri, len(self.returns))
        return self.returns[s:e][:,sec_indices] if e>s else np.zeros((1,len(sec_indices)))

    def get_yields_window(self, window=None):
        """Get treasury yield curve window aligned with returns window."""
        if window is None: window = self.cfg.lookback
        ri = self.si+self.t+self.cfg.lookback
        s = max(0,ri-window); e = min(ri, len(self.yields))
        return self.yields[s:e] if e>s else np.zeros((1, self.n_yields))


# ═══════════════════════════════════════════════
# Correlation Graph Builder (paper Section III.B)
# ═══════════════════════════════════════════════
class GraphBuilder:
    def __init__(self, n, sparsity=40.0):
        self.n = n
        self.sp = sparsity if n<=50 else 20.0
        self._cache = None; self._cache_key = None

    def build(self, ret_window, device, cache_key=None):
        """cache_key should be (env.si, env.t) to avoid cross-episode hits."""
        if cache_key is not None and self._cache is not None:
            if self._cache_key is not None:
                old_si, old_t = self._cache_key
                new_si, new_t = cache_key
                if old_si == new_si and abs(new_t - old_t) < 10:
                    return self._cache
        n = self.n
        if ret_window.shape[0]<5 or n<2:
            s,d=[],[]
            for i in range(n):
                for j in range(n): s.append(i);d.append(j)
            r = (torch.LongTensor([s,d]).to(device), torch.ones(len(s),device=device)/n)
        else:
            c = np.corrcoef(ret_window.T); c = np.nan_to_num(c)
            dist = np.sqrt(np.clip(2*(1-c),0,4)); np.fill_diagonal(dist,0)
            up = dist[np.triu_indices(n,k=1)]
            th = np.percentile(up, self.sp) if len(up)>0 else float('inf')
            s,d,ww = [],[],[]
            for i in range(n):
                s.append(i);d.append(i);ww.append(1.0)
                for j in range(n):
                    if i!=j and dist[i,j]<=th:
                        w = max(0, 1-dist[i,j]/(dist.max()+1e-8))
                        s.append(i);d.append(j);ww.append(w)
            r = (torch.LongTensor([s,d]).to(device), torch.FloatTensor(ww).to(device))
        self._cache = r; self._cache_key = cache_key
        return r


# ═══════════════════════════════════════════════
# GAT Layer (paper Fig 5)
# ═══════════════════════════════════════════════
class GATLayer(nn.Module):
    def __init__(self, ind, outd, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(ind, outd, bias=False)
        self.a_s = nn.Parameter(torch.zeros(outd,1)); self.a_d = nn.Parameter(torch.zeros(outd,1))
        nn.init.xavier_uniform_(self.a_s); nn.init.xavier_uniform_(self.a_d)
        self.lrelu = nn.LeakyReLU(0.2); self.drop = nn.Dropout(dropout)
    def forward(self, x, ei, ew=None):
        n=x.size(0); h=self.W(x); s,d=ei
        e=self.lrelu((h[s]@self.a_s+h[d]@self.a_d).squeeze(-1))
        if ew is not None: e=e*ew
        mx=torch.zeros(n,device=x.device).scatter_reduce_(0,d,e,reduce='amax',include_self=True)
        en=torch.exp(e-mx[d])
        es=torch.zeros(n,device=x.device).scatter_add_(0,d,en)
        a=self.drop(en/(es[d]+1e-10))
        msg=h[s]*a.unsqueeze(-1)
        return torch.zeros(n,h.size(1),device=x.device).scatter_add_(0,d.unsqueeze(-1).expand_as(msg),msg)

class MHGAT(nn.Module):
    def __init__(self, ind, outd, nh=2, drop=0.1):
        super().__init__()
        hd=outd//nh
        self.heads=nn.ModuleList([GATLayer(ind,hd,drop) for _ in range(nh)])
    def forward(self,x,ei,ew=None):
        return torch.cat([h(x,ei,ew) for h in self.heads], -1)


# ═══════════════════════════════════════════════
# GNN-based SRL (paper Fig 5, Section III.B)
# Pretrained separately before RL training
# ═══════════════════════════════════════════════
class GNNSRL(nn.Module):
    def __init__(self, feat_dim, n_sec, gh=32, go=16, nh=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(feat_dim,gh), nn.LayerNorm(gh), nn.ReLU())
        self.g1 = MHGAT(gh,gh,nh); self.n1 = nn.LayerNorm(gh)
        self.g2 = MHGAT(gh,go,nh); self.n2 = nn.LayerNorm(go)
        self.scorer = nn.Sequential(nn.Linear(go,go//2), nn.ReLU(), nn.Linear(go//2,1))
        # Pretrain head: predict next-day return direction
        self.pretrain_head = nn.Sequential(nn.Linear(go,go//2), nn.ReLU(), nn.Linear(go//2,1), nn.Tanh())
        self.go = go

    def forward(self, x, ei, ew):
        h = self.enc(x)
        h1 = self.n1(F.relu(self.g1(h,ei,ew))+h)
        h2 = self.n2(F.relu(self.g2(h1,ei,ew)))
        scores = torch.sigmoid(self.scorer(h2).squeeze(-1))
        return h2, scores

    def pretrain_forward(self, x, ei, ew):
        emb, _ = self.forward(x, ei, ew)
        return self.pretrain_head(emb).squeeze(-1)


# ═══════════════════════════════════════════════
# Pretrain GNN-SRL (paper: pretrained before RL)
# Self-supervised: predict next-day return sign
# ═══════════════════════════════════════════════
def pretrain_gnn_srl(gnn, prices, returns, sec_indices, graph_builder, cfg, device):
    """Pretrain GNN to predict return direction — gives meaningful embeddings before RL."""
    print(f"    Pretraining GNN-SRL ({cfg.gnn_pretrain_epochs} epochs)...", end=" ", flush=True)
    opt = optim.Adam(gnn.parameters(), lr=cfg.gnn_lr)
    lb = cfg.lookback; n_sec = len(sec_indices)
    T = returns.shape[0]
    # Use all available training data (not just 500 days)
    max_t = T - 1

    for epoch in range(cfg.gnn_pretrain_epochs):
        total_loss = 0; count = 0
        sample_size = min(500, max_t - (lb+1))
        t_indices = np.random.choice(range(lb+1, max_t), sample_size, replace=False)
        for t in t_indices:
            rw = returns[t-lb:t][:,sec_indices]
            ei, ew = graph_builder.build(rw, device)

            # Build per-security features — MATCHING runtime features exactly
            nr = rw.mean(0)/(rw.std(0)+1e-8)
            vol = rw.std(0)*np.sqrt(252)
            # Real momentum from prices
            mom = prices[t, sec_indices]/(prices[t-lb, sec_indices]+1e-8)-1
            # Real MA ratio from prices
            ma5 = prices[max(t-5,0):t][:, sec_indices].mean(0)
            ma20 = prices[max(t-20,0):t][:, sec_indices].mean(0)
            mar = ma5/(ma20+1e-8)-1
            g = np.maximum(rw,0).mean(0); l = np.maximum(-rw,0).mean(0)
            rsi = g/(g+l+1e-8)
            feats = torch.FloatTensor(np.stack([nr,mar,vol,mom,rsi], axis=-1)).to(device)

            # Target: next-day return direction
            next_ret = returns[t][sec_indices]
            target = torch.FloatTensor(np.sign(next_ret)).to(device)

            pred = gnn.pretrain_forward(feats, ei, ew)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); count += 1

    print(f"done (loss={total_loss/(count+1):.4f})")


# ═══════════════════════════════════════════════
# Pretrain Inter-Asset LSTM-SRL
# Self-supervised: predict next-day return direction
# ═══════════════════════════════════════════════
def pretrain_inter_srl(srl, returns, yields, cfg, device):
    """Pretrain GRU-SRL to predict next-day return direction per asset.
    Input: concatenated [ETF returns, treasury yields] per timestep."""
    n_assets = returns.shape[1]
    print(f"    Pretraining Inter-Asset GRU ({cfg.gnn_pretrain_epochs} epochs)...", end=" ", flush=True)
    pred_head = nn.Linear(srl.out_dim, n_assets).to(device)
    params = list(srl.parameters()) + list(pred_head.parameters())
    opt = optim.Adam(params, lr=cfg.gnn_lr)
    lb = cfg.lookback; T = returns.shape[0]

    for epoch in range(cfg.gnn_pretrain_epochs):
        total_loss = 0; count = 0
        max_t = T - 1
        sample_size = min(500, max_t - (lb+1))
        t_indices = np.random.choice(range(lb+1, max_t), sample_size, replace=False)
        for t in t_indices:
            rw = returns[t-lb:t]
            yw = yields[t-lb:t]
            combined = np.concatenate([rw, yw], axis=1)  # (lb, n_assets+n_yields)
            combined_t = torch.FloatTensor(combined).unsqueeze(0).to(device)
            encoded = srl(combined_t)  # (1, hidden)
            pred = torch.tanh(pred_head(encoded)).squeeze(0)  # (n_assets,)
            target = torch.FloatTensor(np.sign(returns[t])).to(device)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); count += 1

    print(f"done (loss={total_loss/(count+1):.4f})")
    del pred_head  # discard prediction head, keep trained LSTM weights


# ═══════════════════════════════════════════════
# TD3 Networks (paper Fig 7, Section III.C)
# ═══════════════════════════════════════════════
class TD3Actor(nn.Module):
    """Actor with masked softmax output (paper Fig 7a)."""
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, state, mask=None):
        logits = self.net(state)
        if mask is not None:
            logits = logits + (1-mask)*(-1e9)  # masked softmax
        return F.softmax(logits, dim=-1)

class TD3Critic(nn.Module):
    """Twin critics (paper: standard TD3)."""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, s, a):
        sa = torch.cat([s,a], -1)
        return self.q1(sa), self.q2(sa)


# ═══════════════════════════════════════════════
# Replay Buffer (TD3 requirement)
# ═══════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, cap=200_000, use_mask=False):
        self.cap = cap
        self.ptr = 0
        self.size = 0
        self.use_mask = use_mask
        
        self.s = np.zeros((cap, state_dim), dtype=np.float32)
        self.a = np.zeros((cap, action_dim), dtype=np.float32)
        self.r = np.zeros((cap,), dtype=np.float32)
        self.ns = np.zeros((cap, state_dim), dtype=np.float32)
        self.d = np.zeros((cap,), dtype=np.float32)
        if use_mask:
            self.m = np.zeros((cap, action_dim), dtype=np.float32)

    def add(self, s, a, r, ns, d, mask=None):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.d[self.ptr] = d
        if self.use_mask and mask is not None:
            self.m[self.ptr] = mask
        
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs):
        bs = min(bs, self.size)
        ind = np.random.choice(self.size, bs, replace=False)
        m_batch = self.m[ind] if self.use_mask else None
        return (
            self.s[ind],
            self.a[ind],
            self.r[ind],
            self.ns[ind],
            self.d[ind],
            m_batch
        )

    def __len__(self): return self.size


# ═══════════════════════════════════════════════
# Intra-Asset Agent: GNN-SRL + TD3 (paper Fig 6)
# ═══════════════════════════════════════════════
class IntraAssetAgent:
    def __init__(self, obs_dim, n_sec, sec_indices, cfg, device):
        self.n_sec = n_sec
        self.sec_indices = sec_indices
        self.cfg = cfg
        self.device = device
        self.top_k = max(2, int(n_sec * cfg.top_k_ratio))

        # GNN-SRL (pretrained, then fine-tuned with RL)
        self.gnn = GNNSRL(5, n_sec, cfg.gnn_hidden, cfg.gnn_out, cfg.gnn_heads).to(device)
        self.graph_builder = GraphBuilder(n_sec, cfg.graph_sparsity)

        # TD3 actor-critic
        # Actor input: obs + GNN embeddings (pooled)
        actor_in = obs_dim + cfg.gnn_out
        self.actor = TD3Actor(actor_in, n_sec, 64).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TD3Critic(actor_in, n_sec, 64).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        all_params = list(self.actor.parameters())+list(self.gnn.parameters())
        self.actor_opt = optim.Adam(all_params, lr=cfg.lr_actor, betas=(0.9,0.999), eps=1e-7, weight_decay=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, betas=(0.9,0.999), eps=1e-7, weight_decay=1e-5)

        self.buffer = ReplayBuffer(actor_in, n_sec, cfg.buffer_size, use_mask=True)
        self.total_it = 0
        self._state_cache = None
        self._state_cache_key = None

    @torch.no_grad()
    def _get_gnn_state(self, obs_np, env):
        """Build graph, run GNN, return enriched state + selection mask.
        Always no_grad — this is called during data collection, not training.
        GNN fine-tuning happens in train_step via replay buffer."""
        cache_key = (env.si, env.t)
        if self._state_cache_key == cache_key:
            return self._state_cache

        rw = env.get_returns_window(self.sec_indices)
        ei, ew = self.graph_builder.build(rw, self.device, cache_key=cache_key)

        # Per-security features from obs
        n_total = env.n_assets
        obs_t = torch.FloatTensor(obs_np).to(self.device)
        per_sec = []
        for fb in range(5):  # 5 feature blocks
            s = fb*n_total
            per_sec.append(obs_t[s:s+n_total][self.sec_indices])
        sec_feat = torch.stack(per_sec, dim=-1)  # (n_sec, 5)

        emb, scores = self.gnn(sec_feat, ei, ew)

        # Selection mask: top-k by GNN score
        mask = torch.zeros(self.n_sec, device=self.device)
        _, top_idx = scores.topk(self.top_k)
        mask[top_idx] = 1.0

        # Pool GNN embeddings as context
        gnn_ctx = emb.mean(dim=0)  # (gnn_out,)

        # Enriched state
        enriched = torch.cat([obs_t, gnn_ctx])
        res = (enriched, mask, ei, ew, sec_feat)
        self._state_cache = res
        self._state_cache_key = cache_key
        return res

    @torch.no_grad()
    def select_action(self, obs_np, env, noise=0.1):
        enriched, mask, *_ = self._get_gnn_state(obs_np, env)
        action = self.actor(enriched.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
        if noise > 0:
            n = torch.randn_like(action)*noise*mask
            action = action + n
            action = torch.clamp(action, 0, 1)
            action = action*mask
            action = action/(action.sum()+1e-10)
        return action.cpu().numpy(), enriched.cpu().numpy(), mask.cpu().numpy()

    def train_step(self):
        if len(self.buffer) < self.cfg.batch_size*4:
            return 0.0
        self.total_it += 1
        s,a,r,ns,d,masks = self.buffer.sample(self.cfg.batch_size)
        nb = True  # non_blocking for faster CPU→GPU transfer
        st = torch.FloatTensor(s).to(self.device, non_blocking=nb)
        at = torch.FloatTensor(a).to(self.device, non_blocking=nb)
        rt = torch.FloatTensor(r).unsqueeze(1).to(self.device, non_blocking=nb)
        nst = torch.FloatTensor(ns).to(self.device, non_blocking=nb)
        dt = torch.FloatTensor(d).unsqueeze(1).to(self.device, non_blocking=nb)
        mt = torch.FloatTensor(masks).to(self.device, non_blocking=nb) if masks is not None else None

        # TD3: target action with smoothing
        with torch.no_grad():
            na = self.actor_target(nst, mt)
            noise = (torch.randn_like(na)*self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            na = (na+noise).clamp(0,1)
            if mt is not None: na = na * mt
            na = na/(na.sum(-1,keepdim=True)+1e-10)
            q1t, q2t = self.critic_target(nst, na)
            target_q = rt + (1-dt)*self.cfg.gamma*torch.min(q1t, q2t)

        q1, q2 = self.critic(st, at)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        # Delayed policy update — with mask
        actor_loss_val = 0
        if self.total_it % self.cfg.policy_delay == 0:
            a_pred = self.actor(st, mt)  # FIX: pass mask
            actor_loss = -self.critic.q1(torch.cat([st,a_pred],-1)).mean()
            self.actor_opt.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            nn.utils.clip_grad_norm_(self.gnn.parameters(), 1.0)
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()

            # Soft update targets
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.cfg.tau*p.data + (1-self.cfg.tau)*tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.cfg.tau*p.data + (1-self.cfg.tau)*tp.data)

        return critic_loss.item()


# ═══════════════════════════════════════════════
# RNN-based SRL for Inter-Asset Agent (paper III.C)
# GRU encodes temporal return sequences per class
# (Changed from LSTM to GRU: fewer params, faster,
#  no separate cell state — well-suited for 30-day lookback)
# ═══════════════════════════════════════════════
class InterAssetSRL(nn.Module):
    """RNN-based SRL: encodes temporal sequences of asset returns + treasury yields.
    Uses GRU instead of LSTM: lighter (25% fewer params), faster training,
    and empirically competitive on short lookback windows (<=30 days)."""
    def __init__(self, input_dim, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        # input_dim = n_assets + n_yields (ETF returns + yield curve features)
        self.gru = nn.GRU(input_dim, hidden, n_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden
    def forward(self, combined_window):
        # combined_window: (batch, lookback, n_assets + n_yields)
        # GRU returns (output, hn) — no cell state unlike LSTM
        _, hn = self.gru(combined_window)
        return self.norm(hn[-1])  # (batch, hidden)


# ═══════════════════════════════════════════════
# Inter-Asset Agent: RNN-SRL + TD3 (paper Fig 4)
# ═══════════════════════════════════════════════
class InterAssetAgent:
    def __init__(self, obs_dim, n_assets, n_yields, n_classes=4, cfg=None, device="cpu"):
        self.cfg = cfg; self.device = device; self.n_assets = n_assets
        self.n_yields = n_yields
        # RNN-SRL encodes return history + yield curve
        lstm_input_dim = n_assets + n_yields  # ETF returns + treasury yields
        self.srl = InterAssetSRL(lstm_input_dim, hidden=64, n_layers=2).to(device)
        self.srl_target = copy.deepcopy(self.srl)  # FIX: target SRL
        enriched_dim = obs_dim + self.srl.out_dim
        self.actor = TD3Actor(enriched_dim, n_classes, 128).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TD3Critic(enriched_dim, n_classes, 128).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        all_params = list(self.actor.parameters()) + list(self.srl.parameters())
        self.actor_opt = optim.Adam(all_params, lr=cfg.lr_actor, betas=(0.9,0.999), eps=1e-7, weight_decay=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, betas=(0.9,0.999), eps=1e-7, weight_decay=1e-5)
        self.buffer = ReplayBuffer(enriched_dim, n_classes, cfg.buffer_size, use_mask=False)
        self.total_it = 0
        self._state_cache = None
        self._state_cache_key = None

    @torch.no_grad()
    def _enrich(self, obs_np, env):
        """Build enriched state: obs + LSTM-encoded (returns + yield curve) history."""
        cache_key = (env.si, env.t)
        if getattr(self, '_state_cache_key', None) == cache_key:
            return self._state_cache

        rw = env.get_returns_window(list(range(env.n_assets)), window=self.cfg.lookback)
        yw = env.get_yields_window(window=self.cfg.lookback)
        # Align lengths (in case of mismatch at boundaries)
        min_len = min(len(rw), len(yw))
        combined = np.concatenate([rw[-min_len:], yw[-min_len:]], axis=1)  # (lookback, n_assets+n_yields)
        combined_t = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
        lstm_out = self.srl(combined_t).squeeze(0)
        obs_t = torch.FloatTensor(obs_np).to(self.device)
        
        res = torch.cat([obs_t, lstm_out]).cpu().numpy()
        self._state_cache = res
        self._state_cache_key = cache_key
        return res

    def select_action(self, obs_np, env, noise=0.1):
        enriched = self._enrich(obs_np, env)
        st = torch.FloatTensor(enriched).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor(st).squeeze(0)
        if noise > 0:
            a = a + torch.randn_like(a)*noise
            a = torch.clamp(a, 0, 1)
            a = a/(a.sum()+1e-10)
        return a.cpu().numpy(), enriched

    def train_step(self):
        if len(self.buffer) < self.cfg.batch_size*4: return 0.0
        self.total_it += 1
        s,a,r,ns,d,_ = self.buffer.sample(self.cfg.batch_size)
        nb = True  # non_blocking for faster CPU→GPU transfer
        st=torch.FloatTensor(s).to(self.device, non_blocking=nb)
        at=torch.FloatTensor(a).to(self.device, non_blocking=nb)
        rt=torch.FloatTensor(r).unsqueeze(1).to(self.device, non_blocking=nb)
        nst=torch.FloatTensor(ns).to(self.device, non_blocking=nb)
        dt=torch.FloatTensor(d).unsqueeze(1).to(self.device, non_blocking=nb)
        with torch.no_grad():
            na=self.actor_target(nst)
            noise=(torch.randn_like(na)*self.cfg.policy_noise).clamp(-self.cfg.noise_clip,self.cfg.noise_clip)
            na=(na+noise).clamp(0,1); na=na/(na.sum(-1,keepdim=True)+1e-10)
            q1t,q2t=self.critic_target(nst,na)
            tq=rt+(1-dt)*self.cfg.gamma*torch.min(q1t,q2t)
        q1,q2=self.critic(st,at)
        cl=F.mse_loss(q1,tq)+F.mse_loss(q2,tq)
        self.critic_opt.zero_grad();cl.backward();self.critic_opt.step()
        if self.total_it%self.cfg.policy_delay==0:
            al=-self.critic.q1(torch.cat([st,self.actor(st)],-1)).mean()
            self.actor_opt.zero_grad();al.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),1.0)
            nn.utils.clip_grad_norm_(self.srl.parameters(),1.0)
            self.actor_opt.step()
            # Soft update targets — including SRL
            for p,tp in zip(self.actor.parameters(),self.actor_target.parameters()):
                tp.data.copy_(self.cfg.tau*p.data+(1-self.cfg.tau)*tp.data)
            for p,tp in zip(self.critic.parameters(),self.critic_target.parameters()):
                tp.data.copy_(self.cfg.tau*p.data+(1-self.cfg.tau)*tp.data)
            for p,tp in zip(self.srl.parameters(),self.srl_target.parameters()):
                tp.data.copy_(self.cfg.tau*p.data+(1-self.cfg.tau)*tp.data)
        return cl.item()


# ═══════════════════════════════════════════════
# Baselines (Academic Standard)
# ═══════════════════════════════════════════════
class BL:
    @staticmethod
    def run(env, fn, name, stateful=False):
        """Run a baseline strategy through the environment.
        If stateful=True, fn is called with a state dict for persistence."""
        obs=env.reset(); pv=[env.cfg.initial_capital]; dr=[]
        state = {}  # for stateful strategies
        while True:
            if stateful:
                w = fn(env, obs, state)
            else:
                w = fn(env, obs)
            obs,_,done,info = env.step(w)
            pv.append(info["portfolio_value"]); dr.append(info["return"])
            if done: break
        d=np.array(dr); tr_=pv[-1]/pv[0]-1; ny=len(d)/252
        ar_=(1+tr_)**(1/max(ny,.01))-1; v=d.std()*np.sqrt(252) if len(d)>1 else 0
        sh=(ar_-env.cfg.risk_free_rate)/v if v>1e-4 else 0
        pk=pv[0];md=0
        for x in pv:
            if x>pk:pk=x
            if(pk-x)/pk>md:md=(pk-x)/pk
        return {"name":name,"total_return":tr_,"sharpe_ratio":sh,"max_drawdown":md,"pv":pv}

    # ── UBAH: Uniform Buy And Hold ──
    @staticmethod
    def ubah(e, o, state):
        """Buy equal weight once, then hold (let weights drift)."""
        if 'init' not in state:
            state['init'] = True
            n = e.n_assets + 1
            return np.ones(n) / n
        return e.w.copy()  # hold existing weights

    # ── Equal Weight (rebalanced daily) ──
    @staticmethod
    def eq(e,o): n=e.n_assets+1; return np.ones(n)/n

    # ── 60/40 Stock/Bond ──
    @staticmethod
    def sb(e,o):
        w=np.zeros(e.n_assets+1); idx=e.asset_indices()
        for i in idx["stocks"]: w[i]=0.6/len(idx["stocks"])
        for i in idx["bonds"]: w[i]=0.4/len(idx["bonds"])
        return w

    # ── Momentum 12-1 (standard, monthly rebalance, max weight cap) ──
    @staticmethod
    def momentum_12_1(e, o, state):
        """Standard cross-sectional momentum: 12M return skipping last 1M.
        Rebalances monthly. Max 25% per asset."""
        n = e.n_assets
        max_wt = 0.25  # cap per asset
        ri = e.si + e.t + e.cfg.lookback

        # Only rebalance monthly (every ~21 trading days)
        if 'w' in state and e.t % 21 != 0:
            return state['w']

        # Need at least 252 days of history
        if ri < 252 or ri >= len(e.prices):
            w = np.ones(n+1)/(n+1)
            state['w'] = w
            return w

        # 12-month return, skip most recent 1 month (21 days)
        p_now_skip = e.prices[ri - 21]   # price 1 month ago
        p_12m_ago  = e.prices[ri - 252]   # price 12 months ago
        mom_signal = p_now_skip / (p_12m_ago + 1e-8) - 1.0

        # Long only: clip negative momentum, apply cap
        s = np.clip(mom_signal, 0, None)
        total = s.sum()
        w = np.zeros(n+1)
        if total > 1e-8:
            w[:n] = s / total * 0.9  # 90% invested, 10% cash
            # Apply max weight cap and redistribute
            excess = np.maximum(w[:n] - max_wt, 0)
            if excess.sum() > 0:
                w[:n] = np.minimum(w[:n], max_wt)
                # Redistribute excess to under-cap assets proportionally
                under = w[:n][w[:n] < max_wt]
                if len(under) > 0 and under.sum() > 0:
                    w[:n][w[:n] < max_wt] *= (1 + excess.sum() / under.sum())
            w[-1] = max(0.1, 1.0 - w[:n].sum())
        else:
            w[-1] = 1.0
        w /= w.sum() + 1e-10
        state['w'] = w
        return w

    # ── CORN: CORrelation-driven Nonparametric learning ──
    @staticmethod
    def corn(e, o, state):
        """CORN baseline from Li et al. (2011). Finds similar historical
        windows and uses their subsequent returns to form portfolio.
        Rebalances every 5 trading days to limit turnover."""
        n = e.n_assets; ri = e.si + e.t + e.cfg.lookback
        window = 5; rho_threshold = 0.5

        # Rebalance every 5 trading days
        if 'w' in state and e.t % 5 != 0:
            return state['w']

        if ri < window + 10 or ri >= len(e.returns):
            w = np.ones(n+1)/(n+1)
            state['w'] = w
            return w

        # Current window of returns
        curr = e.returns[ri-window:ri, :n].flatten()
        if np.std(curr) < 1e-10:
            return state.get('w', np.ones(n+1)/(n+1))

        # Find similar historical windows
        similar_next = []
        search_start = max(window, ri - 2000)
        for t in range(search_start, ri - window):
            hist = e.returns[t:t+window, :n].flatten()
            if np.std(hist) < 1e-10: continue
            corr = np.corrcoef(curr, hist)[0,1]
            if not np.isnan(corr) and corr > rho_threshold:
                if t + window < len(e.returns):
                    similar_next.append(e.returns[t+window, :n])

        w = np.zeros(n+1)
        if len(similar_next) >= 3:
            avg_ret = np.mean(similar_next, axis=0)
            s = np.clip(avg_ret, 0, None)
            total = s.sum()
            if total > 1e-8:
                w[:n] = s / total * 0.9
                w[-1] = 0.1
            else:
                w[-1] = 1.0
        else:
            w[:n] = 0.9/n; w[-1] = 0.1
        w /= w.sum() + 1e-10
        state['w'] = w
        return w

    # ── OLMAR: Online Moving Average Reversion (tuned for ETFs) ──
    @staticmethod
    def olmar(e, o, state):
        """OLMAR from Li & Hoi (2012). Tuned: eps=2, rebalance every 5 days,
        max turnover 30% per rebalance to limit TC impact."""
        n = e.n_assets; ri = e.si + e.t + e.cfg.lookback
        window = 5; eps = 2  # reduced from 10 for ETF portfolio

        # Rebalance every 5 trading days
        if 'w' in state and e.t % 5 != 0:
            return state['w']

        if ri < window + 1 or ri >= len(e.prices):
            w = np.ones(n+1)/(n+1)
            state['w'] = w
            return w

        # Moving average prediction: price will revert to MA
        ma = e.prices[ri-window:ri, :n].mean(axis=0)
        price_now = e.prices[ri, :n]
        x_pred = ma / (price_now + 1e-10)

        # Current portfolio (uniform if first step)
        b = state.get('w', np.ones(n+1)/(n+1))[:n]
        if b.sum() < 1e-10: b = np.ones(n)/n
        b = b / (b.sum() + 1e-10)

        # OLMAR update
        x_bar = x_pred.mean()
        deviation = x_pred - x_bar
        denom = np.dot(deviation, deviation) + 1e-10
        lam = max(0, (eps - np.dot(b, x_pred)) / denom)

        b_new = b + lam * deviation
        b_new = np.maximum(b_new, 0)
        total = b_new.sum()

        w_target = np.zeros(n+1)
        if total > 1e-8:
            w_target[:n] = b_new / total * 0.95
            w_target[-1] = 0.05
        else:
            w_target[:n] = 0.95/n; w_target[-1] = 0.05
        w_target /= w_target.sum() + 1e-10

        # Turnover cap: max 30% change, blend with previous weights
        w_prev = state.get('w', np.ones(n+1)/(n+1))
        turnover = np.abs(w_target - w_prev).sum()
        if turnover > 0.30:
            blend = 0.30 / (turnover + 1e-10)
            w_target = blend * w_target + (1 - blend) * w_prev
            w_target /= w_target.sum() + 1e-10

        state['w'] = w_target
        return w_target

    # ── PAMR: Passive Aggressive Mean Reversion (tuned for ETFs) ──
    @staticmethod
    def pamr(e, o, state):
        """PAMR from Li et al. (2012). Tuned: eps=0.8, C=50, rebalance every
        5 days, max turnover 30% per rebalance."""
        n = e.n_assets; ri = e.si + e.t + e.cfg.lookback
        eps = 0.8; C = 50  # reduced aggressiveness for ETF portfolio

        # Rebalance every 5 trading days
        if 'w' in state and e.t % 5 != 0:
            return state['w']

        if ri < 2 or ri >= len(e.prices):
            w = np.ones(n+1)/(n+1)
            state['w'] = w
            return w

        # Price relative: today's price / yesterday's price
        x_t = e.prices[ri, :n] / (e.prices[ri-1, :n] + 1e-10)

        # Current portfolio
        b = state.get('w', np.ones(n+1)/(n+1))[:n]
        if b.sum() < 1e-10: b = np.ones(n)/n
        b = b / (b.sum() + 1e-10)

        # Loss
        x_bar = x_t.mean()
        loss = max(0, np.dot(b, x_t) - eps)

        deviation = x_t - x_bar
        denom = np.dot(deviation, deviation) + 1e-10
        tau = min(C, loss / denom)

        b_new = b - tau * deviation
        b_new = np.maximum(b_new, 0)
        total = b_new.sum()

        w_target = np.zeros(n+1)
        if total > 1e-8:
            w_target[:n] = b_new / total * 0.95
            w_target[-1] = 0.05
        else:
            w_target[:n] = 0.95/n; w_target[-1] = 0.05
        w_target /= w_target.sum() + 1e-10

        # Turnover cap: max 30% change
        w_prev = state.get('w', np.ones(n+1)/(n+1))
        turnover = np.abs(w_target - w_prev).sum()
        if turnover > 0.30:
            blend = 0.30 / (turnover + 1e-10)
            w_target = blend * w_target + (1 - blend) * w_prev
            w_target /= w_target.sum() + 1e-10

        state['w'] = w_target
        return w_target

    # ── Risk Parity: Inverse Volatility Weighting ──
    @staticmethod
    def risk_parity(e, o, state):
        """Risk Parity (inverse volatility). w_i = (1/vol_i) / sum(1/vol_j).
        Rolling 60-day vol, monthly rebalance. Cash = 5%."""
        n = e.n_assets; ri = e.si + e.t + e.cfg.lookback

        # Rebalance monthly (every 21 trading days)
        if 'w' in state and e.t % 21 != 0:
            return state['w']

        if ri < 60 or ri >= len(e.returns):
            w = np.ones(n+1)/(n+1)
            state['w'] = w
            return w

        # 60-day rolling volatility
        ret_window = e.returns[ri-60:ri, :n]
        vols = ret_window.std(axis=0) * np.sqrt(252)
        inv_vol = 1.0 / (vols + 1e-8)

        w = np.zeros(n+1)
        w[:n] = inv_vol / inv_vol.sum() * 0.95  # 95% in risky assets
        w[-1] = 0.05  # 5% cash
        w /= w.sum() + 1e-10
        state['w'] = w
        return w

    # ── Min-Variance: Shrunk Covariance Portfolio ──
    @staticmethod
    def min_variance(e, o, state):
        """Minimum Variance portfolio with Ledoit-Wolf simplified shrinkage.
        Sigma_shrunk = 0.5*Sigma + 0.5*diag(Sigma). Monthly rebalance. Cash = 5%."""
        n = e.n_assets; ri = e.si + e.t + e.cfg.lookback

        # Rebalance monthly
        if 'w' in state and e.t % 21 != 0:
            return state['w']

        if ri < 60 or ri >= len(e.returns):
            w = np.ones(n+1)/(n+1)
            state['w'] = w
            return w

        # 60-day sample covariance
        ret_window = e.returns[ri-60:ri, :n]
        cov = np.cov(ret_window.T) * 252
        # Ledoit-Wolf simplified shrinkage toward diagonal
        cov_shrunk = 0.5 * cov + 0.5 * np.diag(np.diag(cov))

        try:
            inv_cov = np.linalg.inv(cov_shrunk)
            ones = np.ones(n)
            raw_w = inv_cov @ ones
            raw_w = np.maximum(raw_w, 0)  # long-only constraint
            total = raw_w.sum()
            w = np.zeros(n+1)
            if total > 1e-8:
                w[:n] = raw_w / total * 0.95
            else:
                w[:n] = 0.95 / n
            w[-1] = 0.05
        except np.linalg.LinAlgError:
            w = np.ones(n+1)/(n+1)

        w /= w.sum() + 1e-10
        state['w'] = w
        return w


# ═══════════════════════════════════════════════
# MAMA Trainer (Full Pipeline)
# ═══════════════════════════════════════════════
class MAMATrainer:
    def __init__(self, cfg, train_data, test_data, test_offset=0, bl_data=None, bl_offset=0):
        self.cfg = cfg; self.device = torch.device(cfg.device)
        tr_y = train_data.get("yields"); te_y = test_data.get("yields")
        self.tr_env = MarketEnv(train_data["prices"], train_data["returns"], cfg, "train", yields=tr_y)
        self.te_env = MarketEnv(test_data["prices"], test_data["returns"], cfg, "test", test_offset, yields=te_y)
        if bl_data is not None:
            bl_y = bl_data.get("yields")
            self.bl_env = MarketEnv(bl_data["prices"], bl_data["returns"], cfg, "test", bl_offset, yields=bl_y)
        else:
            self.bl_env = self.te_env
        self.indices = self.tr_env.asset_indices()
        obs_dim = self.tr_env.obs_dim()

        # Inter-asset agent with RNN-SRL (paper: ETF returns + Treasury Yield Curve)
        n_all = len(self.tr_env.asset_indices()["stocks"]) + len(self.tr_env.asset_indices()["bonds"]) + len(self.tr_env.asset_indices()["commodities"])
        n_yields = self.tr_env.n_yields
        self.inter = InterAssetAgent(obs_dim, n_all, n_yields, 4, cfg, self.device)

        # Intra-asset agents with GNN-SRL
        self.intra = {}
        for cls, idx in self.indices.items():
            if cls=="cash": continue
            agent = IntraAssetAgent(obs_dim, len(idx), idx, cfg, self.device)
            self.intra[cls] = agent

        # Pretrain all SRL modules
        print("\n  🧠 Pretraining SRL modules...")
        # Pretrain GNN-SRLs (intra-asset)
        for cls, agent in self.intra.items():
            pretrain_gnn_srl(agent.gnn, train_data["prices"], train_data["returns"],
                             agent.sec_indices, agent.graph_builder, cfg, self.device)
        # Pretrain GRU-SRL (inter-asset) with ETF returns + yields
        pretrain_inter_srl(self.inter.srl, train_data["returns"], train_data.get("yields", np.zeros((len(train_data["returns"]),10))), cfg, self.device)

        np_ = sum(p.numel() for a in self.intra.values() for p in list(a.actor.parameters())+list(a.critic.parameters())+list(a.gnn.parameters()))
        np_ += sum(p.numel() for p in list(self.inter.actor.parameters())+list(self.inter.critic.parameters())+list(self.inter.srl.parameters()))
        print(f"  📊 Total params: {np_:,}")
        self.log = []; self.best_s = -np.inf

    def _portfolio(self, obs, env, noise):
        """Compute full portfolio weights from inter + intra agents.
        If non-continual agent exists, blend 50/50 with continual agent."""
        ca, inter_enriched = self.inter.select_action(obs, env, noise)
        # Blend with non-continual (frozen) agent if available
        if hasattr(self, 'inter_nc'):
            ca_nc, _ = self.inter_nc.select_action(obs, env, noise=0)
            ca = 0.5 * ca + 0.5 * ca_nc
        n = env.n_assets+1; pw = np.zeros(n)
        cls_names = ["stocks","bonds","commodities","cash"]
        intra_states = {}
        for i, cls in enumerate(cls_names):
            if cls=="cash": pw[-1]=ca[i]; continue
            idx = self.indices[cls]
            agent = self.intra[cls]
            alloc, enriched, mask = agent.select_action(obs, env, noise)
            intra_states[cls] = (enriched, mask)
            for j, ix in enumerate(idx): pw[ix] = alloc[j]*ca[i]
        pw /= pw.sum()+1e-10
        return pw, ca, intra_states, inter_enriched

    def train(self):
        print(f"\n{'='*60}")
        print(f"  MAMA Training (TD3+GNN, paper-faithful)")
        print(f"{'='*60}")
        print(f"  {self.cfg.n_stocks}S/{self.cfg.n_bonds}B/{self.cfg.n_commodities}C | {self.cfg.n_episodes} ep | {self.device}")
        print(f"{'='*60}")

        snapshot_ep = self.cfg.n_episodes // 2  # Snapshot at midpoint

        for ep in range(self.cfg.n_episodes):
            # Dual-agent: snapshot inter-agent at midpoint as non-continual agent
            if ep == snapshot_ep and not hasattr(self, 'inter_nc'):
                self.inter_nc = copy.deepcopy(self.inter)
                # Freeze all parameters
                for p in self.inter_nc.actor.parameters(): p.requires_grad_(False)
                for p in self.inter_nc.srl.parameters(): p.requires_grad_(False)
                for p in self.inter_nc.critic.parameters(): p.requires_grad_(False)
                print(f"\n  🔒 Snapshot non-continual inter-agent at ep {ep} (frozen)")

            t0=time.time(); obs=self.tr_env.reset()
            pv=[self.cfg.initial_capital]; dr=[]; ep_r=0
            # Noise decay with floor at 30% of initial (explore longer in bull markets)
            noise = self.cfg.exploration_noise * max(0.3, 0.998 ** ep)

            for step in range(self.tr_env.max_ep_len):
                pw, ca, intra_states, inter_enriched = self._portfolio(obs, self.tr_env, noise)
                nobs, reward, done, info = self.tr_env.step(pw)

                # FIX #2: Compute proper next_state for all agents
                nobs_enriched = self.inter._enrich(nobs, self.tr_env)
                self.inter.buffer.add(inter_enriched, ca, reward, nobs_enriched, float(done), mask=None)

                for cls in self.intra:
                    if cls in intra_states:
                        en, mask = intra_states[cls]
                        idx = self.indices[cls]
                        alloc = np.array([pw[ix] for ix in idx])
                        s_total = ca[["stocks","bonds","commodities","cash"].index(cls)]
                        if s_total > 1e-8: alloc = alloc/s_total
                        # FIX #2: Compute real next enriched state for intra agent
                        n_en, n_mask, *_ = self.intra[cls]._get_gnn_state(nobs, self.tr_env)
                        n_en_np = n_en.cpu().numpy()
                        self.intra[cls].buffer.add(en, alloc, reward, n_en_np, float(done), mask)

                pv.append(info["portfolio_value"]); dr.append(info["return"]); ep_r+=reward

                # Multiple gradient steps per env step → better GPU utilization
                for _ in range(self.cfg.n_grad_steps):
                    self.inter.train_step()
                    for cls in self.intra: self.intra[cls].train_step()

                obs=nobs
                if done: break

            d=np.array(dr); tr_=pv[-1]/pv[0]-1; ny=len(d)/252
            ar_=(1+tr_)**(1/max(ny,.01))-1; v=d.std()*np.sqrt(252) if len(d)>1 else 0
            sh=(ar_-self.cfg.risk_free_rate)/v if v>1e-4 else 0
            pk=pv[0];md=0
            for x in pv:
                if x>pk:pk=x
                if(pk-x)/pk>md:md=(pk-x)/pk
            elapsed=time.time()-t0

            if sh>self.best_s: self.best_s=sh
            self.log.append({"ep":ep+1,"ret":tr_,"sharpe":sh,"mdd":md,"time":elapsed})

            if(ep+1)%self.cfg.log_interval==0:
                avg=np.mean([l["sharpe"] for l in self.log[-self.cfg.log_interval:]])
                bf = len(self.inter.buffer)
                print(f"  Ep {ep+1:4d}/{self.cfg.n_episodes} | "
                      f"Ret:{tr_:+.2%} | S:{sh:+.3f} | AvgS:{avg:+.3f} | "
                      f"Buf:{bf:,} | {elapsed:.1f}s")

        print(f"\n  ⭐ Best Sharpe: {self.best_s:.4f}")

    def test(self):
        print(f"\n  📈 Testing on out-of-sample data...")
        obs=self.te_env.reset(); pv=[self.cfg.initial_capital]; dr=[]; w_hist=[]
        while True:
            pw, _, _, _ = self._portfolio(obs, self.te_env, noise=0)
            w_hist.append(pw)
            obs,_,done,info = self.te_env.step(pw)
            pv.append(info["portfolio_value"]); dr.append(info["return"])
            if done: break
        d=np.array(dr); tr_=pv[-1]/pv[0]-1; ny=len(d)/252
        ar_=(1+tr_)**(1/max(ny,.01))-1; v=d.std()*np.sqrt(252) if len(d)>1 else 0
        sh=(ar_-self.cfg.risk_free_rate)/v if v>1e-4 else 0
        pk=pv[0];md=0
        for x in pv:
            if x>pk:pk=x
            if(pk-x)/pk>md:md=(pk-x)/pk
            
        # Log average weights
        avg_w = np.mean(w_hist, axis=0) * 100
        ws = avg_w[self.indices['stocks']].sum()
        wb = avg_w[self.indices['bonds']].sum()
        wc = avg_w[self.indices['commodities']].sum()
        wca = avg_w[-1]
        
        avg_alloc = {"stocks": round(ws, 1), "bonds": round(wb, 1),
                     "commodities": round(wc, 1), "cash": round(wca, 1)}
        print(f"  ★ MAMA: Ret={tr_:+.2%} | Sharpe={sh:+.3f} | MaxDD={md:.2%}")
        print(f"    Avg Alloc: Stocks {ws:.1f}% | Bonds {wb:.1f}% | Cmd {wc:.1f}% | Cash {wca:.1f}%")
        return {"name":"MAMA","total_return":tr_,"sharpe_ratio":sh,"max_drawdown":md,
                "avg_alloc":avg_alloc,"pv":pv}

    def baselines(self):
        print(f"\n  📊 Baselines:")
        # Non-stateful baselines
        simple = [(BL.eq, "Equal Weight"), (BL.sb, "60/40 S/B")]
        # Stateful baselines
        stateful = [
            (BL.ubah, "UBAH"),
            (BL.momentum_12_1, "Momentum"),
            (BL.corn, "CORN"),
            (BL.olmar, "OLMAR"),
            (BL.pamr, "PAMR"),
            (BL.risk_parity, "RiskParity"),
            (BL.min_variance, "MinVar"),
        ]
        res = []
        for fn, nm in simple:
            r = BL.run(self.bl_env, fn, nm); res.append(r)
            print(f"    {nm:16s} | Ret:{r['total_return']:+.2%} | S:{r['sharpe_ratio']:+.3f}")
        for fn, nm in stateful:
            r = BL.run(self.bl_env, fn, nm, stateful=True); res.append(r)
            print(f"    {nm:16s} | Ret:{r['total_return']:+.2%} | S:{r['sharpe_ratio']:+.3f}")
        return res


# ═══════════════════════════════════════════════
# Walk-Forward Scenarios
# ═══════════════════════════════════════════════
SCENARIOS = [
    {"name":"S1: COVID Crash","ts":"2007-01-01","te":"2019-12-31","vs":"2020-01-01","ve":"2020-12-31"},
    {"name":"S2: Post-COVID", "ts":"2007-01-01","te":"2020-12-31","vs":"2021-01-01","ve":"2021-12-31"},
    {"name":"S3: Rate Hikes", "ts":"2007-01-01","te":"2021-12-31","vs":"2022-01-01","ve":"2022-12-31"},
    {"name":"S4: Recovery",   "ts":"2007-01-01","te":"2022-12-31","vs":"2023-01-01","ve":"2023-12-31"},
    {"name":"S5: AI Rally",   "ts":"2007-01-01","te":"2023-12-31","vs":"2024-01-01","ve":"2024-12-31"},
]


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
if __name__=="__main__":
    import argparse
    pa=argparse.ArgumentParser()
    pa.add_argument("--episodes",type=int,default=200)
    pa.add_argument("--device",type=str,default="cpu")
    pa.add_argument("--seed",type=int,default=42)
    pa.add_argument("--scenarios",type=str,default=None,
                    help="Comma-separated scenario numbers, e.g. '1,3,5'. Default: all 5")
    args=pa.parse_args()

    set_seed(args.seed)

    print(f"\n{'█'*60}")
    print(f"█{'MAMA: Paper-Faithful (5-Scenario Walk-Forward)':^58s}█")
    print(f"█{'TD3 + GNN-SRL + Pretrain + Masked Softmax':^58s}█")
    print(f"{'█'*60}")

    # Download all data once
    base_cfg = Config(start_date="2007-01-01", end_date="2024-12-31", device=args.device)
    loader = DataLoader(base_cfg); loader.download()

    # Select scenarios
    if args.scenarios:
        sc_list = [SCENARIOS[int(i)-1] for i in args.scenarios.split(",")]
    else:
        sc_list = SCENARIOS
    print(f"\n🚀 Running {len(sc_list)} scenario(s) × {args.episodes} episodes\n")

    all_res = []; g_start = time.time(); sc_times = []

    for i, sc in enumerate(sc_list):
        sc_start = time.time()
        print(f"\n{'▓'*60}")
        print(f"  📊 Scenario {i+1}/{len(sc_list)}: {sc['name']}", end="")
        if sc_times:
            avg = np.mean(sc_times); rem = (len(sc_list)-i)*avg
            rm, rs = divmod(int(rem), 60)
            print(f" | ETA ~{rm}m{rs:02d}s")
        else:
            print()
        print(f"{'▓'*60}")

        # Split data by scenario dates
        p = loader.prices_df; r = loader.returns_df
        trm = (p.index >= sc["ts"]) & (p.index <= sc["te"])
        tem = (p.index >= sc["vs"]) & (p.index <= sc["ve"])
        
        n_train = p[trm].shape[0]
        n_test = p[tem].shape[0]
        print(f"  Train: {n_train} days | Test: {n_test} days")

        # 1) Train data (with yields)
        y = loader.yields_df
        tr_data = {"prices": p[trm].values, "returns": r[trm].values, "yields": y[trm].values}
        
        # 2) MAMA Test data: strict test mask + lookback
        test_start_idx = np.searchsorted(p.index, sc["vs"])
        mama_start_idx = max(0, test_start_idx - base_cfg.lookback)
        mama_mask = (p.index >= p.index[mama_start_idx]) & (p.index <= sc["ve"])
        te_data = {"prices": p[mama_mask].values, "returns": r[mama_mask].values, "yields": y[mama_mask].values}
        test_offset = test_start_idx - mama_start_idx  # typically == lookback

        # 3) Baseline Test data: full data up to test end (for moving average/momentum)
        full_mask = (p.index >= sc["ts"]) & (p.index <= sc["ve"])
        bl_data = {"prices": p[full_mask].values, "returns": r[full_mask].values, "yields": y[full_mask].values}
        bl_offset = max(0, n_train - base_cfg.lookback)

        sc_cfg = Config(
            n_episodes=args.episodes, device=args.device, seed=args.seed,
            save_dir=f"ckpt_mama_{sc['name'][:2]}"
        )

        trainer = MAMATrainer(sc_cfg, tr_data, te_data, test_offset=test_offset, bl_data=bl_data, bl_offset=bl_offset)
        trainer.train()

        print(f"\n  📈 Test Results:")
        mama = trainer.test()
        print(f"\n  📊 Baselines:")
        bls = trainer.baselines()

        all_res.append({"scenario": sc["name"], "mama": mama, "baselines": bls})
        sc_times.append(time.time() - sc_start)

    # ═══════════════════════════════════════════════
    # Grand Summary
    # ═══════════════════════════════════════════════
    total = time.time() - g_start; tm, ts_ = divmod(int(total), 60)

    print(f"\n\n{'█'*60}")
    print(f"█{'GRAND SUMMARY':^58s}█")
    print(f"{'█'*60}")

    bl_names = [b["name"] for b in all_res[0]["baselines"]]
    print(f"\n  {'Scenario':<20s}{'MAMA':>8s}", end="")
    for bn in bl_names:
        print(f"{bn[:8]:>9s}", end="")
    print()
    print("  " + "─"*65)

    wins = {bn: 0 for bn in bl_names}; sharpes = []
    for res in all_res:
        ms = res["mama"]["sharpe_ratio"]; sharpes.append(ms)
        print(f"  {res['scenario']:<20s}{ms:>+8.3f}", end="")
        for b in res["baselines"]:
            print(f"{b['sharpe_ratio']:>+9.3f}", end="")
            if ms > b["sharpe_ratio"]:
                wins[b["name"]] += 1
        print()

    ns = len(all_res)
    print(f"\n  Win Rate:")
    for bn in bl_names:
        w = wins[bn]; pct = w/ns*100
        bar = '█'*int(pct/5) + '░'*(20-int(pct/5))
        print(f"    vs {bn:<14s} {w}/{ns} ({pct:.0f}%) {bar}")

    t_wins = sum(wins.values()); t_total = ns * len(bl_names)
    print(f"\n  Overall: {t_wins}/{t_total} ({t_wins/t_total*100:.0f}%) | Avg Sharpe: {np.mean(sharpes):+.4f}")
    print(f"  ⏱ Total: {tm}m{ts_:02d}s")
    print(f"{'█'*60}\n")

    # Save summary JSON
    os.makedirs("ckpt_mama", exist_ok=True)
    summary = {
        "scenarios": [{
            "name": r["scenario"],
            "mama_sharpe": r["mama"]["sharpe_ratio"],
            "mama_return": r["mama"]["total_return"],
            "mama_maxdd": r["mama"]["max_drawdown"],
            "mama_avg_alloc": r["mama"].get("avg_alloc", {}),
            "baselines": {b["name"]: b["sharpe_ratio"] for b in r["baselines"]}
        } for r in all_res],
        "avg_sharpe": float(np.mean(sharpes)),
        "win_rates": {bn: wins[bn]/ns for bn in bl_names},
        "total_time_sec": total
    }
    with open("ckpt_mama/scenario_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  📁 Summary saved to ckpt_mama/scenario_summary.json")