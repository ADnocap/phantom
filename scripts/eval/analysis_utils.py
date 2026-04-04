"""
Phantom v5 Analysis Utilities.

Reusable functions for model evaluation, cross-sectional analysis,
distributional diagnostics, and portfolio backtesting.
"""

import json
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr, t as scipy_t
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────

ASSET_NAMES = {0: 'Crypto', 1: 'Equity', 2: 'Forex', 3: 'Commodity'}
ASSET_COLORS = {0: '#F7931A', 1: '#2196F3', 2: '#4CAF50', 3: '#FF9800'}
HORIZON_SUBSET = [0, 4, 9, 14, 19, 24, 29]  # 1d, 5d, 10d, 15d, 20d, 25d, 30d


# ── Model Loading ──────────────────────────────────────────────

def load_model(checkpoint_path='checkpoints_v5/best.pt'):
    """Load trained Phantom model from checkpoint."""
    from src.model import PhantomConfig, PhantomModel

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    cfg = PhantomConfig(**{k: v for k, v in cfg_dict.items()
                          if k in PhantomConfig.__dataclass_fields__})
    model = PhantomModel(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    info = {
        'step': ckpt['step'],
        'best_val_loss': ckpt['best_val_loss'],
        'n_params': sum(p.numel() for p in model.parameters()),
        'config': cfg,
    }
    return model, info


def load_test_data(data_path='data/processed_v5/test.npz'):
    """Load test dataset with all fields."""
    d = np.load(data_path, allow_pickle=True)
    data = {
        'X': d['X'].astype(np.float32),
        'Y_relative': d['Y_relative'].astype(np.float32),
        'Y_absolute': d['Y'].astype(np.float32),
        'asset_type': d['asset_type'].astype(np.int64),
        'dates': d['dates_end'] if 'dates_end' in d else None,
        'asset_id': d['asset_id'] if 'asset_id' in d else None,
        'realized_vol': d['realized_vol'].astype(np.float32),
    }
    return data


def load_asset_meta(meta_path='data/processed_v5/asset_meta.json'):
    """Load asset metadata (id→name mapping)."""
    with open(meta_path) as f:
        return json.load(f)


# ── Inference ──────────────────────────────────────────────────

def predict(model, X, batch_size=1024, verbose=True):
    """Run model inference on input data.

    Returns dict with mu, sigma, nu arrays, each (N, 30).
    """
    all_mu, all_sigma, all_nu = [], [], []
    N = len(X)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            if verbose and (i // batch_size) % 10 == 0:
                print(f"\r  Batch {i//batch_size+1}/{(N+batch_size-1)//batch_size}", end='', flush=True)
            x = torch.from_numpy(X[i:i+batch_size])
            log_pi, mu, sigma, nu = model(x)
            all_mu.append(mu.numpy())
            all_sigma.append(sigma.numpy())
            if nu is not None:
                all_nu.append(nu.numpy())
    if verbose:
        print()

    return {
        'mu': np.concatenate(all_mu).squeeze(-1),       # (N, 30)
        'sigma': np.concatenate(all_sigma).squeeze(-1),  # (N, 30)
        'nu': np.concatenate(all_nu).squeeze(-1) if all_nu else None,
    }


def get_encoder_embeddings(model, X, batch_size=1024):
    """Extract encoder representations for analysis."""
    all_enc = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i+batch_size])
            enc = model.encode(x)  # (B, N_patches, d_model)
            all_enc.append(enc.mean(dim=1).numpy())  # pool to (B, d_model)
    return np.concatenate(all_enc)


# ── Cross-Sectional Metrics ───────────────────────────────────

def compute_rank_ic(pred, actual, dates, min_assets=5):
    """Spearman rank IC per date, returns array of ICs."""
    unique_dates = np.unique(dates)
    ics, ic_dates = [], []
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < min_assets:
            continue
        ic, _ = spearmanr(pred[mask], actual[mask])
        if np.isfinite(ic):
            ics.append(ic)
            ic_dates.append(d)
    return np.array(ics), np.array(ic_dates)


def compute_pearson_ic(pred, actual, dates, min_assets=5):
    """Pearson IC per date."""
    unique_dates = np.unique(dates)
    ics = []
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < min_assets:
            continue
        ic, _ = pearsonr(pred[mask], actual[mask])
        if np.isfinite(ic):
            ics.append(ic)
    return np.array(ics)


def ic_by_horizon(pred_mu, Y_rel, dates, horizons=None):
    """Compute Rank IC at each horizon. Returns dict {horizon: (mean, std, n)}."""
    if horizons is None:
        horizons = list(range(30))
    results = {}
    for h in horizons:
        ics, _ = compute_rank_ic(pred_mu[:, h], Y_rel[:, h], dates)
        results[h + 1] = {
            'mean': ics.mean(), 'std': ics.std(), 'n': len(ics),
            'tstat': ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-8),
            'ics': ics,
        }
    return results


def ic_by_asset_class(pred_mu, Y_rel, dates, asset_type, h_idx=9):
    """Compute Rank IC per asset class at a given horizon."""
    results = {}
    for t in sorted(np.unique(asset_type)):
        mask = asset_type == t
        ics, _ = compute_rank_ic(pred_mu[mask, h_idx], Y_rel[mask, h_idx], dates[mask])
        name = ASSET_NAMES.get(t, f'type_{t}')
        results[name] = {
            'mean': ics.mean(), 'std': ics.std(), 'n': len(ics),
            'tstat': ics.mean() / (ics.std() / np.sqrt(len(ics)) + 1e-8),
        }
    return results


def ic_over_time(pred_mu, Y_rel, dates, h_idx=9, window=20):
    """Rolling Rank IC over time (moving window of dates)."""
    ics, ic_dates = compute_rank_ic(pred_mu[:, h_idx], Y_rel[:, h_idx], dates)
    if len(ics) < window:
        return ics, ic_dates, ics
    rolling = np.convolve(ics, np.ones(window)/window, mode='valid')
    return ics, ic_dates, rolling


# ── Portfolio Backtest ─────────────────────────────────────────

def long_short_backtest(pred, actual, dates, quantile=0.2, min_assets=10):
    """Long top quintile, short bottom quintile per date."""
    unique_dates = np.unique(dates)
    daily_returns, valid_dates = [], []
    long_picks, short_picks = [], []

    for d in unique_dates:
        mask = dates == d
        n = mask.sum()
        if n < min_assets:
            continue
        p = pred[mask]
        a = actual[mask]
        k = max(1, int(n * quantile))

        top_idx = np.argsort(p)[-k:]
        bot_idx = np.argsort(p)[:k]

        ls_ret = a[top_idx].mean() - a[bot_idx].mean()
        daily_returns.append(ls_ret)
        valid_dates.append(d)
        long_picks.append(k)
        short_picks.append(k)

    daily_returns = np.array(daily_returns)
    valid_dates = np.array(valid_dates)

    stats = {
        'daily_returns': daily_returns,
        'dates': valid_dates,
        'mean_return': daily_returns.mean(),
        'sharpe': daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252),
        'cumulative': np.cumsum(daily_returns),
        'win_rate': (daily_returns > 0).mean(),
        'max_drawdown': _max_drawdown(np.cumsum(daily_returns)),
        'n_days': len(daily_returns),
    }
    return stats


def _max_drawdown(cumulative):
    """Compute maximum drawdown from cumulative returns."""
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    return dd.min()


def backtest_by_quantile(pred, actual, dates, quantiles=[0.1, 0.2, 0.3], min_assets=10):
    """Test different quantile thresholds for the long-short strategy."""
    results = {}
    for q in quantiles:
        stats = long_short_backtest(pred, actual, dates, quantile=q, min_assets=min_assets)
        results[q] = stats
    return results


# ── Distributional Calibration ─────────────────────────────────

def compute_pit(Y, mu, sigma, nu):
    """Probability Integral Transform values."""
    return scipy_t.cdf(Y, df=nu, loc=mu, scale=sigma)


def compute_coverage(Y, mu, sigma, nu, levels=[0.50, 0.80, 0.90, 0.95]):
    """Coverage at confidence levels."""
    coverages = {}
    for level in levels:
        alpha = (1 - level) / 2
        lo = scipy_t.ppf(alpha, df=nu, loc=mu, scale=sigma)
        hi = scipy_t.ppf(1 - alpha, df=nu, loc=mu, scale=sigma)
        coverages[level] = ((Y >= lo) & (Y <= hi)).mean()
    return coverages


def compute_ece(pit_values, n_bins=10):
    """Expected Calibration Error from PIT values."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (pit_values >= bins[i]) & (pit_values < bins[i+1])
        ece += abs(mask.mean() - 1.0 / n_bins)
    return ece / n_bins * n_bins  # normalize


def compute_crps_per_sample(Y, mu, sigma, nu, n_mc=300):
    """Sample-based CRPS per sample. Returns (N,) array."""
    rng = np.random.default_rng(42)
    samples = scipy_t.rvs(df=nu, loc=mu, scale=sigma,
                           size=(n_mc, len(Y)), random_state=rng).T  # (N, n_mc)
    term1 = np.abs(samples - Y[:, None]).mean(axis=1)
    s_sorted = np.sort(samples, axis=1)
    diffs = s_sorted[:, 1:] - s_sorted[:, :-1]
    idx_w = np.arange(1, n_mc)
    weights = idx_w * (n_mc - idx_w)
    term2 = (diffs * weights[None, :]).sum(axis=1) * 2.0 / (n_mc * n_mc)
    return term1 - 0.5 * term2


# ── Term Structure Analysis ────────────────────────────────────

def sigma_term_structure(sigma, asset_type=None):
    """Mean sigma across horizons. Optionally split by asset class."""
    if asset_type is None:
        return sigma.mean(axis=0)  # (30,)
    results = {}
    for t in sorted(np.unique(asset_type)):
        mask = asset_type == t
        name = ASSET_NAMES.get(t, f'type_{t}')
        results[name] = sigma[mask].mean(axis=0)
    return results


def mu_term_structure(mu, asset_type=None):
    """Mean and std of predicted mu across horizons."""
    if asset_type is None:
        return mu.mean(axis=0), mu.std(axis=0)
    results = {}
    for t in sorted(np.unique(asset_type)):
        mask = asset_type == t
        name = ASSET_NAMES.get(t, f'type_{t}')
        results[name] = (mu[mask].mean(axis=0), mu[mask].std(axis=0))
    return results


def nu_distribution(nu, asset_type=None):
    """Analyze degrees-of-freedom distribution."""
    if asset_type is None:
        return {'mean': nu.mean(), 'median': np.median(nu),
                'q10': np.percentile(nu, 10), 'q90': np.percentile(nu, 90)}
    results = {}
    for t in sorted(np.unique(asset_type)):
        mask = asset_type == t
        name = ASSET_NAMES.get(t, f'type_{t}')
        results[name] = {'mean': nu[mask].mean(), 'median': np.median(nu[mask])}
    return results


# ── Feature Importance (ablation) ──────────────────────────────

def ablate_channel(model, X, channel_idx, batch_size=1024):
    """Zero out one input channel and measure prediction change."""
    X_ablated = X.copy()
    X_ablated[:, :, channel_idx] = 0.0
    preds_orig = predict(model, X, batch_size=batch_size, verbose=False)
    preds_ablated = predict(model, X_ablated, batch_size=batch_size, verbose=False)
    mu_diff = np.abs(preds_orig['mu'] - preds_ablated['mu']).mean()
    sigma_diff = np.abs(preds_orig['sigma'] - preds_ablated['sigma']).mean()
    return {'mu_change': mu_diff, 'sigma_change': sigma_diff}


CHANNEL_NAMES = ['Log Return', 'Intraday Range', 'Body Ratio',
                 'Volume Ratio', 'Trailing Vol', 'Momentum']


# ── Architecture Visualization ─────────────────────────────────

def draw_architecture_block_diagram(cfg=None):
    """Draw a clean block diagram of the Phantom v5 architecture."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    if cfg is None:
        from src.model import PhantomConfig
        cfg = PhantomConfig(context_len=120, n_input_channels=6, d_model=512,
                            n_layers=8, n_decoder_layers=2, n_heads=8, d_ff=2048,
                            head_type='student_t', multi_horizon=True, max_horizon=30,
                            use_asset_classifier=True)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def box(x, y, w, h, text, color='#E3F2FD', ec='#1565C0', fontsize=9, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight)

    def arrow(x1, y1, x2, y2, color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    def label(x, y, text, fontsize=8, color='#666'):
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=color)

    # Title
    ax.text(8, 9.7, 'Phantom v5 Architecture', ha='center', fontsize=16, fontweight='bold')
    ax.text(8, 9.3, f'{sum(p.numel() for p in []) if False else "31.7M"} parameters | '
            f'Student-t distributional output at 30 horizons',
            ha='center', fontsize=10, color='#666')

    # ── Input ──
    box(0.5, 7.5, 3, 1.2, f'Input: (B, 120, 6)\n120 daily bars\n6 OHLCV channels',
        color='#FFF3E0', ec='#E65100', fontsize=9)

    # Channel labels
    channels = ['Returns', 'Range', 'Body', 'Volume', 'Vol30d', 'Mom10d']
    for i, ch in enumerate(channels):
        ax.text(0.5 + i*0.5, 7.2, ch, fontsize=6, ha='center', color='#E65100', rotation=45)

    # ── Patch Embedding ──
    arrow(2, 7.5, 2, 6.8)
    box(0.5, 5.8, 3, 1, f'Patch Embedding\nLinear({cfg.patch_len}x6 → {cfg.d_model})\n→ (B, 24, {cfg.d_model})',
        color='#E8F5E9', ec='#2E7D32', fontsize=9)
    label(2, 5.5, '+ Positional Encoding', fontsize=7)

    # ── Encoder ──
    arrow(2, 5.8, 2, 5.2)
    box(0.5, 3.5, 3, 1.7, f'Transformer Encoder\n{cfg.n_layers} layers\n'
        f'{cfg.n_heads} heads, d={cfg.d_model}\nFFN dim={cfg.d_ff}\n'
        f'Pre-LN, GELU, dropout={cfg.dropout}',
        color='#E3F2FD', ec='#1565C0', fontsize=9, bold=True)

    # ── Encoder output branches ──
    # Branch 1: to decoder
    arrow(3.5, 4.35, 5.5, 4.35)
    label(4.5, 4.6, 'enc_out\n(B, 24, 512)', fontsize=7)

    # Branch 2: to auxiliary heads
    arrow(2, 3.5, 2, 2.7)
    box(0.3, 1.5, 3.4, 1.2, f'Auxiliary Heads\nAdaptiveAvgPool → Linear\n'
        f'Asset classifier ({cfg.n_asset_types} classes)\nVol regressor (1 output)',
        color='#F3E5F5', ec='#6A1B9A', fontsize=8)
    arrow(2, 1.5, 2, 0.9)
    box(0.5, 0.3, 3, 0.6, 'Asset type + Realized vol',
        color='#F3E5F5', ec='#6A1B9A', fontsize=8)

    # ── Condition Dropout ──
    box(5.5, 4.8, 2.2, 0.6, 'Condition Dropout\np=0.15 (training)',
        color='#FFF9C4', ec='#F9A825', fontsize=8)
    arrow(6.6, 4.8, 6.6, 4.2)

    # ── Horizon Queries ──
    box(5, 2.5, 3, 1, f'30 Horizon Queries\nEmbed(1..30) → (B, 30, {cfg.d_model})\nLearned per-horizon identity',
        color='#FFF3E0', ec='#E65100', fontsize=9)
    arrow(6.5, 3.5, 6.5, 4.0)

    # ── Cross-Attention Decoder ──
    box(5, 4.0, 3.5, 0.8, f'Cross-Attention Decoder\n{cfg.n_decoder_layers} layers: Q=horizons, K/V=encoder',
        color='#E3F2FD', ec='#1565C0', fontsize=9, bold=True)
    label(7, 3.85, '→ (B, 30, 512)', fontsize=7, color='#1565C0')

    # ── Student-t Head ──
    arrow(8.5, 4.35, 9.5, 4.35)
    box(9.5, 3.5, 3, 1.7, f'Student-t Head (per horizon)\nLinear({cfg.d_model} → 256)\n'
        f'GELU + Dropout\nLinear(256 → 3)\n→ mu, sigma, nu',
        color='#FFEBEE', ec='#C62828', fontsize=9, bold=True)

    # ── Output ──
    arrow(12.5, 4.35, 13.5, 4.35)
    box(13.5, 3.2, 2.2, 2.3,
        f'Output\n(B, 30, 3)\n\nmu: location\nsigma: scale\nnu: tail weight\n\n'
        f'→ Student-t\ndistribution\nper horizon',
        color='#FFEBEE', ec='#C62828', fontsize=8)

    # ── Loss labels ──
    arrow(14.6, 3.2, 14.6, 2.2)
    box(13, 1.2, 3, 1, f'Loss = NLL + CRPS\n+ sqrt(h)-weighted Mean MSE\n+ Asset CE + Vol MSE',
        color='#E0F7FA', ec='#00838F', fontsize=8)
    label(14.5, 0.9, 'Target: Y_relative = Y - Y_class_mean', fontsize=7, color='#00838F')

    plt.tight_layout()
    return fig


def draw_data_flow_diagram():
    """Draw the data pipeline and training flow."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def box(x, y, w, h, text, color='#E3F2FD', ec='#1565C0', fontsize=8):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=color, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize)

    def arrow(x1, y1, x2, y2, color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    ax.text(8, 5.7, 'Phantom v5 Data Pipeline', ha='center', fontsize=14, fontweight='bold')

    # Data sources
    box(0.2, 4, 1.8, 1, 'Binance\nCryptoCompare\n61 crypto',
        color='#FFF3E0', ec='#F7931A', fontsize=7)
    box(2.2, 4, 1.8, 1, 'yfinance\n224 equities\n31 ETFs',
        color='#E3F2FD', ec='#2196F3', fontsize=7)
    box(4.2, 4, 1.8, 1, 'yfinance\n15 forex\n25 commodities\n57 EU equity',
        color='#E8F5E9', ec='#4CAF50', fontsize=7)

    arrow(1.1, 4, 1.1, 3.3)
    arrow(3.1, 4, 3.1, 3.3)
    arrow(5.1, 4, 5.1, 3.3)

    # Feature computation
    box(0.5, 2.3, 5.2, 1, '6-Channel OHLCV Features\nReturns | Range | Body | Volume | Vol30d | Momentum\n413 assets total',
        color='#FFF9C4', ec='#F9A825', fontsize=8)

    arrow(3.1, 2.3, 3.1, 1.6)

    # Rolling windows + relative returns
    box(0.5, 0.5, 5.2, 1.1, 'Rolling Windows (120d context, 30d forward)\n+ Cross-Sectional Normalization\nY_rel = Y_asset - mean(Y_class) per date\n1.09M train | 71K val | 81K test',
        color='#F3E5F5', ec='#6A1B9A', fontsize=8)

    # Arrow to model
    arrow(5.7, 1.05, 7, 1.05)

    # Model
    box(7, 0.2, 3, 1.7, 'Phantom v5 Model\n8-layer Transformer\n30-horizon Student-t\n\nNLL + CRPS + Mean MSE\n+ Asset Classifier',
        color='#E3F2FD', ec='#1565C0', fontsize=8)

    arrow(10, 1.05, 11, 1.05)

    # Evaluation
    box(11, 0.2, 2.2, 1.7, 'Evaluation\n\nRank IC: 0.092\nSharpe: 4.55\nCoverage: 95.7%\nIC t-stat: 11.8',
        color='#FFEBEE', ec='#C62828', fontsize=8)

    arrow(13.2, 1.05, 14, 1.05)

    # Application
    box(14, 0.2, 1.8, 1.7, 'Application\n\nPair Trading\nPortfolio\nAllocation\nRisk Mgmt',
        color='#E8F5E9', ec='#2E7D32', fontsize=8)

    # Training info
    box(7, 2.5, 3, 1, 'Training on LaRuche HPC\nA100 GPU, bfloat16\n20 epochs, batch 512\n~1 hour total',
        color='#E0F7FA', ec='#00838F', fontsize=7)
    arrow(8.5, 2.5, 8.5, 1.9)

    plt.tight_layout()
    return fig


def feature_importance(model, X_sample):
    """Ablation-based feature importance for all 6 channels."""
    results = {}
    for ch in range(6):
        diff = ablate_channel(model, X_sample, ch)
        results[CHANNEL_NAMES[ch]] = diff
    return results


# ── Model Architecture Summary ─────────────────────────────────

def model_summary(model, info):
    """Print model architecture summary."""
    cfg = info['config']
    lines = [
        f"Phantom v5 Model Summary",
        f"{'='*50}",
        f"Parameters:     {info['n_params']:,}",
        f"Checkpoint:     step {info['step']:,}",
        f"Best val loss:  {info['best_val_loss']:.4f}",
        f"",
        f"Encoder:",
        f"  Context:      {cfg.context_len} days ({cfg.context_len//cfg.patch_len} patches)",
        f"  Input:        {cfg.n_input_channels} channels (OHLCV features)",
        f"  d_model:      {cfg.d_model}",
        f"  Layers:       {cfg.n_layers} encoder + {cfg.n_decoder_layers} decoder",
        f"  Heads:        {cfg.n_heads}",
        f"  FFN dim:      {cfg.d_ff}",
        f"  Dropout:      {cfg.dropout}",
        f"",
        f"Decoder:",
        f"  Multi-horizon: {cfg.multi_horizon} ({cfg.max_horizon} horizons)",
        f"  Head:         {cfg.head_type} (Student-t, 3 params per horizon)",
        f"  Cond dropout: {cfg.cond_drop_prob}",
        f"",
        f"Auxiliary:",
        f"  Asset classifier: {cfg.n_asset_types} classes",
        f"  Vol regressor:    yes",
    ]
    return "\n".join(lines)
