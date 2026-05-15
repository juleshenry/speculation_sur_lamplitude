import yfinance as yf, pandas as pd, numpy as np, matplotlib.pyplot as plt

def ma_strategy(price):
    m = pd.DataFrame({9: price.rolling(9).mean(), 21: price.rolling(21).mean(),
                      50: price.rolling(50).mean(), 200: price.rolling(200).mean()}).dropna()
    p = price.reindex(m.index)

    def interp_price(idx, fast, slow):
        a = (fast.iloc[idx-1] - slow.iloc[idx-1])
        b = (fast.iloc[idx]   - slow.iloc[idx])
        x = a / (a - b)
        return p.iloc[idx-1] + x * (p.iloc[idx] - p.iloc[idx-1])

    buy  = (m[50] > m[200]) & (m[50].shift() <= m[200].shift()) & (m[200] > m[200].shift())
    sell = (m[9]  < m[21])  & (m[9].shift()  >= m[21].shift())

    pos, trades, entry = 0, [], None
    for i, dt in enumerate(m.index):
        if i == 0:
            continue
        if not pos and buy[dt]:
            pos, entry = 1, interp_price(i, m[50], m[200])
        elif pos and sell[dt]:
            ex = interp_price(i, m[9], m[21])
            trades.append((entry, ex, (ex-entry)/entry*100))
            pos, entry = 0, None

    t = pd.DataFrame(trades, columns=["entry","exit","pct"])
    return m, p, buy, sell, t

def strategy_return(price, entry_fast, entry_slow, exit_fast, exit_slow):
    m = pd.DataFrame({entry_fast: price.rolling(entry_fast).mean(),
                      entry_slow: price.rolling(entry_slow).mean(),
                      exit_fast: price.rolling(exit_fast).mean(),
                      exit_slow: price.rolling(exit_slow).mean()}).dropna()
    p = price.reindex(m.index)
    buy = (m[entry_fast] > m[entry_slow]) & (m[entry_fast].shift() <= m[entry_slow].shift()) & (m[entry_slow] > m[entry_slow].shift())
    sell = (m[exit_fast] < m[exit_slow]) & (m[exit_fast].shift() >= m[exit_slow].shift())
    pos = pd.Series(0, index=m.index)
    state = 0
    trades = 0
    for dt in m.index:
        if buy[dt]:
            state = 1
        elif sell[dt] and state:
            state = 0
            trades += 1
        pos.at[dt] = state
    strat = (1 + p.pct_change().fillna(0) * pos).cumprod()
    return strat.iloc[-1] - 1, trades

def rebalance_75_25(tqqq, ief):
    df = pd.concat({"tqqq": tqqq, "ief": ief}, axis=1).dropna()
    r = df.pct_change().fillna(0)
    months = r.index.to_period("M")
    v = 1.0
    vals = []
    for _, idx in r.groupby(months).groups.items():
        v_t = 0.75 * v
        v_i = 0.25 * v
        for dt in idx:
            v_t *= 1 + r.at[dt, "tqqq"]
            v_i *= 1 + r.at[dt, "ief"]
            v = v_t + v_i
            vals.append((dt, v))
    return pd.Series(dict(vals)).sort_index()

start = pd.Timestamp("2012-01-01")
tqqq = yf.download("TQQQ", period="15y", auto_adjust=True)["Close"].squeeze()
ief = yf.download("IEF", period="15y", auto_adjust=True)["Close"].squeeze()
tqqq = tqqq.loc[tqqq.index >= start]
ief = ief.loc[ief.index >= start]

m, p, buy, sell, trades = ma_strategy(tqqq)
bh = (tqqq / tqqq.iloc[0])
rb = rebalance_75_25(tqqq, ief)

pos = pd.Series(0, index=m.index)
state = 0
for dt in m.index:
    if buy[dt]:
        state = 1
    elif sell[dt]:
        state = 0
    pos.at[dt] = state
strat = (1 + p.pct_change().fillna(0) * pos).cumprod()

print("TQQQ 2012-01-01")
print(f"B&H    : {(bh.iloc[-1]-1)*100:.1f}%")
print(f"75/25  : {(rb.iloc[-1]-1)*100:.1f}%")
print(f"MA strat: {(strat.iloc[-1]-1)*100:.1f}%")
print(trades.to_string(index=False))

ef_list = [10, 15, 20, 30, 40, 50]
es_list = [100, 150, 200, 250]
xf_list = [5, 9, 12, 15]
xs_list = [21, 30, 40, 50]
rows = []
for ef in ef_list:
    for es in es_list:
        if ef >= es:
            continue
        for xf in xf_list:
            for xs in xs_list:
                if xf >= xs:
                    continue
                ret, tr = strategy_return(tqqq, ef, es, xf, xs)
                rows.append((ret, tr, ef, es, xf, xs))

top = sorted(rows, reverse=True)[:10]
print("\nTop MA sweeps (total return):")
for ret, tr, ef, es, xf, xs in top:
    print(f"{ret*100:7.1f}%  trades:{tr:2d}  entry {ef}/{es}  exit {xf}/{xs}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(p.index, p, color="#1f1f1f", lw=1.1, label="TQQQ")
ax1.plot(m.index, m[50], color="#d97706", lw=1, label="MA50")
ax1.plot(m.index, m[200], color="#2563eb", lw=1, label="MA200")
gc = m.index[(m[50] > m[200]) & (m[50].shift() <= m[200].shift())]
ax1.scatter(gc, p.loc[gc], marker="^", s=40, color="#111827", label="Golden cross")
ax1.set_title("TQQQ price with 50/200 crosses")
ax1.legend(loc="upper left", frameon=False, ncol=3)
ax1.grid(alpha=0.2)

ax2.plot(bh.index, bh, lw=1.2, label="TQQQ buy&hold")
ax2.plot(rb.index, rb, lw=1.2, label="75% TQQQ / 25% IEF")
ax2.plot(strat.index, strat, lw=1.2, label="MA strategy")
ax2.set_title("TQQQ strategies (from 2012)")
ax2.legend(loc="upper left", frameon=False)
ax2.grid(alpha=0.2)
fig.tight_layout()
plt.show()
