import yfinance as y,pandas as p,numpy as n,matplotlib.pyplot as l
d=y.download('^GSPC VBMFX',start='1987-1-1').Close.dropna()
s,b=d['^GSPC'],d['VBMFX']
r=s.pct_change()
m=lambda k:s.rolling(k).mean()
x=lambda u,v:(u>v)&(u.shift()<=v.shift())
w=s*n.nan;w[x(m(50),m(200))&(m(200)>m(200).shift())]=1;w[x(m(9),m(21))]=0
p.DataFrame({'MA Strat':r*w.ffill().shift(),'Vanilla SP500':r,'75/25':r*.75+b.pct_change()*.25}).add(1).cumprod().plot(logy=True,figsize=(10,6),title='Strategy vs Vanilla vs 75/25')
l.show()
