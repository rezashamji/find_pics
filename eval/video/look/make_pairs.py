import json, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
Q = {d['k']: d['query'] for d in json.load(open('eval/video/queries.json'))}
rng = np.random.default_rng(0)
def load(u):
    im = Image.open(f'data/public/pexels_frames/{u}.jpg').convert('RGB')
    return im.resize((900, round(im.height*900/im.width)))
try: font = ImageFont.truetype('DejaVuSans.ttf', 28)
except Exception: font = ImageFont.load_default()
for k in [13, 21, 15, 7]:
    df = pd.read_parquet(f'eval/video/oracle_{k}.parquet')
    cands = []
    for v, g in df.groupby('item_row'):
        b = g.loc[g.look.idxmax()]; t = g.loc[g.p.idxmax()]
        if b.p < 0.7 and t.p >= 0.7: cands.append(v)
    pick = sorted(rng.choice(cands, size=min(2, len(cands)), replace=False)) if cands else []
    print(k, Q[k], '| candidates', len(cands), 'picked', pick)
    for v in pick:
        g = df[df.item_row == v]
        b = g.loc[g.look.idxmax()]; t = g.loc[g.p.idxmax()]
        A, B = load(int(b.unit)), load(int(t.unit))
        H = max(A.height, B.height) + 50
        c = Image.new('RGB', (1810, H), 'white'); c.paste(A, (0, 50)); c.paste(B, (910, 50))
        d = ImageDraw.Draw(c)
        d.text((5, 8), f'BEST-LOOK unit {int(b.unit)} p={b.p:.2f} look={b.look:.3f}', fill='black', font=font)
        d.text((915, 8), f'MAX-P unit {int(t.unit)} p={t.p:.2f} look={t.look:.3f}', fill='black', font=font)
        c.save(f'eval/video/look/k{k}_{v}.jpg', quality=90)
        print('  ', v, 'nframes', len(g), 'best', int(b.unit), round(b.p,2), 'maxp', int(t.unit), round(t.p,2))
