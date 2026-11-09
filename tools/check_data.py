import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import get_pool
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();rows={}
for split in ('train','val','test'):
 p=get_pool(split,'circuit');groups=sorted(set((str(f),int(w)) for f,w in zip(p.algo,p.width)))
 rows[split]={'rows':p.n,'groups':groups,'n_groups':len(groups)}
for x,y in [('train','val'),('train','test'),('val','test')]:assert not set(map(tuple,rows[x]['groups'])) & set(map(tuple,rows[y]['groups']))
Path(a.out).write_text(json.dumps(rows,indent=2));print({s:(d['rows'],d['n_groups']) for s,d in rows.items()})
