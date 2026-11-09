import sys,zipfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();out=r.parent/(r.name+'_return.zip')
with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(r.rglob('*')):
  if p.is_file() and p.suffix not in {'.pt','.zip','.pyc'}:z.write(p,p.relative_to(r))
print('SEND THIS FILE:',out)
