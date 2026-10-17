import subprocess,json,hashlib,os,re
from pathlib import Path
D=Path(__file__).resolve().parents[1];pdf=D.parent.parents[2]/'paper/Park2024SPNv2.pdf'
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
raw=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True);pages=raw.split('\f');evidence=[]
for i,text in enumerate(pages,1):
 matches=list(re.finditer(r'Table\s+[12346](?:\b|:)|N = 1024|N=1024|on.line domain refinement',text,re.I))
 if matches:
  excerpts=[text[max(0,m.start()-180):min(len(text),m.start()+2900)] for m in matches]
  evidence.append(dict(pdf_page=i,excerpts=excerpts));subprocess.run(['pdftoppm','-f',str(i),'-l',str(i),'-scale-to','2000','-singlefile','-png',str(pdf),str(D/'reference'/('paper_page_'+str(i)))],check=True)
report=dict(pdf=str(pdf),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pdfinfo=subprocess.check_output(['pdfinfo',str(pdf)],text=True),excerpts=evidence,full_paper_text_not_saved=True)
(D/'reference/pdf_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
