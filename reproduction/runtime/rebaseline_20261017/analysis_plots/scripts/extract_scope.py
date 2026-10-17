from pathlib import Path
import json,subprocess,hashlib,os
D=Path(__file__).resolve().parents[1];ref=json.loads((D/'reference/paper_reference.json').read_text());assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
raw=subprocess.check_output(['pdftotext','-f','9','-l','9','-raw',ref['pdf'],'-'],text=True);start=raw.index('Based on this ob-');end=raw.index('Finally,',start);text=raw[start:end]
output=dict(pdf=ref['pdf'],pdf_sha256=ref['pdf_sha256'],pdf_page=9,location='Sec5.4 right column, second paragraph end, before Finally',extraction_command='pdftotext -f9 -l9 -raw PDF -',excerpt=text,meaning='Full configuration onward: E translation + H rotation from all samples unless noted; current H-accepted MAT differs')
(D/'reference/page9_all_samples_scope.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(output,ensure_ascii=False))
