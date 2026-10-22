"""Render the requested counts as a self-contained, filterable table."""
import html
import json
from pathlib import Path
from explore.rlprompt.run import HERE

def build():
    out=HERE/'results'
    rows=json.loads((out/'summary.json').read_text())
    diag=json.loads((out/'verification.json').read_text())['diagnostics']
    model_keys={json.loads((out/k/'preflight.json').read_text())['model'].split('/')[-1]:k for k in diag}
    body=[]
    for model in dict.fromkeys(r['model'] for r in rows):
        cells=[r for r in rows if r['model']==model]
        trs=[]
        for r in cells:
            fields=[r['condition'],r['mode'],r['n'],r['violation'],r['leak'],r['boxed'],r['rule_mention'],r['grader_mention']]
            trs.append('<tr data-mode="'+r['mode']+'">'+''.join('<td>'+html.escape(str(v))+'</td>' for v in fields)+'</tr>')
        d=diag[model_keys[model]]
        note=f'<p class="diagnostic">200 replies; cap hit {d["cap_hit"]}; no final {d["no_final"]}; protocol failure {d["protocol_failure"]}.</p>'
        body.append('<section data-model="'+html.escape(model)+'"><h3>'+html.escape(model)+'</h3>'+note+'<div class="scroll"><table><thead><tr>'+''.join('<th>'+s+'</th>' for s in ['Condition','Thinking','n','Violation','Leak','Boxed','Rule','Grader'])+'</tr></thead><tbody>'+''.join(trs)+'</tbody></table></div></section>')
    models=list(dict.fromkeys(r['model'] for r in rows))
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font:14px var(--font-sans,system-ui);color:var(--foreground,#18212a);margin:0}p{color:var(--muted-foreground,#64707c);line-height:1.5;margin:0 0 14px}.diagnostic{font-size:12px;margin-bottom:5px}h3{font-size:15px;font-weight:650;margin:22px 0 8px}nav{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:14px}label{display:flex;gap:8px;align-items:center}select{font:inherit;color:inherit;background:var(--background,transparent);border:1px solid var(--border,#bbb);border-radius:5px;padding:5px 8px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--border,#ddd)}th:first-child,td:first-child{text-align:left;padding-left:0}th{font-weight:550;color:var(--muted-foreground,#64707c)}[hidden]{display:none!important}
</style></head><body><p>Exploratory RL-prompt run, not preregistered. Every value is a reply count. Each cell has 25 problems and one sample per problem. Rule and grader mentions are measured in reasoning only. Capped and empty finals remain in every denominator.</p><nav><label>Model <select id="model"><option value="all">All models</option>'''+''.join('<option>'+html.escape(m)+'</option>' for m in models)+'''</select></label><label>Thinking <select id="mode"><option value="all">Both</option><option value="on">On</option><option value="off">Off</option></select></label></nav>'''+''.join(body)+'''<script>
const model=document.querySelector('#model'),mode=document.querySelector('#mode');function update(){document.querySelectorAll('section').forEach(s=>s.hidden=model.value!=='all'&&s.dataset.model!==model.value);document.querySelectorAll('tbody tr').forEach(r=>r.hidden=mode.value!=='all'&&r.dataset.mode!==mode.value)}model.addEventListener('change',update);mode.addEventListener('change',update);console.log(JSON.stringify({models:document.querySelectorAll('section').length,cells:document.querySelectorAll('tbody tr').length,allN25:[...document.querySelectorAll('tbody tr')].every(r=>r.children[2].textContent==='25')}));</script></body></html>'''
    (out/'summary.html').write_text(page)
    return page

if __name__=='__main__':build()
