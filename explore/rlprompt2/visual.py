"""A self-contained view of the report counts, without interpretation."""
import html
import json
from explore.rlprompt2.run import HERE

def build():
    rows=json.loads((HERE/'results/summary.json').read_text())
    ledger=json.loads((HERE/'results/execution.json').read_text())
    sections=[]
    for model in dict.fromkeys(r['model'] for r in rows):
        cells=[]
        for r in rows:
            if r['model']!=model: continue
            counts=[f'{r[k]}/40' for k in ('violation','leak','boxed','rule_mention','grader_mention','tutor_grader_conflict')]
            counts.append(f'{r["conflict_violation"]}/{r["conflict_n"]}' if r['conflict_n'] else '—')
            cells.append('<tr><th scope="row">'+html.escape(r['condition']+' · '+r['label'])+'</th>'+''.join('<td>'+v+'</td>' for v in counts)+'</tr>')
        headers=['Condition','Violation','Leak','Boxed','Rule','Grader','Conflict','V / conflict']
        sections.append('<section><h3>'+html.escape(model)+'</h3><div class="scroll"><table><thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table></div></section>')
    document='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font-family:var(--font-sans,system-ui);color:var(--foreground,#222);margin:0;font-size:12px}section{margin:0 0 24px}h3{font-size:15px;margin:12px 0}p{line-height:1.5;color:var(--muted-foreground,#666)}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:10px 5px;text-align:right;border-bottom:1px solid var(--border,#ddd);white-space:nowrap}thead th{font-size:10px;color:var(--muted-foreground,#666);font-weight:600}th:first-child{text-align:left;white-space:normal;min-width:135px;font-weight:500}td:last-child{font-weight:650}caption{text-align:left}
</style></head><body>'''+''.join(sections)+f'<p>Thinking on · 40 problems per condition · 1 sample · Conflict = requested regex on reasoning.<br>V / conflict = violations among conflict-coded replies. Empty and truncated finals remain included.<br>Conservative spend ${ledger["conservative_usd"]:.4f} / $1.50 · Owned rental destroyed.</p></body></html>'
    (HERE/'results/summary.html').write_text(document)

if __name__=='__main__':build()
