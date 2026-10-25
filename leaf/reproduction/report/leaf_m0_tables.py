"""Tables of the earlier variant LEAF-M0 (its own campaign, results/leaf_m0) from the collected campaign data
(results/leaf_m0_summaries/paper_final_data.json) and its per-backbone reports: main_dr.tex (LEAF-M0 vs DiffRate),
augreg.tex, batch.tex, grid_<bb>.tex. Ported from the earlier paper tools; the computations are unchanged."""
import json,re,sys,os


def main(data_path, reports_dir, out_dir):
    d=json.load(open(data_path));rep=reports_dir;out=out_dir;os.makedirs(out,exist_ok=True)
    NAME={'deit_t':'DeiT-T','deit_s':'DeiT-S','deit_b':'DeiT-B','augreg_s':'AugReg ViT-S','augreg_b':'AugReg ViT-B','mae_b':'MAE ViT-B','mae_l':'MAE ViT-L'}
    ROW=re.compile(r'^\| (val|v2) \| (latency(?:_gmac)?/\S+) \| (\S+) \| tolerant \| ([+-][\d.]+) \| ([\d.e+-]+) \| [\d.]+ \| (True|False) \| ([^|]+) \|')
    pairs={}
    for bb in d:
        for line in open(f'{rep}/paper-final-{bb}-2026-10-14.md'):
            m=ROW.match(line)
            if m:
                s,a,b,dl,h,dom,fr=m.groups()
                pairs.setdefault(bb,{})[(s,a,b)]=dict(delta=float(dl),holm=float(h),dom=dom=='True',frontier=fr.strip())
    json.dump({bb:{'|'.join(k):v for k,v in p.items()} for bb,p in pairs.items()},open(os.path.join(out,'pairs_parsed.json'),'w'),indent=0)
    def drpair(bb,arm,split):
        for (s,a,b),v in pairs[bb].items():
            if s==split and a==arm and b.startswith('DR'):return b,v
        return None,None
    def mark(bb,arm,split):
        b,v=drpair(bb,arm,split)
        if not v:return ''
        if v['dom']:return '$^{\\dagger}$'
        if v['delta']<0 and v['holm']<.05:return '$^{\\ddagger}$'
        return ''
    f2=lambda x:'--' if x is None else f'{x:.2f}'
    def lat(r,b='32'):
        l=r['lat'].get(b);return '--' if not l else f'{l[0]:.2f}'
    # main table: backbones with DR
    L=[]
    L.append('\\begin{table}[!htbp]\\centering\\small\\setlength{\\tabcolsep}{3.2pt}')
    L.append('\\caption{Ours versus DiffRate~\\cite{chen2023diffrate} at every released DiffRate operating point. All rows are re-run on the same frozen backbone, RTX~5060~Ti GPU and BF16 inference. ``Lat.\'\' plans under the measured-latency budget alone; ``Lat.+G\'\' adds the DiffRate backbone GMAC as a second constraint. Latency is bs32 milliseconds per batch, averaged over three interleaved rounds. $^{\\dagger}$: dominates the latency-paired DiffRate point (higher top-1, Holm-adjusted McNemar $p<0.05$, faster by at least $\\tau$ in all three rounds). $^{\\ddagger}$: significantly less accurate than the paired point. Repeated schedules (identical to another row) are marked ``=\'\' and were evaluated once.}')
    L.append('\\label{tab:main-dr}')
    L.append('\\begin{tabular}{ll rrrr rrrrr rrrrr}\\toprule')
    L.append('& & \\multicolumn{4}{c}{DiffRate (re-run)} & \\multicolumn{5}{c}{Ours, Lat.} & \\multicolumn{5}{c}{Ours, Lat.+G}\\\\')
    L.append('\\cmidrule(lr){3-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-16}')
    L.append('Model & Anchor & GMAC & ms & Val & V2 & $K$ & GMAC & ms & Val & V2 & $K$ & GMAC & ms & Val & V2\\\\\\midrule')
    for bb in ('deit_t','deit_s','deit_b','mae_b','mae_l'):
        rows=d[bb];anchors=[n for n in rows if n.startswith('DR')]
        de=rows['dense']
        L.append(f"\\multicolumn{{2}}{{l}}{{{NAME[bb]} (dense)}} & {de['gmac']:.2f} & {lat(de)} & {de['top1']['val']:.2f} & {de['top1']['v2']:.2f} &&&&&&&&&&\\\\")
        for a in anchors:
            r=rows[a];cells=[a[2:],f"{r['gmac']:.2f}",lat(r),f"{r['top1']['val']:.2f}",f"{r['top1']['v2']:.2f}"]
            for meth in ('latency','latency_gmac'):
                n=f'{meth}/{a}';o=rows.get(n)
                if o is None:cells+=['']*5;continue
                K=len(o['state']['sites'])
                if o['top1']['val'] is None:
                    cells+=[str(K),f"{o['gmac']:.2f}",lat(o),'=','=' ];continue
                cells+=[str(K),f"{o['gmac']:.2f}",lat(o),f"{o['top1']['val']:.2f}{mark(bb,n,'val')}",f"{o['top1']['v2']:.2f}{mark(bb,n,'v2')}"]
            L.append('& '+' & '.join(cells)+'\\\\')
        for e in ('extra1','extra2'):
            n=f'latency/{e}';o=rows.get(n)
            if o:L.append(f"& {e} & & & & & {len(o['state']['sites'])} & {o['gmac']:.2f} & {lat(o)} & {o['top1']['val']:.2f} & {o['top1']['v2']:.2f} &&&&&\\\\")
        L.append('\\midrule' if bb!='mae_l' else '\\bottomrule')
    L.append('\\end{tabular}\\end{table}')
    open(f'{out}/main_dr.tex','w').write('\n'.join(L)+'\n')
    # augreg table
    L=['\\begin{table}[!htbp]\\centering\\small','\\caption{AugReg backbones (no released DiffRate schedule). Budgets are fractions of measured dense bs32 latency. $\\Delta$ToMe and $\\Delta$PPT are validation top-1 differences to the respective measured frontier interpolated at our latency (``--\'\': outside the measured frontier; no extrapolation).}','\\label{tab:augreg}','\\begin{tabular}{llrrrrrrr}\\toprule','Model & Budget & $K$ & GMAC & ms & Val & V2 & $\\Delta$ToMe & $\\Delta$PPT\\\\\\midrule']
    for bb in ('augreg_s','augreg_b'):
        rows=d[bb];de=rows['dense']
        L.append(f"{NAME[bb]} & dense & -- & {de['gmac']:.2f} & {lat(de)} & {de['top1']['val']:.2f} & {de['top1']['v2']:.2f} & & \\\\")
        for n,o in rows.items():
            if not n.startswith('latency/'):continue
            fr={}
            for fam in ('tome','ppt'):
                fr[fam]='--'
                for (s,a,b),v in pairs[bb].items():
                    if s=='val' and a==n and b.startswith(fam+'/'):
                        x=v['frontier'];fr[fam]=f'{float(x):+.2f}' if x.startswith(('+','-')) else '--'
            L.append(f"& {n.split('/')[1].replace('dense','')}$\\times$ & {len(o['state']['sites'])} & {o['gmac']:.2f} & {lat(o)} & {o['top1']['val']:.2f} & {o['top1']['v2']:.2f} & {fr['tome']} & {fr['ppt']}\\\\")
        L.append('\\midrule' if bb=='augreg_s' else '\\bottomrule')
    L.append('\\end{tabular}\\end{table}')
    open(f'{out}/augreg.tex','w').write('\n'.join(L)+'\n')
    # batch size table: speedup vs paired DR anchor at same anchor, ours latency arm
    L=['\\begin{table}[!htbp]\\centering\\small','\\caption{Latency (ms per batch) across batch sizes for the latency-only schedules and the DiffRate anchor they were budgeted from. Schedules are planned for bs32 only and reused unchanged at other batch sizes.}','\\label{tab:batch}','\\begin{tabular}{llrrrrrrrr}\\toprule','& & \\multicolumn{2}{c}{bs1} & \\multicolumn{2}{c}{bs32} & \\multicolumn{2}{c}{bs128} & \\multicolumn{2}{c}{bs512}\\\\','\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}','Model & Anchor & DR & Ours & DR & Ours & DR & Ours & DR & Ours\\\\\\midrule']
    for bb in ('deit_t','deit_s','deit_b','mae_b','mae_l'):
        rows=d[bb];de=rows['dense']
        L.append(f"{NAME[bb]} & dense & \\multicolumn{{2}}{{c}}{{{lat(de,'1')}}} & \\multicolumn{{2}}{{c}}{{{lat(de,'32')}}} & \\multicolumn{{2}}{{c}}{{{lat(de,'128')}}} & \\multicolumn{{2}}{{c}}{{{lat(de,'512')}}}\\\\")
        for a in [n for n in rows if n.startswith('DR')]:
            o=rows.get('latency/'+a)
            if not o:continue
            L.append(f"& {a[2:]} & "+' & '.join(f"{lat(rows[a],b)} & {lat(o,b)}" for b in ('1','32','128','512'))+'\\\\')
        L.append('\\midrule' if bb!='mae_l' else '\\bottomrule')
    L.append('\\end{tabular}\\end{table}')
    open(f'{out}/batch.tex','w').write('\n'.join(L)+'\n')
    # full grids appendix
    for bb,rows in d.items():
        L=['\\begin{longtable}{lrrrrrrr}',f'\\caption{{Complete fixed grid for {NAME[bb]}. GMAC is the analytic backbone count and is given only for schedules expressed as prune/merge token counts (dense, DiffRate, ours).}}\\label{{tab:grid-{bb}}}\\\\','\\toprule','Configuration & GMAC & bs1 & bs32 & bs128 & bs512 & Val & V2\\\\\\midrule\\endhead']
        for n,r in rows.items():
            g=f"{r['gmac']:.2f}" if r.get('gmac') else '--'
            v=r['top1']['val'];v2=r['top1']['v2']
            L.append(f"{n.replace('_','\\_')} & {g} & "+' & '.join(lat(r,b) for b in ('1','32','128','512'))+f" & {'=' if v is None else f'{v:.2f}'} & {'=' if v2 is None else f'{v2:.2f}'}\\\\")
        L.append('\\bottomrule\\end{longtable}')
        open(f'{out}/grid_{bb}.tex','w').write('\n'.join(L)+'\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])
