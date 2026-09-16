def _true_waterfall(ax, idx, tag, group_label):
    # Preserve the original manuscript's signed-contribution panel and samples.
    sv=np.asarray(shap_values[idx]); order=np.argsort(np.abs(sv))[::-1]
    ys=np.arange(6,0,-1); vals=sv[order]
    ax.barh(ys,vals,height=.54,color=[POS_COL if v>0 else NEG_COL for v in vals],alpha=.92)
    ax.axvline(0,color='#999999',lw=.7,ls='--')
    labels=[]
    for yy,j,v in zip(ys,order,vals):
        sym,_=SYMUNIT[FEAT_LABELS[j]];labels.append(f'{sym} = {_fmt_num(X[idx,j])}')
        inside=abs(v)>.075
        ax.text(v/2 if inside else v+(.008 if v>0 else -.008),yy,f'{v:+.3f}',ha='center' if inside else ('left' if v>0 else 'right'),va='center',color='white' if inside else '#444444',fontsize=11,fontweight='bold')
    ax.set_yticks(ys);ax.set_yticklabels(labels)
    ax.set_ylim(.1,8.2);ax.set_xlim(min(vals.min()-.07,-.01),max(vals.max()+.075,.075))
    ax.set_xlabel(r'Contribution to log$_{10}(V_0)$')
    ax.text(.98,.98,f'$f(x)$ = {y_pred[idx]:.2f}\n$V_0$ ≈ {10**y_pred[idx]:.0f} ×10$^4$ m$^3$',transform=ax.transAxes,ha='right',va='top',fontsize=11,color=NEG_COL if idx==54 else POS_COL)
    ax.text(.02,.02,f'$E[f(x)]$ = {base_values[idx]:.2f}',transform=ax.transAxes,va='bottom',fontsize=10)
    _panel_label(ax,tag)
