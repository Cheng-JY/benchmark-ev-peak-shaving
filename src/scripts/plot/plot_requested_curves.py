r"""Three scenario curves from held-out predictions. Plot exclusions never change training data.
    python src\scripts\plot_requested_curves.py^
  --input "data/training/region_validation_predictions.csv" ^
  --output "data/training/requested_curves" ^
  --station "ST. Feldstr. 630KVATR" ^
  --ev-count 97 ^
  --seed 13

"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def subset(df,settings):
    """Subset .
    
    Parameters
    ----------
    df : object
        Data frame used by this operation.
    settings : object
        Settings used by this operation.
    
    Returns
    -------
    object
        Result computed as ``result``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    mask=pd.Series(True,index=df.index)
    for key,value in settings.items():
        if isinstance(value,(int,float)):mask &= np.isclose(pd.to_numeric(df[key],errors='coerce'),value)
        else:mask &= df[key].eq(value)
    result=df.loc[mask].copy()
    if result.empty:raise ValueError(f'No records for {settings}')
    return result


def draw(data,x,label,destination,title,seed_groups=False):
    """Draw .
    
    Parameters
    ----------
    data : object
        Input data used by this operation.
    x : object
        X used by this operation.
    label : object
        Label used by this operation.
    destination : object
        Destination used by this operation.
    title : object
        Title used by this operation.
    seed_groups : object
        Seed groups used by this operation."""
    fig,ax=plt.subplots(figsize=(7.5,5.3))
    groups=list(data.groupby('seed')) if seed_groups else [(None,data)]
    for index,(seed,g) in enumerate(groups):
        g=g.sort_values(x,kind='stable')
        suffix=f' (seed {seed})' if seed_groups else ''
        # Keep all existing duplicate-x records except the explicitly removed point in figure 2.
        if seed_groups:
            color=plt.get_cmap('tab10')(index)
            actual_color=pred_color=color
        else:actual_color,pred_color='#2878b5','#e07b39'
        ax.plot(g[x],g.actual_kW,'-o',label='Simulation'+suffix,color=actual_color,lw=1.8,markersize=5)
        ax.plot(g[x],g.predicted_kW,'--^',label='Final prediction'+suffix,color=pred_color,lw=1.8,markersize=5)
    ax.set(xlabel=label,ylabel='Peak-load reduction (kW)',title=title)
    ax.grid(alpha=.25);ax.legend(fontsize=9);fig.tight_layout()
    fig.savefig(destination.with_suffix('.png'),dpi=220);fig.savefig(destination.with_suffix('.pdf'));plt.close(fig)


def main():
    """Run the command-line workflow for this module.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/training/rural_seed_validation/rural_seed_validation_predictions.csv')
    parser.add_argument('--output',default='data/training/requested_curves')
    parser.add_argument('--station',default='ST. Feldstr. 630KVATR')
    parser.add_argument('--ev-count',type=int,default=97)
    parser.add_argument('--seed',type=int,default=13)
    parser.add_argument('--first-all-seeds',action='store_true',help='Figure 1: show separate seed curves within the chosen station.')
    parser.add_argument('--keep-lower-04',action='store_true',help='Figure 2: retain both records at ch_refuel=0.4.')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input,encoding='utf-8-sig')
    for c in ['success','predicted_success','peak_reduction_MW','predicted_reduction_if_success_MW']:
        df[c]=pd.to_numeric(df[c],errors='raise')
        if not np.isfinite(df[c]).all():raise ValueError(f'Nonfinite values in {c}')
    if not df.success.isin([0,1]).all() or not df.predicted_success.isin([0,1]).all():raise ValueError('Labels must be binary')
    df['actual_kW']=np.where(df.success==1,df.peak_reduction_MW*1000,0.)
    df['predicted_kW']=df.predicted_success*df.predicted_reduction_if_success_MW*1000
    common=dict(station_name=args.station,ev_count=args.ev_count,scenario_hp_ratio=1.,household_ratio=1.,flexible_window=1,charging_avail_reduction_type='flat')
    f1=dict(common,charging_avail_reduction_fraction=1.,availability_family='ch_avail_cp',soc_min_family='soc_min_cp',option=1)
    if not args.first_all_seeds:f1['seed']=args.seed
    first=subset(df,f1)
    f2=dict(common,seed=args.seed,participation_rate_requested=1.,charging_avail_reduction_fraction=1.,availability_family='ch_avail_refuel_cp',soc_min_family='soc_min_refuel_cp',option=2)
    second=subset(df,f2);removed=second.iloc[:0].copy()
    if not args.keep_lower_04:
        duplicate=second.loc[np.isclose(second.ch_refuel,.4)]
        if len(duplicate)>1:
            index=duplicate.actual_kW.idxmin()
            removed=second.loc[[index]].copy();second=second.drop(index)
    removed.to_csv(out/'figure_2_omitted_records.csv',index=False,encoding='utf-8-sig')
    f3=dict(common,seed=args.seed,participation_rate_requested=1.,ch_refuel=1.,availability_family='ch_avail_refuel_cp',soc_min_family='soc_min_refuel_cp')
    third=subset(df,f3)
    if third.charging_avail_reduction_fraction.duplicated().any():
        raise ValueError('Figure 3 has multiple records per reduction value. Specify more filters before connecting them.')
    title=args.station.replace('ST. ','').replace(' 630KVATR','')
    draw(first,'participation_rate_requested','V2G participation rate',out/'figure_1_participation',title,args.first_all_seeds)
    draw(second,'ch_refuel','Plug-in motivation (ch_refuel)',out/'figure_2_ch_refuel',title)
    draw(third,'charging_avail_reduction_fraction','Availability reduction fraction (stored value)',out/'figure_3_reduction',title)
    for i,g in enumerate([first,second,third],1):g.to_csv(out/f'figure_{i}_records.csv',index=False,encoding='utf-8-sig')
    (out/'settings.json').write_text(json.dumps(dict(figure_1=f1,figure_2=f2,figure_3=f3,figure_3_options=third.sort_values('charging_avail_reduction_fraction').option.tolist(),figure_2_display_exclusions=len(removed)),indent=2),encoding='utf-8')
    print('Rows plotted:',len(first),len(second),len(third),'Display exclusions:',len(removed))

if __name__=='__main__':main()
