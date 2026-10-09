"""Two-stage polynomial V2G model; regression target = uncontrolled_peak_MW - optimized_peak_MW (MW). Requires pandas, numpy, scikit-learn, joblib.
Train: python train_cross_region_v2g.py train --input "training set.zip" --output models
Predict: python train_cross_region_v2g.py predict --model models/model.joblib --input new.csv --output predictions.csv
Only load joblib models you trust. Fixed hyperparameters; held-out regions are not used for fitting.
"""
import argparse
import json
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from v2g_features import BinaryAwarePolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score

NUMERIC = ['ev_count', 'participation_rate_requested', 'scenario_hp_ratio', 'ch_refuel',
           'charging_avail_reduction_fraction', 'flexible_window', 'base_peak_MW',
           'base_load_factor', 'hp_share_at_base_peak', 'ev_per_household',
           'ev_power_to_base_peak', 'soc_range_to_base_peak_hours']
CATEGORY = ['charging_avail_reduction_type', 'availability_family', 'soc_min_family']
FEATURES = NUMERIC + CATEGORY


def read_inputs(paths):
    """Read inputs.
    
    Parameters
    ----------
    paths : object
        Paths used by this operation.
    
    Returns
    -------
    object
        Result computed as ``pd.concat(frames, ignore_index=True)``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    frames = []
    def add(handle, name):
        """Add .
        
        Parameters
        ----------
        handle : object
            Handle used by this operation.
        name : object
            Name used by this operation."""
        df = pd.read_csv(handle, sep=None, engine='python', encoding='utf-8-sig')
        df.columns = df.columns.str.lstrip('\ufeff')
        df['source_file'] = name
        frames.append(df)
    for value in paths:
        path = Path(value)
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive:
                for name in sorted(archive.namelist()):
                    if name.lower().endswith('.csv'):
                        with archive.open(name) as handle:
                            add(handle, name)
        else:
            add(path, str(path))
    if not frames:
        raise ValueError('No CSV inputs found.')
    return pd.concat(frames, ignore_index=True)


def check_features(df):
    """Check features.
    
    Parameters
    ----------
    df : object
        Data frame used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return np.isfinite(df[NUMERIC].to_numpy()).all(axis=1) & df[CATEGORY].notna().all(axis=1) & df.flexible_window.isin([0, 1])


def transform(degree):
    """Transform the input data into the feature representation used by the model.
    
    Parameters
    ----------
    degree : object
        Degree used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch."""
    return ColumnTransformer([
        ('numeric', make_pipeline(StandardScaler(), BinaryAwarePolynomialFeatures(degree=degree, binary_indices=(NUMERIC.index('flexible_window'),)), StandardScaler()), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORY)])


def fit_models(df, args):
    """Fit models.
    
    Parameters
    ----------
    df : object
        Data frame used by this operation.
    args : object
        Parsed configuration or command-line arguments.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    if df.success.nunique() != 2:
        raise ValueError('Classifier training requires both success and fail rows.')
    success = df[df.success == 1]
    if len(success) < 2:
        raise ValueError('Too few successful rows for regression.')
    classifier = make_pipeline(transform(args.classifier_degree), LogisticRegression(C=args.classifier_c, max_iter=2000))
    regression = make_pipeline(transform(args.degree), Ridge(alpha=args.alpha))
    classifier.fit(df[FEATURES], df.success)
    regression.fit(success[FEATURES], success.peak_reduction_MW)
    return dict(classifier=classifier, regression=regression, features=FEATURES,
                threshold=args.threshold, degree=args.degree, alpha=args.alpha,
                training_regions=sorted(df.station_name.unique().tolist()))


def predict(bundle, df):
    """Generate predictions using the fitted model.
    
    Parameters
    ----------
    bundle : object
        Bundle used by this operation.
    df : object
        Data frame used by this operation.
    
    Returns
    -------
    object
        Result computed as ``result``."""
    probability = bundle['classifier'].predict_proba(df[FEATURES])[:, 1]
    label = (probability >= bundle['threshold']).astype(int)
    raw = bundle['regression'].predict(df[FEATURES])
    result = df.copy()
    result['predicted_success_probability'] = probability
    result['predicted_success'] = label
    result['predicted_reduction_if_success_MW'] = raw
    result['predicted_peak_reduction_MW'] = np.maximum(0.0, label * raw)
    # Apply the nonnegative constraint to the final output; keep raw regression for diagnostics.
    return result


def train(args):
    """Train .
    
    Parameters
    ----------
    args : object
        Parsed configuration or command-line arguments.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = read_inputs(args.input)
    if {'uncontrolled_peak_MW', 'optimized_peak_MW'}.issubset(df.columns):
        df['peak_reduction_MW'] = pd.to_numeric(df.uncontrolled_peak_MW, errors='coerce') - pd.to_numeric(df.optimized_peak_MW, errors='coerce')
    required = {'station_name', 'seed', 'success', 'peak_reduction_MW'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing training columns: {sorted(required-set(df.columns))}')
    valid = check_features(df)
    if 'feature_status' in df:
        valid &= df.feature_status.eq('complete')
    for col in ['success', 'peak_reduction_MW']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    valid &= df.success.isin([0, 1]) & np.isfinite(df.peak_reduction_MW)
    valid &= df.station_name.notna() & df.seed.notna()
    df.loc[~valid].to_csv(out/'excluded_rows.csv', index=False, encoding='utf-8-sig')
    print(f'Input: {len(df)}; excluded: {int((~valid).sum())}', flush=True)
    df = df.loc[valid].copy()
    if df.station_name.nunique() < 3:
        raise ValueError('This cross-region workflow requires at least three regions.')
    metrics = []; predictions = []
    for i, region in enumerate(sorted(df.station_name.unique())):
        training = df[df.station_name != region]; test = df[df.station_name == region]
        bundle = fit_models(training, args)
        result = predict(bundle, test)
        y = test.peak_reduction_MW; pred = result.predicted_peak_reduction_MW
        mask = test.success == 1
        row = dict(test_region=region, test_rows=len(test),
                   accuracy=accuracy_score(test.success, result.predicted_success),
                   balanced_accuracy=balanced_accuracy_score(test.success, result.predicted_success),
                   success_precision=precision_score(test.success, result.predicted_success, zero_division=0),
                   success_recall=recall_score(test.success, result.predicted_success, zero_division=0),
                   joint_MAE_MW=mean_absolute_error(y, pred),
                   joint_RMSE_MW=mean_squared_error(y, pred)**.5, joint_R2=r2_score(y, pred),
                   actual_success_MAE_MW=mean_absolute_error(y[mask], result.loc[mask, 'predicted_reduction_if_success_MW']) if mask.any() else np.nan)
        metrics.append(row); predictions.append(result)
        joblib.dump(bundle, out/f'fold_{i+1}.joblib')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(metrics).to_csv(out/'region_validation_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(predictions).to_csv(out/'region_validation_predictions.csv', index=False, encoding='utf-8-sig')
    final = fit_models(df, args); joblib.dump(final, out/'model.joblib')
    config = vars(args).copy(); config.update(numeric_features=NUMERIC, categorical_features=CATEGORY,
        training_rows=len(df), regions=sorted(df.station_name.unique().tolist()),
        note='Fixed hyperparameters; choosing parameters from these validation scores makes them development scores, not an independent final test.')
    (out/'training_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved final model: {out / "model.joblib"}')


def main():
    """Run the command-line workflow for this module.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--input', nargs='+', required=True)
    t.add_argument('--output', default='cross_region_models')
    t.add_argument('--degree', type=int, choices=[1, 2, 3], default=3)
    t.add_argument('--alpha', type=float, default=10.)
    t.add_argument('--classifier-degree', type=int, choices=[1, 2, 3], default=2)
    t.add_argument('--classifier-c', type=float, default=1.)
    t.add_argument('--threshold', type=float, default=.5)
    r = sub.add_parser('predict')
    r.add_argument('--model', required=True); r.add_argument('--input', nargs='+', required=True)
    r.add_argument('--output', default='predictions.csv')
    args = parser.parse_args()
    if args.command == 'train':
        if args.alpha <= 0 or args.classifier_c <= 0 or not 0 < args.threshold < 1:
            parser.error('alpha/C must be positive and threshold must lie between 0 and 1.')
        train(args)
    else:
        df = read_inputs(args.input)
        if not check_features(df).all():
            raise ValueError('Prediction inputs contain missing or nonfinite features.')
        bundle = joblib.load(args.model)
        destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
        predict(bundle, df).to_csv(destination, index=False, encoding='utf-8-sig')
        print(f'Saved predictions: {destination}')



# Project workflow and interactive validation plot.
def resolve_inputs(value):
    """Resolve inputs.
    
    Parameters
    ----------
    value : object
        Value used by this operation.
    
    Returns
    -------
    object
        Computed result for the selected input and execution branch.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails.
    FileNotFoundError
        If an explicit validation or processing check in this operation fails."""
    path = Path(value)
    if not path.exists():
        path = next((p for p in [Path(value+'.csv'), Path(value+'.zip')] if p.exists()), path)
    if path.is_dir():
        paths = sorted(path.rglob('*.csv'))
        if not paths: paths = sorted(path.glob('*.zip'))
        if not paths: raise ValueError(f'No CSV/ZIP files in {path}')
        return [str(p) for p in paths]
    if not path.exists(): raise FileNotFoundError(path)
    return [str(path)]


def add_actual_results(df, result_source):
    """Add actual results.
    
    Parameters
    ----------
    df : object
        Data frame used by this operation.
    result_source : object
        Result source used by this operation.
    
    Returns
    -------
    object
        Result computed as ``df.merge(truth, on=keys, how='left', validate='many_to_one')``.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    import re
    records = []
    source = Path(result_source)
    archive = zipfile.ZipFile(source) if source.suffix.lower()=='.zip' else None
    names = archive.namelist() if archive else [str(p) for p in source.rglob('greedy*options*.csv')]
    try:
        for name in names:
            if not name.endswith(('greedy_all_options_summary.csv', 'greedy_failed_options.csv')): continue
            normalized = name.replace('\\','/')
            sid = re.search(r'(hh_[^/]+)/data_\d+_(\d+)/greedy_results_nctrl([\d.]+)/', normalized)
            if not sid: continue
            handle = archive.open(name) if archive else open(name, 'rb')
            with handle: table = pd.read_csv(handle, sep=None, engine='python', encoding='utf-8-sig')
            for _, row in table.iterrows():
                failed = str(row.get('run_status', '')).lower()=='failed' or name.endswith('greedy_failed_options.csv')
                records.append(dict(scenario_id=sid[1], ev_count=int(sid[2]),
                    participation_rate_requested=float(sid[3]), option=int(row.option),
                    actual_success=0 if failed else 1,
                    actual_peak_reduction_MW=0. if failed else float(row.peak_reduction_MW)))
    finally:
        if archive: archive.close()
    if not records: raise ValueError('No matching scenario results found.')
    truth = pd.DataFrame(records)
    keys=['scenario_id','ev_count','participation_rate_requested','option']
    # A contradictory result is an error rather than an arbitrary overwrite.
    for _, group in truth.groupby(keys):
        if group[['actual_success','actual_peak_reduction_MW']].drop_duplicates().shape[0]>1:
            raise ValueError('Conflicting actual results for the same scenario and option.')
    truth=truth.drop_duplicates(keys)
    return df.merge(truth, on=keys, how='left', validate='many_to_one')


def interactive_plot(df, destination):
    """Interactive plot.
    
    Parameters
    ----------
    df : object
        Data frame used by this operation.
    destination : object
        Destination used by this operation."""
    try:
        from plotly.offline import get_plotlyjs
        plotly_source = get_plotlyjs()
    except ImportError:
        plotly_source = '</script><script src="https://cdn.plot.ly/plotly-2.35.2.min.js">'
    filters = [c for c in NUMERIC[:6]+CATEGORY+['station_name','seed','option'] if c in df]
    axes = [c for c in NUMERIC if c in df]
    columns = list(dict.fromkeys(filters+axes+['predicted_peak_reduction_MW','actual_peak_reduction_MW','predicted_success_probability','actual_success']))
    data = json.loads(df.reindex(columns=columns).to_json(orient='records'))
    payload = json.dumps(dict(rows=data,filters=filters,axes=axes),ensure_ascii=False).replace('</','<\\/')
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>V2G predicted and actual reduction</title>
<style>body{font:15px Arial;margin:24px;color:#263348}h1{font-size:23px}.controls{display:flex;flex-wrap:wrap;gap:12px}label{display:flex;flex-direction:column;gap:5px}select{padding:7px;max-width:270px}#chart{height:650px}#status{margin-top:15px}button{padding:8px}p{max-width:1100px}</style>
<script>__PLOTLY__</script></head><body><h1>Peak reduction: predicted and actual</h1>
<p>Select an x-axis. Fix the remaining scenario parameters to one combination. Curves are grouped by region, seed and option when those filters are set to All. Predictions use the final model trained on the training inputs.</p>
<div class="controls"><label>X-axis<select id="axis"></select></label><button id="reset">Reset to a valid combination</button></div>
<div id="filters" class="controls" style="margin-top:18px"></div><div id="status"></div><div id="chart"></div>
<script>const payload=__DATA__,rows=payload.rows; const axis=document.getElementById('axis'), controls={};
const labelMap={participation_rate_requested:'v2g_participation',scenario_hp_ratio:'hp_factor',charging_avail_reduction_fraction:'reduce_factor'};
const name=c=>labelMap[c]||c; const values=c=>[...new Set(rows.map(r=>r[c]).filter(v=>v!==null))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b)));
for(const c of payload.axes){const o=document.createElement('option');o.value=c;o.textContent=name(c);axis.append(o)}
for(const c of payload.filters){const l=document.createElement('label');l.append(name(c));const sel=document.createElement('select');sel.add(new Option('All','*'));for(const v of values(c))sel.add(new Option(String(v),JSON.stringify(v)));l.append(sel);document.getElementById('filters').append(l);controls[c]=sel;sel.onchange=draw}
function reset(){const first=rows[0];for(const c of payload.filters)controls[c].value=c===axis.value?'*':JSON.stringify(first[c]);draw()}
function draw(){const x=axis.value;for(const c of payload.filters)controls[c].disabled=c===x;
const selected=rows.filter(r=>payload.filters.every(c=>c===x||controls[c].value==='*'||JSON.stringify(r[c])===controls[c].value));
const groups={};for(const r of selected){const k=['station_name','seed','option'].map(c=>r[c]??'').join(' / ');(groups[k]??=[]).push(r)}
let traces=[];const colors=['#1f77b4','#e76f51','#2a9d8f','#9b5de5','#b58900'];let i=0;
for(const [key,g]of Object.entries(groups)){g.sort((a,b)=>a[x]-b[x]);const color=colors[i++%colors.length];for(const [col,title,dash,symbol]of [['predicted_peak_reduction_MW','Predicted','dash','circle'],['actual_peak_reduction_MW','Actual','solid','diamond']]){
 const valid=g.filter(r=>r[col]!==null&&r[col]!==undefined);if(!valid.length)continue;
 traces.push({x:valid.map(r=>r[x]),y:valid.map(r=>r[col]),name:title+' · '+key,mode:'lines+markers',line:{color,dash},marker:{symbol,size:8},hovertemplate:name(x)+': %{x}<br>'+title+': %{y:.5f} MW<extra>%{fullData.name}</extra>'})}}
const actual=selected.filter(r=>r.actual_peak_reduction_MW!==null&&r.actual_peak_reduction_MW!==undefined);
const mae=actual.length?actual.reduce((s,r)=>s+Math.abs(r.predicted_peak_reduction_MW-r.actual_peak_reduction_MW),0)/actual.length:null;
document.getElementById('status').textContent=selected.length+' rows; '+actual.length+' with actual results'+(mae===null?'':'; MAE '+mae.toFixed(3)+' MW')+(selected.length?'':' — this combination is absent; choose All or reset.');
Plotly.react('chart',traces,{xaxis:{title:name(x)},yaxis:{title:'Peak reduction (MW)'},hovermode:'closest',legend:{orientation:'h'},margin:{t:25,b:100},annotations:selected.length?[]:[{text:'No data for this combination',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}]},{responsive:true,displaylogo:false});}
axis.onchange=draw;document.getElementById('reset').onclick=reset;reset();</script></body></html>'''
    html=html.replace('__PLOTLY__',plotly_source).replace('__DATA__',payload)
    if Path(destination).name.startswith('training_set'):
        html=html.replace('Predictions use the final model trained on the training inputs.', 'Training-set predictions come from models fitted on the other two regions; each row is held out by region.')
        html=html.replace('<h1>Peak reduction: predicted and actual</h1>', '<h1>Training set: predicted and actual peak reduction</h1>')
    Path(destination).write_text(html,encoding='utf-8')


def project_main():
    """Project main.
    
    Raises
    ------
    ValueError
        If an explicit validation or processing check in this operation fails."""
    parser=argparse.ArgumentParser(description='Train two-stage polynomial V2G models and plot validation predictions.')
    parser.add_argument('--train-input',default='data/traing/traing_set')
    parser.add_argument('--prediction-input',default='data/training/prediction_inputs_complete')
    parser.add_argument('--output',default='data/training')
    parser.add_argument('--actual-results',help='Scenario output ZIP or directory; needed when prediction input has no true reduction.')
    parser.add_argument('--degree',type=int,choices=[1,2,3],default=3)
    parser.add_argument('--alpha',type=float,default=10.)
    parser.add_argument('--classifier-degree',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--classifier-c',type=float,default=1.)
    parser.add_argument('--threshold',type=float,default=.5)
    args=parser.parse_args()
    if args.alpha<=0 or args.classifier_c<=0 or not 0<args.threshold<1: parser.error('Invalid alpha, C or threshold.')
    args.input=resolve_inputs(args.train_input)
    train(args)
    out=Path(args.output)
    training_predictions=pd.read_csv(out/'region_validation_predictions.csv',encoding='utf-8-sig')
    training_predictions['actual_peak_reduction_MW']=training_predictions.peak_reduction_MW
    training_predictions['actual_success']=training_predictions.success
    interactive_plot(training_predictions,out/'training_set_interactive.html')
    validation=read_inputs(resolve_inputs(args.prediction_input))
    if not check_features(validation).all(): raise ValueError('Prediction inputs have missing/nonfinite features.')
    if args.actual_results:
        validation=add_actual_results(validation,args.actual_results)
    else:
        if {'uncontrolled_peak_MW','optimized_peak_MW'}.issubset(validation.columns):
            validation['peak_reduction_MW']=pd.to_numeric(validation.uncontrolled_peak_MW,errors='coerce')-pd.to_numeric(validation.optimized_peak_MW,errors='coerce')
        if 'peak_reduction_MW' in validation:validation['actual_peak_reduction_MW']=validation.peak_reduction_MW
        if 'success' in validation:validation['actual_success']=validation.success
    bundle=joblib.load(out/'model.joblib');result=predict(bundle,validation)
    result.to_csv(out/'prediction_check.csv',index=False,encoding='utf-8-sig')
    interactive_plot(result,out/'prediction_vs_actual.html')
    truth=result.get('actual_peak_reduction_MW',pd.Series(np.nan,index=result.index)).notna()
    print(f'Prediction rows: {len(result)}; matched actual results: {int(truth.sum())}')
    if truth.any():
        metrics={'rows_with_actual':int(truth.sum()),'MAE_MW':mean_absolute_error(result.loc[truth,'actual_peak_reduction_MW'],result.loc[truth,'predicted_peak_reduction_MW'])}
        (out/'prediction_check_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print(metrics)
    else: print('Actual results missing: plot contains predictions only. Supply --actual-results scenarios.zip to compare.')
    print(f'Interactive plot: {out / "prediction_vs_actual.html"}')

if __name__=='__main__':
    project_main()
