import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ShieldCheck, LayoutDashboard, FilePlus2, History, BarChart3, AlertTriangle, CheckCircle2, ChevronRight, Activity, Sparkles } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import './styles.css';

const API = 'http://localhost:8000';
const GROUP_ORDER = ['Claim & Timing','Policyholder','Vehicle','Accident & Claim','Policy & Agent','Other'];

function App(){
  const [page,setPage]=useState('dashboard');
  const [schema,setSchema]=useState(null);
  const [performance,setPerformance]=useState(null);
  const [dashboard,setDashboard]=useState(null);
  const [history,setHistory]=useState([]);
  const [result,setResult]=useState(null);
  const [loading,setLoading]=useState(true);

  const refresh=async()=>{
    setLoading(true);
    const [s,p,d,h]=await Promise.all([fetch(`${API}/api/schema`).then(r=>r.json()),fetch(`${API}/api/performance`).then(r=>r.json()),fetch(`${API}/api/dashboard`).then(r=>r.json()),fetch(`${API}/api/history`).then(r=>r.json())]);
    setSchema(s); setPerformance(p); setDashboard(d); setHistory(h); setLoading(false);
  };
  useEffect(()=>{refresh().catch(e=>{console.error(e);setLoading(false)})},[]);
  const nav=[['dashboard','Dashboard',LayoutDashboard],['predict','Analyze Claim',FilePlus2],['history','Prediction History',History],['models','Model Performance',BarChart3]];
  if(loading) return <div className="boot"><div className="logoMark"><ShieldCheck/></div><h2>FraudShield</h2><p>Loading intelligence engine…</p></div>;
  return <div className="appShell">
    <aside className="sidebar">
      <div className="brand"><div className="logoMark"><ShieldCheck/></div><div><strong>FraudShield</strong><span>Risk Intelligence</span></div></div>
      <div className="navLabel">WORKSPACE</div>
      <nav>{nav.map(([id,label,Icon])=><button key={id} className={page===id?'navItem active':'navItem'} onClick={()=>setPage(id)}><Icon size={18}/><span>{label}</span></button>)}</nav>
      <div className="sideBottom"><div className="statusDot"/> <span>4 models online</span></div>
    </aside>
    <main className="main">
      <header className="topbar"><div><div className="eyebrow">INSURANCE ANALYTICS</div><h1>{page==='dashboard'?'Command Center':page==='predict'?'Claim Analysis':page==='history'?'Prediction History':'Model Benchmark'}</h1></div><div className="modelPill"><Activity size={16}/> Live inference</div></header>
      {page==='dashboard'&&<Dashboard dashboard={dashboard} performance={performance} onAnalyze={()=>setPage('predict')}/>} 
      {page==='predict'&&<Predict schema={schema} onDone={async r=>{setResult(r); await refresh();}} result={result}/>} 
      {page==='history'&&<HistoryPage rows={history}/>} 
      {page==='models'&&<Models performance={performance}/>} 
    </main>
  </div>
}

function Dashboard({dashboard,performance,onAnalyze}){
 const chart=performance?Object.entries(performance.models).map(([model,m])=>({model,Accuracy:m.accuracy*100,Precision:m.precision*100,Recall:m.recall*100,F1:m.f1*100,'ROC-AUC':m.roc_auc*100})):[];
 return <div className="content">
   <section className="hero"><div><div className="heroTag"><Sparkles size={14}/> ML-powered claim triage</div><h2>Detect suspicious claims<br/><span>before they become losses.</span></h2><p>Compare four classification models, score incoming claims, and keep an auditable prediction trail.</p><button className="primary" onClick={onAnalyze}>Analyze a claim <ChevronRight size={18}/></button></div><div className="heroOrb"><ShieldCheck size={86}/><div>4<br/><small>models</small></div></div></section>
   <div className="statsGrid">
    <Stat title="Claims analyzed" value={dashboard.total_claims_analyzed}/><Stat title="Genuine claims" value={dashboard.genuine_claims} tone="good"/><Stat title="Potential fraud" value={dashboard.fraud_claims} tone="bad"/><Stat title="Fraud percentage" value={`${dashboard.fraud_percentage}%`} tone="warn"/>
   </div>
   <section className="panel"><div className="panelHead"><div><h3>Model performance</h3><p>Held-out test set · higher is better</p></div><button className="ghost" onClick={()=>location.hash='models'}>View details</button></div><div className="chartWrap"><ResponsiveContainer width="100%" height={330}><BarChart data={chart} margin={{top:10,right:20,left:0,bottom:10}}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="model"/><YAxis domain={[0,100]} tickFormatter={v=>`${v}%`}/><Tooltip formatter={(v)=>`${Number(v).toFixed(1)}%`}/><Legend/><Bar dataKey="Accuracy"/><Bar dataKey="Precision"/><Bar dataKey="Recall"/><Bar dataKey="F1"/><Bar dataKey="ROC-AUC"/></BarChart></ResponsiveContainer></div></section>
 </div>
}
function Stat({title,value,tone=''}){return <div className={`stat ${tone}`}><span>{title}</span><strong>{value}</strong></div>}

function Predict({schema,onDone,result}){
 const initial=useMemo(()=>Object.fromEntries(schema.fields.map(f=>[f.name,f.default])),[schema]);
 const [form,setForm]=useState(initial); const [model,setModel]=useState('Random Forest'); const [busy,setBusy]=useState(false); const [error,setError]=useState('');
 useEffect(()=>setForm(initial),[initial]);
 const grouped=GROUP_ORDER.map(g=>[g,schema.fields.filter(f=>f.group===g)]).filter(([,fs])=>fs.length);
 const submit=async e=>{e.preventDefault();setBusy(true);setError('');try{const r=await fetch(`${API}/api/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model,claim:form})});const data=await r.json();if(!r.ok)throw new Error(data.detail||'Prediction failed');onDone(data)}catch(e){setError(e.message)}finally{setBusy(false)}};
 return <div className="content twoCol"><div className="panel formPanel"><div className="panelHead"><div><h3>New insurance claim</h3><p>Enter only information present in the source dataset.</p></div></div><form onSubmit={submit}><label className="modelSelect"><span>Select Prediction Model</span><select value={model} onChange={e=>setModel(e.target.value)}>{['Logistic Regression','KNN','Random Forest','SVM'].map(x=><option key={x}>{x}</option>)}</select></label>{grouped.map(([group,fields])=><fieldset key={group}><legend>{group}</legend><div className="fields">{fields.map(f=><Field key={f.name} field={f} value={form[f.name]} onChange={v=>setForm({...form,[f.name]:v})}/>)}</div></fieldset>)}{error&&<div className="error">{error}</div>}<button className="primary full" disabled={busy}>{busy?'Scoring claim…':'Run fraud prediction'} <ChevronRight size={18}/></button></form></div><ResultCard result={result}/></div>
}
function Field({field,value,onChange}){return <label className="field"><span>{field.name.replaceAll('_',' ').replaceAll('-',' ')}</span>{field.type==='select'?<select value={value??''} onChange={e=>onChange(e.target.value)}>{field.options.map(v=><option key={v}>{v}</option>)}</select>:<input type="number" value={value??''} min={field.min??undefined} max={field.max??undefined} onChange={e=>onChange(Number(e.target.value))}/>}</label>}
function ResultCard({result}){if(!result)return <div className="emptyResult"><div className="emptyIcon"><ShieldCheck/></div><h3>Prediction ready when you are</h3><p>Choose a model, complete the claim details, and the selected model will score the claim without retraining.</p></div>;return <div className="resultStack"><div className={`resultHero ${result.prediction?'fraud':'safe'}`}><div><span>Selected model</span><strong>{result.model}</strong></div><div className="resultIcon">{result.prediction?<AlertTriangle/>:<CheckCircle2/>}</div></div><div className="predictionBox"><span>Prediction</span><h2>{result.prediction_label}</h2><div className={`risk ${result.risk_level.toLowerCase()}`}>{result.risk_level} risk</div></div><div className="probGrid"><div><span>Fraud probability</span><strong>{result.fraud_probability}%</strong><div className="meter"><i style={{width:`${result.fraud_probability}%`}}/></div></div><div><span>Genuine probability</span><strong>{result.genuine_probability}%</strong><div className="meter"><i style={{width:`${result.genuine_probability}%`}}/></div></div></div><div className="explain"><strong>How to read this</strong><p>Risk level is Low below 30%, Medium from 30–69.99%, and High at 70% or above. This is a screening aid, not a final claim decision.</p></div></div>}

function HistoryPage({rows}){return <div className="content"><section className="panel"><div className="panelHead"><div><h3>Recent predictions</h3><p>Every submitted claim is stored in SQLite.</p></div></div>{rows.length===0?<div className="emptyTable">No predictions yet. Analyze your first claim to populate the history.</div>:<div className="tableWrap"><table><thead><tr><th>Time</th><th>Model</th><th>Prediction</th><th>Fraud probability</th><th>Risk</th></tr></thead><tbody>{rows.map(r=><tr key={r.id}><td>{new Date(r.timestamp).toLocaleString()}</td><td>{r.model}</td><td><span className={r.prediction==='Potential Fraud'?'badge bad':'badge good'}>{r.prediction}</span></td><td>{r.fraud_probability}%</td><td><span className={`risk ${r.risk_level.toLowerCase()}`}>{r.risk_level}</span></td></tr>)}</tbody></table></div>}</section></div>}
function Models({performance}){const rows=Object.entries(performance.models);return <div className="content"><section className="panel"><div className="panelHead"><div><h3>Four-model benchmark</h3><p>Metrics are calculated once on a stratified held-out test set.</p></div></div><div className="tableWrap"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>{rows.map(([name,m])=><tr key={name}><td><strong>{name}</strong></td><td>{(m.accuracy*100).toFixed(1)}%</td><td>{(m.precision*100).toFixed(1)}%</td><td>{(m.recall*100).toFixed(1)}%</td><td>{(m.f1*100).toFixed(1)}%</td><td>{(m.roc_auc*100).toFixed(1)}%</td></tr>)}</tbody></table></div>{rows.map(([name,m])=><div className="cm" key={name}><div><strong>{name}</strong><span>Confusion matrix · actual × predicted</span></div><div className="matrix"><b>TN {m.confusion_matrix[0][0]}</b><b>FP {m.confusion_matrix[0][1]}</b><b>FN {m.confusion_matrix[1][0]}</b><b>TP {m.confusion_matrix[1][1]}</b></div></div>)}</section></div>}

createRoot(document.getElementById('root')).render(<App/>);
