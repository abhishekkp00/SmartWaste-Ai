import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import axios from "axios";
import "./style.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const CLASSES = ["cardboard","glass","metal","paper","plastic","trash"];
const LABELS = {cardboard:"Cardboard",glass:"Glass",metal:"Metal",paper:"Paper",plastic:"Plastic",trash:"Trash"};
const COLORS = ["#d6a56b","#8da7c7","#a7b5c4","#83b6a1","#6aa9c5","#9a8d87"];

function App(){
 const [health,setHealth]=useState(null),[analytics,setAnalytics]=useState(null),[file,setFile]=useState(null),[preview,setPreview]=useState(""),[result,setResult]=useState(null),[busy,setBusy]=useState(false),[error,setError]=useState(""),[apiOnline,setApiOnline]=useState(false),[model,setModel]=useState("resnet18_stage_b");
 const modelReady=(health?.available_models||[]).includes(model);
 const refresh=async()=>{try{const [h,a]=await Promise.all([axios.get(API+"/api/health"),axios.get(API+"/api/analytics?limit=8")]);setHealth(h.data);setAnalytics(a.data);setApiOnline(true);const ready=h.data.available_models||[];if(ready.length&&!ready.includes(model))setModel(ready[0])}catch(e){setApiOnline(false);setError("Backend is not reachable. Start FastAPI on port 8000 to enable live classification and analytics.")}};
 useEffect(()=>{refresh();const id=setInterval(refresh,10000);return()=>clearInterval(id)},[]);
 const counts=analytics?.summary?.class_distribution||{};
 const chart=CLASSES.map(k=>({name:LABELS[k],count:counts[k]||0,key:k}));
 const stats=analytics?.summary||{};
 const choose=(f)=>{setFile(f);setResult(null);setError("");if(preview)URL.revokeObjectURL(preview);setPreview(f?URL.createObjectURL(f):"")};
 const predict=async()=>{if(!file)return;setBusy(true);setError("");try{const body=new FormData();body.append("file",file);body.append("model_name",model);body.append("top_k","3");const r=await axios.post(API+"/api/predict",body,{headers:{"Content-Type":"multipart/form-data"}});setResult(r.data);refresh()}catch(e){setError(e.response?.data?.detail||"Prediction failed. Check backend logs and trained checkpoint availability.")}finally{setBusy(false)}};
 const recent=analytics?.recent_predictions||[];
 return <div className="shell">
  <aside className="sidebar"><div className="brand"><div className="logo">S</div><div><b>SmartWaste</b><small>AI • WASTE INTELLIGENCE</small></div></div><div className="navlabel">WORKSPACE</div><a className="active">◫ &nbsp; Overview</a><a href="#classify">◎ &nbsp; Classify waste</a><a href="#analytics">▥ &nbsp; Analytics</a><a href="#history">◷ &nbsp; History</a><div className="sidebottom"><div className="avatar">SW</div><div><b>SmartWaste AI</b><small>Computer vision platform</small></div></div></aside>
  <main className="main"><header><div><div className="eyebrow">WASTE INTELLIGENCE PLATFORM</div><h1>Overview</h1><p>Monitor waste classification and turn images into actionable recycling guidance.</p></div><div className="status"><i className={apiOnline?"online":"offline"}></i>{apiOnline?(modelReady?"API + MODEL READY":"API ONLINE • CHECKPOINT MISSING"):"API OFFLINE"}</div></header>
  <section className="hero"><div><div className="eyebrow light">SMARTER SEGREGATION STARTS HERE</div><h2>Make every piece<br/>of waste count.</h2><p>AI-powered visual classification for a cleaner, more circular future.</p><a className="heroBtn" href="#classify">Classify an image <span>↗</span></a></div><div className="heroArt"><div className="orbit orbit1"></div><div className="orbit orbit2"></div><div className="leaf">♻</div><div className="heroTag">VISION • SORT • RECYCLE</div></div></section>
  <section className="cards">
   <Metric label="Images analyzed" value={stats.total_predictions??"—"} note="Recorded predictions" icon="▧"/>
   <Metric label="Average confidence" value={stats.total_predictions?((stats.average_confidence||0)*100).toFixed(1)+"%":"—"} note="Across recorded predictions" icon="◉"/>
   <Metric label="Needs review" value={stats.low_confidence_count??"—"} note="Below confidence threshold" icon="△"/>
   <Metric label="Waste categories" value="06" note="TrashNet classification labels" icon="▦"/>
  </section>
  <section className="grid two" id="classify">
   <div className="panel classify"><div className="panelHead"><div><div className="eyebrow">IMAGE ANALYSIS</div><h3>Classify your waste</h3></div><span className="pill">PyTorch inference</span></div>
    <label className="drop" htmlFor="file"><input id="file" type="file" accept="image/*" onChange={e=>choose(e.target.files?.[0])}/>{preview?<img src={preview}/>:<><div className="uploadIcon">↑</div><b>Drop an image here or browse</b><span>JPG, PNG or WEBP · Max 10 MB</span></>}</label>
    <div className="formrow"><label>Model<select value={model} onChange={e=>setModel(e.target.value)}><option value="resnet18_stage_b">ResNet18 fine-tuned</option><option value="custom_cnn">Custom CNN</option></select></label><button onClick={predict} disabled={!file||busy||!apiOnline||!modelReady}>{busy?"Analyzing…":"Analyze image →"}</button></div>
    {apiOnline&&!modelReady&&<div className="error">Backend connected, but no trained checkpoint is available for this model. Add a trained .pth file to the models/ directory to enable genuine predictions.</div>}{error&&<div className="error">{error}</div>}
    {result&&<div className="result"><div className="eyebrow">PREDICTION RESULT</div><div className="prediction"><b>{result.display_name}</b><strong>{(result.confidence*100).toFixed(1)}%</strong></div><p>{result.warning||"Prediction confidence is above the configured threshold."}</p><div className="probabilities">{result.top_k?.map(x=><div key={x.class_name}><span>{x.display_name}</span><b>{(x.confidence*100).toFixed(1)}%</b><div className="track"><i style={{width:(x.confidence*100)+"%"}}/></div></div>)}</div><div className="recommend"><b>{result.recommendation?.disposal_bin}</b><p>{result.recommendation?.prep}</p><small>{result.recommendation?.notes}</small></div></div>}
   </div>
   <div className="panel" id="analytics"><div className="panelHead"><div><div className="eyebrow">CLASS DISTRIBUTION</div><h3>Classification activity</h3></div><span className="muted">From saved predictions</span></div>{analytics&&stats.total_predictions>0?<ResponsiveContainer width="100%" height={280}><BarChart data={chart}><XAxis dataKey="name" tick={{fontSize:11}}/><YAxis allowDecimals={false}/><Tooltip/><Bar dataKey="count" radius={[6,6,0,0]}>{chart.map((x,i)=><Cell key={x.key} fill={COLORS[i]}/>)}</Bar></BarChart>:<div className="emptyChart"><div>▥</div><b>No classification history yet</b><span>Analyze an image to start building insights.</span></div>}<div className="legend">{chart.map((x,i)=><span key={x.key}><i style={{background:COLORS[i]}}/>{x.name}</span>)}</div></div>
  </section>
  <section className="panel history" id="history"><div className="panelHead"><div><div className="eyebrow">RECENT ACTIVITY</div><h3>Prediction history</h3></div><button className="refresh" onClick={refresh}>↻ Refresh</button></div>{recent.length?<div className="tablewrap"><table><thead><tr><th>IMAGE</th><th>PREDICTED CLASS</th><th>MODEL</th><th>CONFIDENCE</th><th>STATUS</th></tr></thead><tbody>{recent.map(x=><tr key={x.id}><td>{x.image_name}</td><td><b>{LABELS[x.class_name]||x.class_name}</b></td><td>{x.model_name}</td><td>{(x.confidence*100).toFixed(1)}%</td><td><span className={x.is_confident?"good":"warn"}>{x.is_confident?"Confident":"Review"}</span></td></tr>)}</tbody></table></div>:<div className="emptyHistory">Your analyzed images will appear here.</div>}</section>
  <footer><span>SMARTWASTE AI</span><span>Built with PyTorch · FastAPI · React</span></footer>
  </main></div>
}
function Metric({label,value,note,icon}){return <div className="metric"><div className="metricTop"><span>{label}</span><i>{icon}</i></div><strong>{value}</strong><small>{note}</small></div>}
createRoot(document.getElementById("root")).render(<App/>);
