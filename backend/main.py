from pathlib import Path
import io, os, re
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DEMO = ROOT / "data" / "demo" / "retail_demo.csv"

app = FastAPI(title="InsightForge AI", version="1.0.0", description="AI-powered decision intelligence for business data")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

state: dict[str, Any] = {"df": None, "filename": None, "docs": [], "doc_chunks": []}


def safe(v: Any):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v) if np.isfinite(v) else None
    if isinstance(v, (pd.Timestamp,)): return v.isoformat()
    if pd.isna(v): return None
    return v


def num_cols(df): return list(df.select_dtypes(include=np.number).columns)


def find_col(df, keywords, numeric_only=False):
    cols = num_cols(df) if numeric_only else list(df.columns)
    for c in cols:
        n = str(c).lower()
        if any(k in n for k in keywords): return c
    return None


def revenue_col(df): return find_col(df, ["revenue","sales","amount","price","value","income","spend","total"], True)

def customer_col(df): return find_col(df, ["customer","client","user","account","member"])

def date_col(df):
    for c in df.columns:
        n = str(c).lower()
        if any(k in n for k in ["date","time","timestamp","month","year"]):
            p = pd.to_datetime(df[c], errors="coerce")
            if p.notna().sum() >= max(3, int(len(df)*.5)): return c
    return None


def group_col(df):
    cats = list(df.select_dtypes(exclude=np.number).columns)
    preferred = [c for c in cats if not any(k in str(c).lower() for k in ["id","code","uuid","email","name"])]
    return preferred[0] if preferred else (cats[0] if cats else None)


def profile(df):
    missing = int(df.isna().sum().sum()); cells=max(1, df.shape[0]*df.shape[1]); dup=int(df.duplicated().sum())
    quality=max(0,min(100,100-missing/cells*100-dup/max(1,len(df))*100))
    return {"rows":int(len(df)),"columns":int(df.shape[1]),"numeric_columns":len(num_cols(df)),"categorical_columns":int(df.shape[1]-len(num_cols(df))),"missing_values":missing,"duplicate_rows":dup,"quality":round(quality,1),"columns_detail":[{"name":str(c),"dtype":str(df[c].dtype),"missing":int(df[c].isna().sum()),"unique":int(df[c].nunique(dropna=True))} for c in df.columns]}


def anomaly_engine(df):
    nc=num_cols(df)
    if len(nc)==0 or len(df)<5: return []
    X=df[nc].replace([np.inf,-np.inf],np.nan); X=X.fillna(X.median(numeric_only=True)).fillna(0)
    model=IsolationForest(n_estimators=250, contamination="auto", random_state=42).fit(X)
    labels=model.predict(X); scores=-model.score_samples(X)
    means=X.mean(); std=X.std().replace(0,1)
    out=[]
    for i in np.argsort(scores)[::-1][:15]:
        if labels[i] != -1: continue
        z=((X.iloc[i]-means)/std).abs().sort_values(ascending=False)
        reasons=[f"{c} is {z[c]:.1f}σ from the dataset center" for c in z.index[:3] if z[c]>=1.5]
        out.append({"row":int(i)+1,"score":round(float(scores[i]),3),"severity":"High" if scores[i]>=np.percentile(scores,90) else "Review","reasons":reasons,"values":{str(c):safe(df.iloc[i][c]) for c in nc[:8]}})
    return out


def chart(df):
    rc=revenue_col(df); dc=date_col(df)
    if rc:
        if dc:
            t=pd.DataFrame({"d":pd.to_datetime(df[dc],errors="coerce"),"v":pd.to_numeric(df[rc],errors="coerce")}).dropna()
            if len(t)>=3:
                g=t.groupby(t.d.dt.to_period("M")).v.sum().tail(18)
                return {"title":f"{rc} trend","description":f"Monthly {rc} from the uploaded data","labels":[str(x) for x in g.index],"values":[round(float(x),2) for x in g.values],"kind":"line","format":"currency"}
        gc=group_col(df)
        if gc:
            g=df.assign(_v=pd.to_numeric(df[rc],errors="coerce")).groupby(gc)._v.sum().sort_values(ascending=False).head(12)
            return {"title":f"{rc} by {gc}","description":"Top groups ranked by business value","labels":[str(x) for x in g.index],"values":[round(float(x),2) for x in g.values],"kind":"bar","format":"currency"}
    nc=num_cols(df)
    if nc:
        c=nc[0]; vals=pd.to_numeric(df[c],errors="coerce").fillna(0).head(20)
        return {"title":f"{c} by record","description":"First 20 records","labels":[str(i+1) for i in range(len(vals))],"values":[float(x) for x in vals],"kind":"line","format":"number"}
    return {"title":"Dataset overview","description":"No numeric field is available","labels":[],"values":[],"kind":"bar","format":"number"}


def insights(df, anomalies):
    out=[]; rc=revenue_col(df); nc=num_cols(df)
    if rc:
        s=pd.to_numeric(df[rc],errors="coerce").dropna()
        if len(s):
            out.append({"severity":"growth","title":f"{rc} totals {s.sum():,.0f}","detail":f"Based on {len(s):,} usable records; median is {s.median():,.0f}."})
            if s.median() and s.max()/s.median()>3: out.append({"severity":"attention","title":f"{rc} has a high-value outlier","detail":f"The largest value is {s.max()/s.median():.1f}× the median; review it before forecasting."})
    if nc:
        cvs={c:abs(pd.to_numeric(df[c],errors="coerce").std()/pd.to_numeric(df[c],errors="coerce").mean()) if pd.to_numeric(df[c],errors="coerce").mean()!=0 else 0 for c in nc}
        c=max(cvs,key=cvs.get)
        if cvs[c]>0.8: out.append({"severity":"attention","title":f"{c} is highly variable","detail":f"Coefficient of variation is {cvs[c]:.2f}; segmentation or anomaly review may reveal why."})
    gc=group_col(df)
    if gc:
        top=df[gc].value_counts(dropna=True).head(1)
        if not top.empty: out.append({"severity":"opportunity","title":f"{top.index[0]} is the largest {gc} group","detail":f"It represents {top.iloc[0]/len(df)*100:.1f}% of rows."})
    if anomalies: out.append({"severity":"critical","title":f"{len(anomalies)} unusual record(s) need review","detail":"Isolation Forest found observations that differ materially from the numeric profile."})
    return out[:5]


def segment(df):
    nc=num_cols(df)
    if len(df)<6 or len(nc)<2: return {"available":False,"reason":"Need at least 6 rows and 2 numeric fields for meaningful clustering."}
    cols=nc[:6]; X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(df[cols].median(numeric_only=True)).fillna(0)
    Xs=StandardScaler().fit_transform(X)
    k=min(4,max(2,int(round(np.sqrt(len(df)/2)))))
    k=min(k,len(df))
    model=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xs)
    counts=pd.Series(model.labels_).value_counts().sort_index()
    summaries=[]
    for lab in range(k):
        idx=model.labels_==lab; row=df.loc[idx,cols].mean(numeric_only=True)
        summaries.append({"segment":lab+1,"records":int(idx.sum()),"share":round(float(idx.sum()/len(df)*100),1),"profile":{str(c):round(float(row[c]),2) for c in cols}})
    return {"available":True,"k":k,"features":[str(c) for c in cols],"segments":summaries}


def forecast(df):
    rc=revenue_col(df); dc=date_col(df)
    if not rc or not dc: return {"available":False,"reason":"Forecasting requires a date/time field and a numeric business metric such as revenue or sales."}
    t=pd.DataFrame({"d":pd.to_datetime(df[dc],errors="coerce"),"v":pd.to_numeric(df[rc],errors="coerce")}).dropna()
    if len(t)<8: return {"available":False,"reason":"At least 8 dated observations are recommended for a stable forecast."}
    g=t.groupby(t.d.dt.to_period("M")).v.sum().sort_index()
    if len(g)<4: return {"available":False,"reason":"At least 4 distinct months are needed for the forecast studio."}
    y=g.values.astype(float); x=np.arange(len(y)).reshape(-1,1); model=LinearRegression().fit(x,y)
    horizon=min(6,max(3,round(len(y)*.25))); fx=np.arange(len(y),len(y)+horizon).reshape(-1,1); pred=model.predict(fx)
    resid=y-model.predict(x); sd=float(np.std(resid)) if len(resid)>1 else 0
    labels=[str(p) for p in g.index]+[str(g.index[-1]+i) for i in range(1,horizon+1)]
    return {"available":True,"metric":str(rc),"historical_labels":[str(p) for p in g.index],"historical_values":[round(float(v),2) for v in y],"forecast_labels":[str(g.index[-1]+i) for i in range(1,horizon+1)],"forecast_values":[round(float(v),2) for v in pred],"lower":[round(float(v-1.96*sd),2) for v in pred],"upper":[round(float(v+1.96*sd),2) for v in pred],"next_value":round(float(pred[0]),2),"model":"Linear trend baseline","note":"Baseline forecast; use larger historical datasets for production forecasting."}


def dashboard(df, filename):
    p=profile(df); rc=revenue_col(df); cc=customer_col(df); an=anomaly_engine(df)
    return {"filename":filename,"profile":p,"metrics":{"revenue":float(pd.to_numeric(df[rc],errors="coerce").fillna(0).sum()) if rc else None,"customers":int(df[cc].nunique()) if cc else len(df),"anomalies":len(an),"quality":p["quality"]},"chart":chart(df),"insights":insights(df,an),"anomalies":an,"segments":segment(df),"forecast":forecast(df),"preview":df.head(12).map(safe).to_dict(orient="records")}


def current():
    if state["df"] is None: return pd.read_csv(DEMO)
    return state["df"]


def local_answer(q, df):
    ql=q.lower(); rc=revenue_col(df); cc=customer_col(df); an=anomaly_engine(df); seg=segment(df); fc=forecast(df); p=profile(df)
    if any(x in ql for x in ["revenue","sales","income","amount"]):
        if rc:
            s=pd.to_numeric(df[rc],errors="coerce").dropna(); return f"The strongest business metric is {rc}. Total is {s.sum():,.2f}, average is {s.mean():,.2f}, and median is {s.median():,.2f}."
    if "anomal" in ql or "unusual" in ql: return f"I found {len(an)} high-priority unusual record(s). " + ("The strongest signals are " + "; ".join(an[0]["reasons"]) + "." if an else "No strong anomalies were detected.")
    if "segment" in ql or "group" in ql:
        if seg["available"]: return f"I identified {seg['k']} behavioral segments using {', '.join(seg['features'])}. The largest contains {max(seg['segments'],key=lambda x:x['records'])['records']} records."
    if "forecast" in ql or "future" in ql:
        if fc["available"]: return f"The baseline forecast for the next period is {fc['next_value']:,.2f} for {fc['metric']}. The model is a linear trend baseline, so treat it as directional rather than a production-grade prediction."
        return fc["reason"]
    if "quality" in ql or "missing" in ql: return f"Data quality is {p['quality']:.1f}%. There are {p['missing_values']} missing cells and {p['duplicate_rows']} duplicate rows across {p['rows']} records."
    if "customer" in ql and cc:
        return f"The dataset contains {df[cc].nunique()} unique customers based on the {cc} field."
    return f"I analyzed {len(df):,} rows and {df.shape[1]} columns. Ask me about revenue, anomalies, segments, forecasts, customers, or data quality."


def copilot_answer(q, df):
    # Optional real GenAI provider. Falls back to grounded local generation when no key is configured.
    key=os.getenv("OPENAI_API_KEY")
    if not key: return {"answer":local_answer(q,df),"mode":"grounded-local","reliability":88,"sources":["Current uploaded dataset","InsightForge analytics engine"]}
    try:
        import requests
        p=profile(df); rc=revenue_col(df); an=anomaly_engine(df); seg=segment(df); fc=forecast(df)
        context={"profile":p,"revenue_column":str(rc) if rc else None,"anomalies":an[:5],"segments":seg,"forecast":fc,"local_analysis":local_answer(q,df)}
        payload={"model":os.getenv("OPENAI_MODEL","gpt-5-mini"),"input":[{"role":"system","content":[{"type":"input_text","text":"You are InsightForge AI, a business data analyst. Answer only from the supplied analysis context. Be concise, explain reasoning, and never invent numbers. If evidence is insufficient, say so."}]},{"role":"user","content":[{"type":"input_text","text":f"Question: {q}\nAnalysis context:\n{context}"}]}],"max_output_tokens":500}
        r=requests.post("https://api.openai.com/v1/responses",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json=payload,timeout=45); r.raise_for_status(); data=r.json(); text=data.get("output_text")
        if not text:
            parts=[]
            for o in data.get("output",[]):
                for c in o.get("content",[]):
                    if c.get("type") in ("output_text","text"): parts.append(c.get("text",""))
            text=" ".join(parts).strip()
        return {"answer":text or local_answer(q,df),"mode":"generative-ai","reliability":93,"sources":["Current uploaded dataset","InsightForge analytics engine"]}
    except Exception as e:
        return {"answer":local_answer(q,df)+" (The configured AI provider was unavailable, so InsightForge used its grounded local analyst.)","mode":"grounded-local-fallback","reliability":82,"sources":["Current uploaded dataset","InsightForge analytics engine"]}


@app.get("/api/health")
def health(): return {"status":"ok","service":"InsightForge AI","version":app.version,"genai":bool(os.getenv("OPENAI_API_KEY"))}

@app.get("/api/demo")
def demo(): return dashboard(current(),state.get("filename") or "retail_demo.csv")

@app.post("/api/upload")
async def upload(file: UploadFile=File(...)):
    name=file.filename or "dataset"
    if not name.lower().endswith((".csv",".xlsx",".xls")): raise HTTPException(400,"Please upload a CSV or Excel file.")
    raw=await file.read()
    try: df=pd.read_csv(io.BytesIO(raw)) if name.lower().endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    except Exception as e: raise HTTPException(400,f"Could not read the file: {e}")
    if df.empty: raise HTTPException(400,"The uploaded dataset is empty.")
    if len(df)>100000: df=df.head(100000)
    state.update(df=df,filename=name)
    return dashboard(df,name)

@app.get("/api/ask")
def ask(q:str): return copilot_answer(q,current())

@app.get("/api/segmentation")
def segmentation(): return segment(current())

@app.get("/api/forecast")
def forecast_api(): return forecast(current())

@app.get("/api/insights")
def insights_api():
    df=current(); an=anomaly_engine(df); return {"insights":insights(df,an),"reliability":88}

@app.post("/api/knowledge/upload")
async def knowledge_upload(file: UploadFile=File(...)):
    name=file.filename or "document"; raw=await file.read(); text=""
    try:
        if name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            reader=PdfReader(io.BytesIO(raw)); text="\n".join((p.extract_text() or "") for p in reader.pages)
        elif name.lower().endswith((".txt",".md")): text=raw.decode("utf-8",errors="ignore")
        else: raise HTTPException(400,"Knowledge upload supports PDF, TXT, and Markdown files.")
    except HTTPException: raise
    except Exception as e: raise HTTPException(400,f"Could not read document: {e}")
    text=re.sub(r"\s+"," ",text).strip()
    if not text: raise HTTPException(400,"The document did not contain extractable text.")
    chunks=[text[i:i+1200] for i in range(0,len(text),1000)][:200]
    state["docs"].append({"name":name,"chars":len(text)})
    state["doc_chunks"].extend([{"name":name,"text":c} for c in chunks])
    return {"filename":name,"characters":len(text),"chunks":len(chunks),"documents":state["docs"]}

@app.get("/api/knowledge")
def knowledge(): return {"documents":state["docs"],"chunks":len(state["doc_chunks"])}

@app.get("/api/knowledge/ask")
def knowledge_ask(q:str):
    chunks=state["doc_chunks"]
    if not chunks: return {"answer":"No documents have been uploaded yet. Upload a PDF, TXT, or Markdown file first.","sources":[]}
    texts=[c["text"] for c in chunks]; vec=TfidfVectorizer(stop_words="english").fit(texts+[q]); mat=vec.transform(texts+[q]); scores=(mat[:-1] @ mat[-1].T).toarray().ravel(); idx=np.argsort(scores)[::-1][:3]
    sources=[{"document":chunks[i]["name"],"score":round(float(scores[i]),3),"snippet":chunks[i]["text"][:350]} for i in idx if scores[i]>0]
    answer="\n\n".join(s["snippet"] for s in sources) if sources else "I could not find a relevant passage in the uploaded documents."
    return {"answer":answer,"sources":sources,"mode":"retrieval-grounded"}

@app.get("/")
def root(): return FileResponse(FRONTEND/"index.html")

app.mount("/static",StaticFiles(directory=FRONTEND),name="static")
