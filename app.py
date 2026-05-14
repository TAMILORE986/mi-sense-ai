import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
import os, io, json
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="MI-Sense AI", page_icon="🏥")
st.markdown("""<style>
.main{background-color:#f8f9fa;}
.stButton>button{width:100%;border-radius:5px;height:3em;
  background-color:#ff4b4b;color:white;font-weight:bold;}
</style>""", unsafe_allow_html=True)

XGB_FEATURES  = ['age','sex','hs_cTnT','systolic_bp','bmi','height','weight']
CNN_SCALER    = {
    'age':{'min':18,'max':90},'sex':{'min':0,'max':1},
    'hs_cTnT':{'min':2,'max':1000},'systolic_bp':{'min':80,'max':200},
    'bmi':{'min':15,'max':50},'height':{'min':140,'max':210},
    'weight':{'min':40,'max':160}}
CNN_FEATURES  = list(CNN_SCALER.keys())
TROP_NORMAL   = 14   # ng/L

# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess_ecg(sig, fs=500, n=1000):
    nyq=0.5*fs; b,a=butter(3,[0.5/nyq,40/nyq],btype='band')
    f=filtfilt(b,a,sig,axis=0)
    f=(f-np.mean(f,0))/(np.std(f,0)+1e-9)
    if f.shape[0]>n: return f[::f.shape[0]//n][:n]
    if f.shape[0]<n: return np.pad(f,((0,n-f.shape[0]),(0,0)))
    return f

def load_ecg(up):
    if up is None: return None
    try:
        d=np.load(io.BytesIO(up.read())) if up.name.endswith('.npy') \
          else pd.read_csv(up,header=None).values
        if d.ndim==1: d=d.reshape(-1,1)
        if d.shape[1]!=12:
            d=np.pad(d,((0,0),(0,max(0,12-d.shape[1])))) if d.shape[1]<12 else d[:,:12]
        return d.astype(np.float32)
    except Exception as e: st.error(f"ECG load error: {e}"); return None

def card(label,value,color=None):
    bl=f"border-left:4px solid {color};" if color else ""
    st.markdown(f"""<div style="padding:10px 14px;margin-bottom:7px;border-radius:8px;
    border:1px solid rgba(128,128,128,0.2);background:rgba(128,128,128,0.07);{bl}">
    <div style="font-size:11px;color:gray;">{label}</div>
    <div style="font-size:18px;font-weight:600;">{value}</div></div>""",
    unsafe_allow_html=True)

def alert(msg,level='warning'):
    c={'danger':('#c0392b','#fadbd8'),'warning':('#e67e22','#fef9e7'),
       'info':('#2980b9','#d6eaf8'),'ok':('#27ae60','#d5f5e3')}
    fc,bg=c.get(level,c['info'])
    st.markdown(f"""<div style="background:{bg};border-left:5px solid {fc};
    padding:10px 14px;border-radius:6px;margin-bottom:8px;color:{fc};
    font-weight:500;">{msg}</div>""",unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cnn=tf.keras.models.load_model('mi_detection_model_full.h5') \
        if os.path.exists('mi_detection_model_full.h5') else None
    xm=None
    if os.path.exists('xgboost_model.json'):
        xm=xgb.XGBClassifier(); xm.load_model('xgboost_model.json')
    tm=None
    if os.path.exists('mi_type_classifier.json'):
        tm=xgb.XGBClassifier(); tm.load_model('mi_type_classifier.json')
    tl=None
    if os.path.exists('mi_type_label_map.json'):
        with open('mi_type_label_map.json') as f:
            tl={int(k):v for k,v in json.load(f).items()}
    return cnn,xm,tm,tl

cnn_model,xgb_model,type_model,type_labels=load_models()

if 'log' not in st.session_state: st.session_state.log=[]

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png",width=80)
    st.title("MI-Sense AI")
    st.markdown("---")
    engine=st.selectbox("🧠 AI Engine",["CNN-LSTM (Multimodal)","XGBoost (Clinical)"])

    # ECG upload
    st.markdown("---")
    st.header("📤 Upload 12-Lead ECG")
    st.caption("Accepted: .npy or .csv (12 columns)")
    ecg_up=st.file_uploader("ECG signal file",type=["npy","csv"])
    ecg_data=load_ecg(ecg_up)
    if ecg_data is not None:
        st.success(f"✅ ECG loaded: {ecg_data.shape[0]}×{ecg_data.shape[1]}")
    else:
        st.info("ℹ️ No ECG — synthetic used for CNN-LSTM")

    # ── A. Core Vitals ─────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📋 A. Core Vitals")
    st.caption("🔬 Model inputs")
    age   =st.slider("Age (years)",18,90,60)
    sex   =st.selectbox("Sex",["Female","Male"])
    st.caption("⚠️ MI risk ↑ sharply >45 (M) or >55 (F)")
    height=st.slider("Height (cm)",140,210,170)
    weight=st.slider("Weight (kg)",40,160,70)
    bmi   =weight/((height/100)**2)
    trop  =st.slider("Troponin T hs-cTnT (ng/L)",2,1000,14)

    st.markdown("---")
    st.caption("📊 Monitored Vitals")
    hr_rhythm=st.selectbox("Heart Rhythm",
        ["Normal Sinus","Sinus Tachycardia (>100)","Sinus Bradycardia (<60)",
         "Atrial Fibrillation","Ventricular Tachycardia","Complete Heart Block"])
    # Set HR default from rhythm
    hr_default={"Normal Sinus":72,"Sinus Tachycardia (>100)":115,
                "Sinus Bradycardia (<60)":48,"Atrial Fibrillation":110,
                "Ventricular Tachycardia":160,"Complete Heart Block":35}
    hr   =st.slider("Heart Rate (bpm)",30,220,hr_default.get(hr_rhythm,72))
    bps  =st.slider("Systolic BP (mmHg)",60,220,120)
    bpd  =st.slider("Diastolic BP (mmHg)",40,130,80)
    pp   =bps-bpd  # pulse pressure
    rr   =st.slider("Respiratory Rate (br/min)",8,40,16)
    cheyne=st.checkbox("Cheyne-Stokes breathing pattern")
    spo2 =st.slider("SpO2 (%)",70,100,98)
    desat_exertion=st.checkbox("Desaturation on exertion (SpO2 drops >4%)")
    temp =st.slider("Temperature (°C)",35.0,40.0,36.6)
    chol =st.slider("Cholesterol (mg/dL)",100,300,200)
    gluc =st.slider("Glucose (mg/dL)",70,200,100)

    # ── B. Symptoms ────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("🩺 B. Symptoms")
    st.caption("Toggle all that apply")
    s_chest  =st.checkbox("Chest pain (substernal/crushing/radiating to arm/jaw/back)")
    s_sob    =st.checkbox("Shortness of breath")
    s_nausea =st.checkbox("Nausea / vomiting")
    s_diaphor=st.checkbox("Diaphoresis (cold sweat) ← highly specific for MI")
    s_syncope=st.checkbox("Syncope / near-syncope")
    s_indig  =st.checkbox("Indigestion (common in inferior MI)")
    s_epigast=st.checkbox("Epigastric pain (women/elderly)")
    sym_score=sum([s_chest*2,s_sob,s_nausea,s_diaphor*2,s_syncope,s_indig,s_epigast])

    # ── C. ECG Features ────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📉 C. ECG Features")
    ecg_feat=st.selectbox("ECG Finding",
        ["Normal","ST Elevation","ST Depression",
         "T-Wave Inversion (Wellens')","New LBBB","Pathologic Q Waves"])
    ecg_leads=[]
    if ecg_feat=="ST Elevation":
        st.caption("Pick territory:")
        territory=st.radio("Territory",
            ["Inferior (II,III,aVF)","Anterior (V1-V4)",
             "Lateral (I,aVL,V5,V6)","Custom"])
        if territory=="Inferior (II,III,aVF)":
            ecg_leads=["II","III","aVF"]
        elif territory=="Anterior (V1-V4)":
            ecg_leads=["V1","V2","V3","V4"]
        elif territory=="Lateral (I,aVL,V5,V6)":
            ecg_leads=["I","aVL","V5","V6"]
        else:
            ecg_leads=st.multiselect("Select leads",
                ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"])
    ecg_score={"Normal":0,"ST Elevation":2,"ST Depression":1,
               "T-Wave Inversion (Wellens')":1,"New LBBB":2,"Pathologic Q Waves":1
               }.get(ecg_feat,0)

    # ── D. Risk Factors ────────────────────────────────────────────────────
    st.markdown("---")
    st.header("⚠️ D. Risk Factors & History")
    rf_dm  =st.checkbox("Diabetes (silent ischemia risk)")
    rf_mi  =st.checkbox("Prior MI / stent / CABG")
    rf_smoke=st.checkbox("Smoking")
    rf_htn =st.checkbox("Hypertension (diagnosed)")
    rf_lipid=st.checkbox("Hyperlipidemia")
    rf_fam =st.checkbox("Family history of CAD")
    rf_count=sum([rf_dm,rf_mi,rf_smoke,rf_htn,rf_lipid,rf_fam])

# ── Build model inputs ─────────────────────────────────────────────────────────
xgb_df=pd.DataFrame([{
    'age':age,'sex':1 if sex=="Male" else 0,'hs_cTnT':trop,
    'systolic_bp':bps,'bmi':round(bmi,2),'height':height,'weight':weight
}],columns=XGB_FEATURES)

cnn_df=pd.DataFrame([{k:xgb_df[k].iloc[0] for k in CNN_FEATURES}])
for col in CNN_FEATURES:
    mn,mx=CNN_SCALER[col]['min'],CNN_SCALER[col]['max']
    cnn_df[col]=(cnn_df[col]-mn)/(mx-mn+1e-9)
cnn_scaled=cnn_df.values

# ══════════════════════════════════════════════════════════════════════════════
#  REAL-TIME ALERTS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
live_alerts=[]
if hr>100 and bps<90:
    live_alerts.append(("🚨 CARDIOGENIC SHOCK suspected — HR >100 + SBP <90 mmHg. "
        "IV fluids cautiously, inotropes may be needed.","danger"))
if hr<50 and bps>=90 and ecg_feat=="ST Elevation" and \
   any(l in ecg_leads for l in ["II","III","aVF"]):
    live_alerts.append(("🚨 RV INFARCT pattern — Bradycardia + inferior ST elevation. "
        "AVOID nitrates and diuretics. Give IV saline.","danger"))
if rr>24 and spo2<92:
    live_alerts.append(("⚠️ PULMONARY OEDEMA suspected — RR >24 + SpO2 <92%. "
        "Left heart failure possible.","warning"))
if bps>=180 and bpd>=100 and s_chest:
    live_alerts.append(("⚠️ AORTIC DISSECTION must be ruled out — BP 180/100+ with chest pain. "
        "Check pulse differential, get CT aorta.","warning"))
if pp<25:
    live_alerts.append((f"⚠️ NARROW PULSE PRESSURE ({pp} mmHg) — "
        "Suggests low stroke volume (tamponade, severe AS, cardiogenic shock).","warning"))
if ecg_feat=="New LBBB":
    live_alerts.append(("🚨 NEW LBBB — Treat as STEMI equivalent per Sgarbossa criteria. "
        "Activate cath lab.","danger"))
if trop>3*TROP_NORMAL:
    live_alerts.append((f"🚨 Troponin {trop} ng/L — >3× normal upper limit. "
        "Significant myocardial injury.","danger"))
elif trop>TROP_NORMAL:
    live_alerts.append((f"⚠️ Troponin elevated ({trop} ng/L > {TROP_NORMAL} ng/L normal limit).","warning"))
if cheyne:
    live_alerts.append(("⚠️ Cheyne-Stokes breathing — Associated with severe LV dysfunction "
        "or cerebral hypoperfusion.","warning"))
if desat_exertion:
    live_alerts.append(("ℹ️ Exertional desaturation — Suggests impaired cardiac output or "
        "pulmonary congestion.","info"))
if spo2<90:
    live_alerts.append(("🚨 SpO2 <90% — Supplemental oxygen indicated immediately.","danger"))
# Age-based risk note
if (sex=="Male" and age>45) or (sex=="Female" and age>55):
    live_alerts.append((f"ℹ️ Age/sex risk threshold exceeded "
        f"({'M >45' if sex=='Male' else 'F >55'}). "
        "MI risk rises sharply in this demographic.","info"))
# Atypical presentation warning for women
if sex=="Female" and (s_indig or s_epigast or s_nausea) and not s_chest:
    live_alerts.append(("⚠️ ATYPICAL PRESENTATION — Women more commonly present with "
        "nausea/epigastric pain/indigestion without classic chest pain. "
        "Do not rule out MI.","warning"))

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col1,col2=st.columns([2,1])

with col1:
    st.title("🏥 Cardiac Diagnostic Dashboard")
    st.info("System Status: Online  |  Model Ready  |  Realistic Performance Mode")

    if live_alerts:
        st.markdown("### 🔔 Active Clinical Alerts")
        for msg,lvl in live_alerts: alert(msg,lvl)
        st.markdown("---")

    tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
        "📊 Diagnostic","📈 ECG","🧮 Risk Scores",
        "📉 Lab Trends","🔄 Serial","⚗️ Treatment","🔍 Explainability"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — DIAGNOSTIC
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        if st.button("🚀 RUN FULL DIAGNOSTIC"):
            with st.spinner("Analysing…"):
                if engine=="CNN-LSTM (Multimodal)":
                    if cnn_model is None: st.error("Model file missing."); st.stop()
                    raw=ecg_data if ecg_data is not None \
                        else np.random.randn(1000,12).astype(np.float32)
                    if ecg_data is None:
                        st.warning("No ECG uploaded — synthetic signal used.")
                    eb=np.expand_dims(preprocess_ecg(raw),0)
                    prob=float(cnn_model.predict([eb,cnn_scaled])[0][0])
                else:
                    if xgb_model is None: st.error("Model file missing."); st.stop()
                    prob=float(xgb_model.predict_proba(xgb_df)[0][1])

                fig=go.Figure(go.Indicator(
                    mode="gauge+number",value=prob*100,
                    domain={'x':[0,1],'y':[0,1]},
                    title={'text':"MI Risk Probability (%)","font":{"size":22}},
                    gauge={'axis':{'range':[0,100]},
                           'bar':{'color':"#ff4b4b" if prob>0.5 else "#2ecc71"},
                           'steps':[{'range':[0,30],'color':'#e8f8f5'},
                                    {'range':[30,70],'color':'#fef9e7'},
                                    {'range':[70,100],'color':'#fdedec'}]}))
                st.plotly_chart(fig,use_container_width=True)

                if prob>0.5:
                    st.error(f"### ⚠️ HIGH RISK ({prob:.1%}) — Immediate cardiology consult.")

                    # ── Mimic Check ───────────────────────────────────────
                    st.markdown("---")
                    st.subheader("🔍 False Positive / Mimic Check")
                    mimics=[]
                    all_leads=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
                    if ecg_feat=="ST Elevation" and \
                       len([l for l in all_leads if l in (ecg_leads or [])]) >= 5:
                        mimics.append(("Pericarditis",
                            "Diffuse ST elevation in ≥5 leads with no reciprocal changes. "
                            "Check for PR depression and saddle-shaped morphology.","warning"))
                    if age<40 and not rf_mi and ecg_feat=="ST Elevation":
                        mimics.append(("Early Repolarisation",
                            "Young patient with ST elevation — J-point elevation with "
                            "concave ST is common in athletes. Compare with old ECG.","warning"))
                    if hr>100 and spo2<94 and ecg_feat in ["ST Depression","T-Wave Inversion (Wellens')"]:
                        mimics.append(("Pulmonary Embolism",
                            "Tachycardia + hypoxaemia + right-heart strain pattern. "
                            "Check for S1Q3T3. Get D-dimer / CTPA.","warning"))
                    if ecg_feat=="T-Wave Inversion (Wellens')" and \
                       any(l in (ecg_leads or all_leads) for l in ["V5","V6"]):
                        mimics.append(("LVH with Strain",
                            "Deep T-wave inversions in V5–V6 may represent LVH strain pattern. "
                            "Check voltage criteria.","info"))
                    if s_sob and not s_chest and s_epigast and \
                       age<55 and trop<TROP_NORMAL:
                        mimics.append(("Takotsubo (Broken Heart Syndrome)",
                            "Acute emotional/physical stress + apical wall motion abnormality "
                            "with normal coronaries. Troponin mildly elevated. "
                            "Echo is diagnostic (apical ballooning).","warning"))
                    if rf_smoke and bps>=160 and ecg_feat=="ST Elevation":
                        mimics.append(("Coronary Spasm (Prinzmetal)",
                            "Transient ST elevation in young smoker with hypertension. "
                            "May resolve spontaneously. CCB therapy.","info"))
                    if mimics:
                        st.warning("⚠️ These conditions can mimic MI — review before diagnosing:")
                        for name,desc,lvl in mimics: alert(f"**{name}:** {desc}",lvl)
                    else:
                        st.success("✅ No common MI mimics flagged.")

                    # ── MI Type ───────────────────────────────────────────
                    st.markdown("---")
                    st.subheader("🫀 MI Type Classification")
                    if type_model and type_labels:
                        tp=type_model.predict_proba(xgb_df)[0]
                        ti=int(np.argmax(tp))
                        tn=type_labels[ti]
                        tc=float(tp[ti])
                        bmap={'Inferior MI':'#e74c3c','Anterior MI':'#e67e22',
                              'Anteroseptal MI':'#d35400','Anterolateral MI':'#c0392b',
                              'Inferolateral MI':'#8e44ad','Inferoposterior MI':'#2980b9',
                              'Inferoposterolateral MI':'#1abc9c','Lateral MI':'#16a085',
                              'Posterior MI':'#27ae60'}
                        bc=bmap.get(tn,'#7f8c8d')
                        st.markdown(f"""<div style="background:{bc}22;border-left:5px solid {bc};
                        padding:14px 18px;border-radius:8px;margin-bottom:12px;">
                        <div style="font-size:11px;color:gray;">Predicted MI Type</div>
                        <div style="font-size:24px;font-weight:700;color:{bc};">{tn}</div>
                        <div style="font-size:11px;color:gray;">Confidence: {tc:.1%}</div>
                        </div>""",unsafe_allow_html=True)
                        tdf=pd.DataFrame({'Type':[type_labels[i] for i in range(len(tp))],
                                          'Prob':tp}).sort_values('Prob')
                        fig_t=go.Figure(go.Bar(x=tdf['Prob']*100,y=tdf['Type'],
                            orientation='h',
                            marker_color=[bc if t==tn else '#bdc3c7' for t in tdf['Type']],
                            text=[f"{p:.1f}%" for p in tdf['Prob']*100],
                            textposition='outside'))
                        fig_t.update_layout(xaxis_title="%",xaxis=dict(range=[0,115]),
                            height=300,margin=dict(l=5,r=5,t=30,b=5))
                        st.plotly_chart(fig_t,use_container_width=True)
                    else:
                        st.warning("Run `python train_mi_type_classifier.py` to enable this.")
                        st.markdown("""| ECG Leads|Suspected Type|
|---|---|
|II,III,aVF|Inferior MI|
|V1–V4|Anterior MI|
|I,aVL,V5–V6|Lateral MI|
|V1–V2 reciprocal|Posterior MI|""")

                    st.session_state.log.append({
                        'Time':datetime.now().strftime("%H:%M:%S"),
                        'Risk %':round(prob*100,1),'HR':hr,'SBP':bps,'Trop':trop,'SpO2':spo2})
                else:
                    st.success(f"### ✅ LOW RISK ({prob:.1%}) — Within normal variance.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — ECG VISUALISATION
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("12-Lead ECG Signal Analysis")
        lead_names=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        if ecg_data is not None:
            sig_display=preprocess_ecg(ecg_data); label="(Patient ECG)"
        else:
            sig_display=np.random.randn(1000,12).astype(np.float32)
            label="(Synthetic — upload ECG for real analysis)"

        # Apply ECG feature overlays
        sig_mod=sig_display.copy()
        for i,ln in enumerate(lead_names):
            if ecg_feat=="ST Elevation" and ln in ecg_leads:
                sig_mod[400:700,i]+=0.5
            elif ecg_feat=="ST Depression":
                sig_mod[400:700,i]-=0.35
            elif ecg_feat=="T-Wave Inversion (Wellens')":
                sig_mod[700:900,i]*=-0.9
            elif ecg_feat=="New LBBB":
                sig_mod[300:500,i]+=0.3*np.sin(np.linspace(0,np.pi,200))
            elif ecg_feat=="Pathologic Q Waves":
                sig_mod[100:160,i]-=0.4

        t=np.linspace(0,10,sig_mod.shape[0])
        fig_ecg,axes=plt.subplots(12,1,figsize=(12,11),sharex=True)
        for i in range(12):
            color='#c0392b'
            bg=None
            if ecg_feat=="ST Elevation" and lead_names[i] in ecg_leads: bg='#fff0f0'
            elif ecg_feat=="ST Depression": bg='#f0f0ff'
            if bg: axes[i].set_facecolor(bg)
            axes[i].plot(t,sig_mod[:,i],color=color,lw=0.8)
            axes[i].set_ylabel(lead_names[i],fontsize=8,rotation=0,labelpad=22)
            axes[i].grid(True,alpha=0.2); axes[i].set_yticks([])
        axes[-1].set_xlabel("Time (s)")
        fig_ecg.suptitle(f"12-Lead ECG {label} | Finding: {ecg_feat}"
            +(" Leads: "+", ".join(ecg_leads) if ecg_leads else ""),fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_ecg)
        if hr_rhythm!="Normal Sinus":
            st.info(f"**Rhythm noted:** {hr_rhythm} (HR: {hr} bpm)")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — RISK SCORES
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("🧮 Clinical Risk Score Calculators")
        rc1,rc2,rc3=st.columns(3)

        # HEART Score
        with rc1:
            st.markdown("#### HEART Score")
            st.caption("History · ECG · Age · Risk factors · Troponin")
            hh=2 if (s_chest and s_diaphor) else (1 if s_chest else 0)
            he=ecg_score
            ha=2 if age>=65 else (1 if age>=45 else 0)
            hr2=2 if (rf_mi or rf_count>=3) else (1 if rf_count>=1 else 0)
            ht=2 if trop>3*TROP_NORMAL else (1 if trop>TROP_NORMAL else 0)
            heart=hh+he+ha+hr2+ht
            for k,v in [("History",hh),("ECG",he),("Age",ha),
                        ("Risk factors",hr2),("Troponin",ht)]:
                st.markdown(f"**{k}:** {v}/2")
            st.markdown("---")
            hc="#e74c3c" if heart>=7 else ("#e67e22" if heart>=4 else "#27ae60")
            hrisk="HIGH — admit" if heart>=7 else ("MODERATE — observe" if heart>=4 else "LOW — discharge safe")
            st.markdown(f"""<div style="background:{hc}22;border-left:5px solid {hc};
            padding:12px;border-radius:8px;text-align:center;">
            <div style="font-size:30px;font-weight:700;color:{hc};">{heart}/10</div>
            <div style="font-size:12px;color:{hc};">{hrisk}</div></div>""",
            unsafe_allow_html=True)

        # TIMI Score (STEMI)
        with rc2:
            st.markdown("#### TIMI Score (STEMI)")
            st.caption("30-day mortality risk")
            ta=3 if age>=75 else (2 if age>=65 else 0)
            th=1 if (rf_dm or rf_htn or s_chest) else 0
            ts=3 if bps<100 else 0
            thr=2 if hr>100 else 0
            tant=1 if (ecg_feat=="ST Elevation" and
                       any(l in ecg_leads for l in ["V1","V2","V3","V4"])) \
                      or ecg_feat=="New LBBB" else 0
            timi=ta+th+ts+thr+tant
            mort={0:0.1,1:1.6,2:2.2,3:4.4,4:7.3,5:12.4,6:16.1,7:23.4,8:26.8}
            mp=mort.get(min(timi,8),26.8)
            for k,v in [("Age pts",ta),("DM/HTN/AP",th),
                        ("SBP <100",ts),("HR >100",thr),("Ant/LBBB",tant)]:
                st.markdown(f"**{k}:** {v}")
            st.markdown("---")
            tc2="#e74c3c" if timi>=5 else ("#e67e22" if timi>=3 else "#27ae60")
            st.markdown(f"""<div style="background:{tc2}22;border-left:5px solid {tc2};
            padding:12px;border-radius:8px;text-align:center;">
            <div style="font-size:30px;font-weight:700;color:{tc2};">{timi}/14</div>
            <div style="font-size:12px;color:{tc2};">~{mp}% 30-day mortality</div></div>""",
            unsafe_allow_html=True)

        # GRACE Score
        with rc3:
            st.markdown("#### GRACE Score")
            st.caption("In-hospital & 6-month mortality")
            ga=(age-30)//10*8
            ghr=0 if hr<70 else (3 if hr<90 else (9 if hr<110 else (15 if hr<150 else 24)))
            gs=0 if bps>=200 else (10 if bps>=160 else (24 if bps>=140 else (
               34 if bps>=120 else (43 if bps>=100 else (53 if bps>=80 else 58)))))
            gecg=28 if ecg_feat in ["ST Elevation","New LBBB"] else \
                 (14 if ecg_feat in ["ST Depression","T-Wave Inversion (Wellens')"] else 0)
            gshock=39 if (hr>100 and bps<90) else 0
            grace=ga+ghr+gs+gecg+gshock
            for k,v in [("Age",ga),("HR",ghr),("SBP",gs),("ECG",gecg),("Shock",gshock)]:
                st.markdown(f"**{k}:** {v}")
            st.markdown("---")
            grisk="HIGH" if grace>140 else ("MODERATE" if grace>108 else "LOW")
            gc3="#e74c3c" if grace>140 else ("#e67e22" if grace>108 else "#27ae60")
            st.markdown(f"""<div style="background:{gc3}22;border-left:5px solid {gc3};
            padding:12px;border-radius:8px;text-align:center;">
            <div style="font-size:30px;font-weight:700;color:{gc3};">{grace}</div>
            <div style="font-size:12px;color:{gc3};">{grisk} risk</div></div>""",
            unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — E. LAB TRENDS
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("📉 E. Lab Trends — Simulated Over Time")
        hours=np.array([0,1,2,3,4,6,8,12,24,36,48])
        base=np.array([0.05,0.08,0.15,0.35,0.55,0.75,0.88,0.97,1.0,0.90,0.70])

        # hs-cTnT curve peaking at entered value
        trop_c=base*max(trop,TROP_NORMAL*2)
        # CK-MB peaks earlier (~12-18h) then declines
        ckmb_base=np.array([0.05,0.10,0.20,0.45,0.70,0.90,0.99,0.95,0.75,0.50,0.30])
        ckmb_c=ckmb_base*max(trop*0.05,8)
        # BNP elevated if heart failure pattern
        bnp_val=350 if (hr>100 and bps<90) else (150 if rr>24 else 60)
        bnp_base=np.array([0.2,0.3,0.4,0.5,0.65,0.80,0.90,0.97,1.0,0.98,0.95])
        bnp_c=bnp_base*bnp_val

        fig_lab=go.Figure()
        fig_lab.add_trace(go.Scatter(x=hours,y=trop_c,name='hs-cTnT (ng/L)',
            line=dict(color='#e74c3c',width=2.5),mode='lines+markers'))
        fig_lab.add_trace(go.Scatter(x=hours,y=ckmb_c,name='CK-MB (ng/mL)',
            line=dict(color='#e67e22',width=2,dash='dot'),mode='lines+markers'))
        fig_lab.add_trace(go.Scatter(x=hours,y=bnp_c,name='BNP (pg/mL)',
            line=dict(color='#8e44ad',width=2,dash='dash'),mode='lines+markers'))
        fig_lab.add_hline(y=TROP_NORMAL,line_dash="dash",line_color="gray",
            annotation_text=f"Normal limit ({TROP_NORMAL} ng/L)")
        fig_lab.update_layout(
            title="Biomarker Rise & Fall Curve — Undetectable → Peak at 12–24h → Resolution",
            xaxis_title="Hours from symptom onset",yaxis_title="Level",
            height=380,hovermode='x unified')
        st.plotly_chart(fig_lab,use_container_width=True)

        if trop>TROP_NORMAL:
            pct=((trop-TROP_NORMAL)/TROP_NORMAL)*100
            if pct>20:
                alert(f"🚨 Troponin {pct:.0f}% above normal — "
                    "Serial troponin at 3h required (ESC 0h/3h protocol).","danger")

        st.info("""**ESC 0h/3h Rapid Rule-Out Protocol:**
- hs-cTnT <5 ng/L at 0h → Rule OUT
- hs-cTnT >52 ng/L at 0h → Rule IN  
- Rise ≥6 ng/L at 3h → Rule IN
- Intermediate → observe and recheck at 3h""")

        # CK-MB teaching note
        st.markdown("---")
        st.markdown("""**CK-MB (Older Marker — Teachable):**  
CK-MB peaks earlier than troponin (~12–18h) and falls faster (~36–48h).  
Useful for detecting **reinfarction** (second rise after initial decline).""")
        st.markdown("""**BNP (Heart Failure Marker):**  
BNP rises when ventricular wall stress increases — elevated in acute MI with  
LV dysfunction, pulmonary oedema, or cardiogenic shock.""")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — A. SERIAL ASSESSMENT
    # ══════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("🔄 A. Time-Stamped Serial Assessment")
        st.caption("Log vitals every 15–30 min to monitor trend. "
                   "Alert fires if troponin rises >20% between readings.")
        sc1,sc2=st.columns(2)
        with sc1:
            if st.button("📝 Log Current Vitals"):
                st.session_state.log.append({
                    'Time':datetime.now().strftime("%H:%M:%S"),
                    'Risk %':None,'HR':hr,'SBP':bps,'Trop':trop,'SpO2':spo2})
                st.success("Vitals logged.")
        with sc2:
            if st.button("🗑️ Clear Log"):
                st.session_state.log=[]; st.success("Log cleared.")

        if st.session_state.log:
            df_log=pd.DataFrame(st.session_state.log)
            st.dataframe(df_log,use_container_width=True)
            if len(df_log)>=2:
                idx=list(range(len(df_log)))
                fig_tr=go.Figure()
                fig_tr.add_trace(go.Scatter(x=idx,y=df_log['HR'],
                    name='Heart Rate',line=dict(color='#e74c3c')))
                fig_tr.add_trace(go.Scatter(x=idx,y=df_log['SBP'],
                    name='SBP',line=dict(color='#3498db')))
                fig_tr.add_trace(go.Scatter(x=idx,y=df_log['Trop'],
                    name='Troponin',line=dict(color='#e67e22',dash='dot')))
                fig_tr.update_layout(title="Vital Trend Over Serial Assessments",
                    xaxis_title="Assessment #",height=300)
                st.plotly_chart(fig_tr,use_container_width=True)
                # Troponin delta alert
                trops=df_log['Trop'].dropna().tolist()
                if len(trops)>=2 and trops[0]>0:
                    delta=(trops[-1]-trops[0])/trops[0]*100
                    if delta>20:
                        alert(f"🚨 Troponin rose {delta:.0f}% since first reading "
                            "— significant delta rise, consider rule-in MI.","danger")
                    elif delta>0:
                        alert(f"ℹ️ Troponin trend: +{delta:.0f}% — continue monitoring.","info")
                    else:
                        alert("✅ Troponin stable or falling — reassuring.","ok")
        else:
            st.info("No logs yet. Run a diagnostic or log vitals above.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — B. TREATMENT SIMULATOR
    # ══════════════════════════════════════════════════════════════════════
    with tab6:
        st.subheader("⚗️ B. Treatment Simulator (Educational)")
        st.caption("Select interventions to see their simulated effect and clinical rationale.")
        tc1,tc2=st.columns(2)
        with tc1:
            g_asp  =st.checkbox("💊 Aspirin 324 mg (antiplatelet)")
            g_nitro=st.checkbox("💊 Nitroglycerin (vasodilator)")
            g_morph=st.checkbox("💊 Morphine (analgesia)")
            g_o2   =st.checkbox("💊 Oxygen supplementation")
        with tc2:
            g_hep  =st.checkbox("💊 Heparin (anticoagulation)")
            g_beta =st.checkbox("💊 Beta-blocker (metoprolol)")
            g_acei =st.checkbox("💊 ACE inhibitor (ramipril)")
            g_statin=st.checkbox("💊 High-dose statin (atorvastatin 80mg)")

        sim_hr=hr; sim_bps=bps; sim_rr=rr; sim_spo2=spo2
        effects=[]

        if g_asp:
            effects.append(("✅ Aspirin 324 mg",
                "Irreversibly inhibits COX-1, reducing thromboxane A2 and platelet aggregation. "
                "ISIS-2 trial: 23% reduction in 30-day mortality. "
                "Give immediately unless active GI bleed.","ok"))

        if g_nitro:
            if bps<90 or hr<50 or \
               (any(l in ecg_leads for l in ["II","III","aVF"]) and ecg_feat=="ST Elevation"):
                sim_bps=max(50,bps-35); sim_hr=hr+20
                effects.append(("🚨 Nitroglycerin — CONTRAINDICATED",
                    f"BP crashed: {bps} → {sim_bps} mmHg! "
                    "Nitrates are contraindicated in RV infarct, hypotension, or bradycardia. "
                    "Venodilation reduces preload — fatal in RV-dependent states.","danger"))
            else:
                sim_bps-=15
                effects.append(("✅ Nitroglycerin",
                    f"Venodilation → preload reduction. SBP: {bps} → {sim_bps} mmHg. "
                    "Relieves ischaemic chest pain. Caution if SBP <90.","ok"))

        if g_morph:
            sim_rr=max(8,rr-5); sim_hr=hr-8
            effects.append(("⚠️ Morphine",
                f"Analgesia achieved. RR: {rr} → {sim_rr} br/min. HR: {hr} → {sim_hr} bpm. "
                "CRUSADE registry: associated with increased mortality in NSTEMI. "
                "May mask worsening ischaemia and delay diagnosis. "
                "Current ESC guidelines: caution, consider fentanyl instead.","warning"))

        if g_o2:
            if spo2>=94:
                effects.append(("⚠️ Oxygen — NOT indicated",
                    f"SpO2 is {spo2}% (≥94%). Hyperoxia causes coronary vasoconstriction "
                    "and increases infarct size (AVOID-HF trial). "
                    "Only give O2 if SpO2 <90%.","warning"))
            else:
                sim_spo2=min(100,spo2+6)
                effects.append(("✅ Oxygen indicated",
                    f"SpO2 corrected: {spo2}% → {sim_spo2}%. "
                    "SpO2 was <94% — supplemental oxygen is appropriate.","ok"))

        if g_hep:
            effects.append(("✅ Heparin (UFH/LMWH)",
                "Prevents clot propagation by inhibiting thrombin (factor IIa) and Xa. "
                "Standard of care in STEMI (before PCI) and NSTEMI. "
                "Monitor aPTT if UFH used (target 60–100s).","ok"))

        if g_beta:
            if hr<60 or bps<100 or hr_rhythm in ["Complete Heart Block","Sinus Bradycardia (<60)"]:
                effects.append(("🚨 Beta-blocker CONTRAINDICATED",
                    f"HR {hr} bpm + SBP {bps} mmHg + rhythm: {hr_rhythm}. "
                    "Beta-blocker will worsen bradycardia/hypotension/heart block. "
                    "Hold until haemodynamically stable.","danger"))
            else:
                sim_hr-=14; sim_bps-=10
                effects.append(("✅ Metoprolol",
                    f"HR: {hr} → {sim_hr} bpm, SBP: {bps} → {sim_bps} mmHg. "
                    "Reduces myocardial O2 demand. Decreases VF risk in acute MI. "
                    "Start low-dose PO when stable.","ok"))

        if g_acei:
            if bps<100:
                effects.append(("🚨 ACE inhibitor — Hold",
                    f"SBP {bps} mmHg — risk of dangerous hypotension. "
                    "Start when SBP >100 mmHg, typically after 24h if stable.","danger"))
            else:
                sim_bps-=8
                effects.append(("✅ Ramipril (ACE inhibitor)",
                    f"SBP: {bps} → {sim_bps} mmHg. "
                    "Reduces LV remodelling post-MI. AIRE trial: 27% mortality reduction. "
                    "Start within 24h in anterior STEMI or LV dysfunction.","ok"))

        if g_statin:
            effects.append(("✅ Atorvastatin 80 mg (high-intensity)",
                "Plaque stabilisation, anti-inflammatory, reduces future events. "
                "PROVE IT-TIMI 22: intensive statin reduced death/MI vs moderate dose. "
                "Give as early as possible regardless of baseline LDL.","ok"))

        if effects:
            st.markdown("### 📋 Simulated Treatment Outcomes")
            for title,desc,lvl in effects: alert(f"**{title}** — {desc}",lvl)
            st.markdown("### 📊 Post-Treatment Vitals")
            vc1,vc2,vc3,vc4=st.columns(4)
            vc1.metric("Heart Rate",f"{sim_hr} bpm",f"{sim_hr-hr:+d}")
            vc2.metric("SBP",f"{sim_bps} mmHg",f"{sim_bps-bps:+d}")
            vc3.metric("RR",f"{sim_rr} br/min",f"{sim_rr-rr:+d}")
            vc4.metric("SpO2",f"{sim_spo2}%",f"{sim_spo2-spo2:+d}")
        else:
            st.info("Select one or more interventions above to simulate their effects.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 7 — AI EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════════
    with tab7:
        st.subheader("🔍 SHAP Feature Importance")
        st.write("Shows which clinical factors drove the current prediction.")
        if engine=="CNN-LSTM (Multimodal)":
            if cnn_model is None: st.error("CNN-LSTM model not loaded.")
            else:
                raw=ecg_data if ecg_data is not None \
                    else np.random.randn(1000,12).astype(np.float32)
                fe=np.expand_dims(preprocess_ecg(raw),0)
                def wrap(c):
                    return cnn_model.predict(
                        [np.repeat(fe,c.shape[0],0),c],verbose=0)
                ex=shap.KernelExplainer(wrap,np.zeros((1,cnn_scaled.shape[1])))
                sv=ex.shap_values(cnn_scaled,nsamples=50)
                v=sv[0] if isinstance(sv,list) else sv
                fig_s,_=plt.subplots()
                shap.summary_plot(v,cnn_scaled,feature_names=CNN_FEATURES,
                                  show=False,plot_type="bar")
                plt.tight_layout(); st.pyplot(fig_s)
        else:
            if xgb_model is None: st.error("XGBoost model not loaded.")
            else:
                ex=shap.TreeExplainer(xgb_model)
                sv=ex.shap_values(xgb_df)
                fig_s,_=plt.subplots()
                shap.summary_plot(sv,xgb_df,feature_names=XGB_FEATURES,
                                  show=False,plot_type="bar")
                plt.tight_layout(); st.pyplot(fig_s)

# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN — CLINICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with col2:
    st.subheader("📋 Clinical Summary")

    st.caption("🔬 Model inputs")
    card("Age",f"{age} yrs",
         "#e67e22" if (sex=="Male" and age>45) or (sex=="Female" and age>55) else None)
    card("Sex",sex+"  ⚠️ atypical risk" if sex=="Female" else sex)
    card("BMI",f"{bmi:.1f} kg/m²",
         "#e67e22" if bmi>30 or bmi<18.5 else None)
    card("Troponin T",f"{trop} ng/L",
         "#e74c3c" if trop>TROP_NORMAL else None)
    card("Systolic BP",f"{bps} mmHg",
         "#e74c3c" if bps<90 else ("#e67e22" if bps>160 else None))

    st.markdown("<hr style='opacity:0.2;margin:7px 0'>",unsafe_allow_html=True)
    st.caption("📊 Monitored vitals")
    card("Rhythm",hr_rhythm,
         "#e74c3c" if hr_rhythm in ["Ventricular Tachycardia","Complete Heart Block"] else
         "#e67e22" if hr_rhythm!="Normal Sinus" else None)
    card("Heart Rate",f"{hr} bpm",
         "#e74c3c" if hr>120 or hr<40 else ("#e67e22" if hr>100 or hr<60 else None))
    card("Pulse Pressure",f"{pp} mmHg",
         "#e74c3c" if pp<25 else None)
    card("Diastolic BP",f"{bps}/{bpd} mmHg")
    card("Respiratory Rate",f"{rr} br/min",
         "#e74c3c" if rr>30 else ("#e67e22" if rr>24 else None))
    card("SpO2",f"{spo2}%",
         "#e74c3c" if spo2<90 else ("#e67e22" if spo2<94 else None))
    card("Temperature",f"{temp:.1f} °C",
         "#e67e22" if temp>38.0 or temp<36.0 else None)

    st.markdown("<hr style='opacity:0.2;margin:7px 0'>",unsafe_allow_html=True)
    st.caption("🩺 Symptoms")
    syms=[("Chest pain",s_chest),("Shortness of breath",s_sob),
          ("Nausea/vomiting",s_nausea),("Diaphoresis ⚠️",s_diaphor),
          ("Syncope",s_syncope),("Indigestion",s_indig),("Epigastric pain",s_epigast)]
    active=[s for s,v in syms if v]
    for s in active: st.markdown(f"• {s}")
    if not active: st.caption("None reported")

    st.markdown("<hr style='opacity:0.2;margin:7px 0'>",unsafe_allow_html=True)
    st.caption("📉 ECG")
    ecg_color={"ST Elevation":"#e74c3c","New LBBB":"#e74c3c",
               "ST Depression":"#e67e22","T-Wave Inversion (Wellens')":"#e67e22",
               "Pathologic Q Waves":"#e67e22"}.get(ecg_feat)
    card("ECG Finding",ecg_feat+(" — "+", ".join(ecg_leads) if ecg_leads else ""),ecg_color)

    st.markdown("<hr style='opacity:0.2;margin:7px 0'>",unsafe_allow_html=True)
    st.caption("⚠️ Risk factors")
    rfs=[("Diabetes",rf_dm),("Prior MI/stent/CABG",rf_mi),("Smoking",rf_smoke),
         ("Hypertension",rf_htn),("Hyperlipidemia",rf_lipid),("Family history",rf_fam)]
    aRF=[r for r,v in rfs if v]
    for r in aRF: st.markdown(f"• {r}")
    if not aRF: st.caption("None reported")

    st.markdown("<hr style='opacity:0.2;margin:7px 0'>",unsafe_allow_html=True)
    ecg_st="✅ Patient ECG uploaded" if ecg_data is not None else "⚠️ No ECG (synthetic used)"
    st.info(ecg_st)

st.markdown("---")
st.caption("MI-Sense AI | Final Year Project — Advanced Cardiac Diagnostics v2.0")
