# app.py
# ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP)
# 팀(아군 4명) = 스크린샷 자동 인식(Gemini) / 후보칸 = 비율크롭 + 아이콘 템플릿 매칭
# 후보칸 레이아웃: 상단 10칸(작은 정사각형) 또는 중앙 5칸(스킨 줄) 선택 가능

import os, io, re, json, requests, numpy as np, pandas as pd, streamlit as st, torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict

# gdown(선택)
try:
    import gdown
    HAS_GDOWN = True
except Exception:
    HAS_GDOWN = False

# sklearn 객체 로더 허용
from torch.serialization import add_safe_globals
add_safe_globals([OrdinalEncoder])

st.set_page_config(page_title="ARAM 픽창 개인화 추천", page_icon="🎯", layout="wide")
st.title("🎯 ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP)")

# ----------------------------
# 기본 설정
# ----------------------------
LANG = "ko_KR"
LOCAL_MODEL_PATH = "model/pregame_mlp_comp.pt"
os.makedirs("model", exist_ok=True)

# ----------------------------
# Data Dragon 정적 정보
# ----------------------------
@st.cache_data(show_spinner=False)
def ddragon_latest_version():
    try:
        return requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=10).json()[0]
    except Exception:
        return "14.14.1"

@st.cache_data(show_spinner=True)
def load_champion_static(lang=LANG):
    ver = ddragon_latest_version()
    data = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/{lang}/champion.json", timeout=20).json()["data"]
    rows=[]
    for _, v in data.items():
        rows.append({
            "championId": int(v["key"]),
            "name": v["name"],
            "id": v["id"],
            "tags": v.get("tags", []),
            "icon": f"https://ddragon.leagueoflegends.com/cdn/{ver}/img/champion/{v['id']}.png",
        })
    df = pd.DataFrame(rows).sort_values("championId")
    id2name = {r.championId:r.name for r in df.itertuples()}
    id2icon = {r.championId:r.icon for r in df.itertuples()}
    id2tags = {r.championId:r.tags for r in df.itertuples()}
    name2id = {r.name:r.championId for r in df.itertuples()}
    return df, id2name, id2icon, id2tags, name2id

@st.cache_data(show_spinner=True)
def load_runes_static(lang=LANG):
    ver = ddragon_latest_version()
    data = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/{lang}/runesReforged.json", timeout=20).json()
    style_id2name, keystone_id2name = {}, {}
    for style in data:
        style_id2name[style["id"]] = style["name"]
        for r in style.get("slots", [{}])[0].get("runes", []):
            keystone_id2name[r["id"]] = r["name"]
    return style_id2name, keystone_id2name

champ_df, id2name, id2icon, id2tags, name2id = load_champion_static()
style_id2name, keystone_id2name = load_runes_static()

# ----------------------------
# 모델 로더
# ----------------------------
class CompMLP_Exact(nn.Module):
    def __init__(self, n_champ, d_champ, n_sp, d_sp, n_pri, d_pri, n_sub, d_sub, n_key, d_key, n_pat, d_pat,
                 in_dim, h1, h2, use_dropout, allies, enemies):
        super().__init__()
        self.emb_champ = nn.Embedding(n_champ, d_champ)
        self.emb_sp  = nn.Embedding(n_sp,  d_sp)
        self.emb_pri = nn.Embedding(n_pri, d_pri)
        self.emb_sub = nn.Embedding(n_sub, d_sub)
        self.emb_key = nn.Embedding(n_key, d_key)
        self.emb_pat = nn.Embedding(n_pat, d_pat)
        self.n_allies, self.n_enemies = allies, enemies
        if use_dropout:
            self.mlp = nn.Sequential(nn.Linear(in_dim,h1), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(h1,h2), nn.ReLU(), nn.Linear(h2,1))
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim,h1), nn.ReLU(),
                                     nn.Linear(h1,h2), nn.ReLU(), nn.Linear(h2,1))

    def forward(self, my_idx, ally_lists, enem_lists, misc_idx):
        me = self.emb_champ(my_idx)
        allies = [self.emb_champ(a) for a in ally_lists[: self.n_allies]]
        for _ in range(max(0, self.n_allies-len(allies))): allies.append(self.emb_champ(torch.zeros_like(my_idx)))
        enemies= [self.emb_champ(e) for e in enem_lists[: self.n_enemies]]
        for _ in range(max(0, self.n_enemies-len(enemies))): enemies.append(self.emb_champ(torch.zeros_like(my_idx)))
        sp=self.emb_sp(misc_idx[:,0]); pri=self.emb_pri(misc_idx[:,1]); sub=self.emb_sub(misc_idx[:,2])
        key=self.emb_key(misc_idx[:,3]); pat=self.emb_pat(misc_idx[:,4])
        x = torch.cat([me,*allies,*enemies, sp,pri,sub,key,pat], dim=-1)
        expect = getattr(self.mlp[0], "in_features", None)
        if expect is not None and x.size(-1)!=expect:
            if not hasattr(self,"_dim_warned"):
                st.warning(f"[입력 차원 보정] cur={x.size(-1)} expect={expect} (allies={self.n_allies}, enemies={self.n_enemies})")
                self._dim_warned=True
            if x.size(-1)<expect:
                x = torch.cat([x, torch.zeros(x.size(0), expect-x.size(-1), dtype=x.dtype)], dim=-1)
            else:
                x = x[...,:expect]
        return self.mlp(x).squeeze(-1)

def _infer_model_from_state(sd):
    n_champ,d_champ = sd["emb_champ.weight"].shape
    n_sp,d_sp       = sd["emb_sp.weight"].shape
    n_pri,d_pri     = sd["emb_pri.weight"].shape
    n_sub,d_sub     = sd["emb_sub.weight"].shape
    n_key,d_key     = sd["emb_key.weight"].shape
    n_pat,d_pat     = sd["emb_pat.weight"].shape
    in_dim = sd["mlp.0.weight"].shape[1]; h1 = sd["mlp.0.weight"].shape[0]
    use_dropout = ("mlp.3.weight" in sd and "mlp.2.weight" not in sd)
    h2 = sd["mlp.3.weight"].shape[0] if use_dropout else sd["mlp.2.weight"].shape[0]
    misc_sum = d_sp+d_pri+d_sub+d_key+d_pat
    best=None
    for a in range(0,6):
        for e in range(0,10):
            if d_champ*(1+a+e)+misc_sum==in_dim:
                score = -abs(a-4)*10 + e
                best = max(best or (-1,0,0), (score,a,e))
    if best is None:
        total=(in_dim-misc_sum)//d_champ; a=4; e=max(total-1-a,0)
    else:
        _,a,e = best
    return dict(n_champ=n_champ,d_champ=d_champ,n_sp=n_sp,d_sp=d_sp,n_pri=n_pri,d_pri=d_pri,
                n_sub=n_sub,d_sub=d_sub,n_key=n_key,d_key=d_key,n_pat=n_pat,d_pat=d_pat,
                in_dim=in_dim,h1=h1,h2=h2,use_dropout=use_dropout,allies=a,enemies=e)

def enc_misc_row(enc: OrdinalEncoder, row: dict):
    vals=[[row.get("spell_pair","__UNK__"),row.get("primaryStyle","__UNK__"),
           row.get("subStyle","__UNK__"),row.get("keystone","__UNK__"),
           row.get("patch","__UNK__")]]
    arr=enc.transform(vals).astype(int)
    for j in range(arr.shape[1]):
        if arr[0,j]<0: arr[0,j]=len(enc.categories_[j])
    return torch.tensor(arr,dtype=torch.long)

def _extract_drive_file_id(url:str)->str|None:
    if not url: return None
    for pat in [r"/d/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})"]:
        m=re.search(pat,url); 
        if m: return m.group(1)
    return None

def ensure_model_file(local_path: str, url: str):
    if os.path.exists(local_path): return local_path
    if not url: return None
    fid=_extract_drive_file_id(url)
    try:
        if fid and HAS_GDOWN:
            gdown.download(f"https://drive.google.com/uc?id={fid}", local_path, quiet=False)
        else:
            dl=f"https://drive.google.com/uc?id={fid}&confirm=t" if fid else url
            with requests.get(dl, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path,"wb") as f:
                    for ch in r.iter_content(1024*1024):
                        if ch: f.write(ch)
    except Exception as e:
        st.error(f"모델 다운로드 실패: {e}"); return None
    try:
        with open(local_path,"rb") as f:
            if f.read(32).strip().startswith(b"<"):
                raise ValueError("HTML 응답")
    except Exception as e:
        st.error(f"모델 파일 검증 실패: {e}")
        try: os.remove(local_path)
        except: pass
        return None
    return local_path

@st.cache_resource(show_spinner=True)
def load_model(local_path:str):
    if not os.path.exists(local_path): return None
    obj=torch.load(local_path,map_location="cpu",weights_only=False)
    sd, c2i, enc = obj["state_dict"], obj["champ_id2idx"], obj["enc_misc"]
    spec=_infer_model_from_state(sd)
    m=CompMLP_Exact(spec["n_champ"],spec["d_champ"],spec["n_sp"],spec["d_sp"],spec["n_pri"],spec["d_pri"],
                    spec["n_sub"],spec["d_sub"],spec["n_key"],spec["d_key"],spec["n_pat"],spec["d_pat"],
                    spec["in_dim"],spec["h1"],spec["h2"],spec["use_dropout"],spec["allies"],spec["enemies"])
    m.load_state_dict(sd); m.eval()
    return {"model":m,"champ_id2idx":c2i,"enc_misc":enc,"allies":spec["allies"],"enemies":spec["enemies"]}

@st.cache_resource(show_spinner=True)
def get_bundle():
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            with open(LOCAL_MODEL_PATH,"rb") as f:
                if f.read(32).strip().startswith(b"<"):
                    os.remove(LOCAL_MODEL_PATH)
        except: pass
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            b=load_model(LOCAL_MODEL_PATH); 
            if b: return b
        except Exception as e:
            try: os.remove(LOCAL_MODEL_PATH)
            except: pass
            st.warning(f"로컬 모델 로드 실패: {e}")
    url=os.environ.get("MODEL_URL","") or st.secrets.get("MODEL_URL","")
    if url:
        p=ensure_model_file(LOCAL_MODEL_PATH,url)
        if p:
            try:
                b=load_model(p); 
                if b: return b
            except Exception as e:
                st.error(f"다운로드 모델 로드 실패: {e}")
    st.error("모델 준비 실패 (로컬 파일 없음 & MODEL_URL 미설정)")
    return None

bundle=get_bundle()
if bundle: st.sidebar.success("모델 준비 완료 ✅")
else:      st.sidebar.error("모델 미로딩 ❌")

def predict_prob_comp(bundle, my_cid, ally_ids, enemy_ids, misc_row):
    if bundle is None: return 0.5
    m, c2i, enc = bundle["model"], bundle["champ_id2idx"], bundle["enc_misc"]
    na, ne = bundle.get("allies",4), bundle.get("enemies",5)
    unk=len(c2i)
    def pad(ids, n):
        ids=[int(x) for x in ids][:n]
        while len(ids)<n: ids.append(0)
        return ids
    my=torch.tensor([c2i.get(int(my_cid),unk)],dtype=torch.long)
    ally=torch.tensor([c2i.get(i,unk) for i in pad(ally_ids,na)],dtype=torch.long).unsqueeze(0)
    enem=torch.tensor([c2i.get(i,unk) for i in pad(enemy_ids,ne)],dtype=torch.long).unsqueeze(0)
    misc=enc_misc_row(enc, misc_row)
    with torch.no_grad():
        out=m(my,[ally[:,i] for i in range(ally.shape[1])],[enem[:,i] for i in range(enem.shape[1])],misc)
        return float(torch.sigmoid(out).item())

# ----------------------------
# 내 전적 CSV
# ----------------------------
st.sidebar.header("1) 내 전적 CSV")
mode = st.sidebar.radio("불러오기 방식", ["GitHub RAW URL", "파일 업로드"], horizontal=True)
df_pre=None
if mode=="GitHub RAW URL":
    u=st.sidebar.text_input("RAW CSV URL", value="")
    if u:
        try: df_pre=pd.read_csv(u); st.sidebar.success(f"{len(df_pre)}행 로드")
        except Exception as e: st.sidebar.error(f"로드 실패: {e}")
else:
    up=st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    if up:
        try: df_pre=pd.read_csv(up); st.sidebar.success(f"{len(df_pre)}행 로드")
        except Exception as e: st.sidebar.error(f"로드 실패: {e}")

# ----------------------------
# 개인 성향 등 유틸
# ----------------------------
def build_personal_stats(df: pd.DataFrame):
    if df is None or len(df)==0: return pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
    g=df.groupby("championId").agg(games=("win","size"), wins=("win","sum")).reset_index()
    g["wr"]=g["wins"]/g["games"]; g["personal_score"]=g["wr"]+0.1*np.log1p(g["games"]); return g

def per_champ_misc_modes(df: pd.DataFrame):
    if df is None or len(df)==0: return {}
    need=["championId","spell_pair","primaryStyle","subStyle","keystone","patch"]
    for c in need:
        if c not in df.columns: df[c]="__UNK__"
    modes={}
    for cid, sub in df.groupby("championId"):
        m={}
        for c in need[1:]:
            s=sub[c].mode(dropna=True); m[c]= s.iloc[0] if not s.empty else "__UNK__"
        modes[int(cid)]=m
    return modes

def suggest_spells_for_champ(cid:int, id2tags:dict, ally_ids:list[int], enemy_ids:list[int]):
    tags=set(id2tags.get(cid, [])); second="유체화"
    if "Assassin" in tags or "Mage" in tags: second="점화"
    if any("Assassin" in id2tags.get(e, []) for e in enemy_ids): second="탈진"
    if "Marksman" in tags and any("Assassin" in id2tags.get(e, []) for e in enemy_ids): second="탈진"
    return ["눈덩이", second]

def suggest_runes_from_modes(cid:int, misc_modes:dict):
    m=misc_modes.get(cid,{})
    def to_int(x):
        try: return int(x)
        except: return None
    return {
        "primaryStyle": style_id2name.get(to_int(m.get("primaryStyle","")), str(m.get("primaryStyle",""))),
        "subStyle":     style_id2name.get(to_int(m.get("subStyle","")),    str(m.get("subStyle",""))),
        "keystone":     keystone_id2name.get(to_int(m.get("keystone","")), str(m.get("keystone","")))
    }

def personal_spell_from_df(df: pd.DataFrame, cid:int, min_games:int=3):
    if df is None or len(df)==0 or "spell_pair" not in df.columns: return None
    sub=df[(df["championId"]==cid) & (df["spell_pair"].notna())]
    if sub.empty: return None
    cnt=sub.groupby("spell_pair").size().sort_values(ascending=False)
    if cnt.iloc[0] < min_games: return None
    parts=[p.strip() for p in str(cnt.index[0]).split("+") if p.strip()]
    return parts if len(parts)==2 else None

def comp_bonus_score(my_cid, ally_ids, id2tags):
    tags_me=set(id2tags.get(my_cid, []))
    allies=[set(id2tags.get(i, [])) for i in ally_ids]
    s=0.0
    if "Tank" in tags_me and not any("Tank" in t for t in allies): s+=0.05
    if "Support" in tags_me and not any("Support" in t for t in allies): s+=0.05
    ap={"Mage","Support"}; ad={"Marksman","Fighter","Assassin"}
    apn=sum(any(t in ap for t in a) for a in allies); adn=sum(any(t in ad for t in a) for a in allies)
    if "Mage" in tags_me and apn<=1: s+=0.03
    if "Marksman" in tags_me and adn<=1: s+=0.03
    return min(s,0.15)

# ----------------------------
# 수동 입력(폴백) 섹션
# ----------------------------
st.markdown("## 3) 픽창 입력(폴백)")
c1,c2=st.columns(2)
with c1:
    ally_names_manual = st.multiselect("아군 챔피언(폴백)", champ_df["name"].tolist(), max_selections=4)
with c2:
    enemy_names_manual = st.multiselect("상대 챔피언(선택)", champ_df["name"].tolist(), max_selections=5)
st.session_state["ally_names_manual"]=ally_names_manual
st.session_state["enemy_names_manual"]=enemy_names_manual

# ----------------------------
# 후보칸 비율크롭 + 아이콘 매칭 + 팀 자동 인식
# ----------------------------
st.markdown("---")
st.header("🖼️ 스크린샷 자동 추천 (팀 자동 + 후보칸 비율크롭)")

# 레이아웃 선택 + 기본값(네 스샷 기준 상단 10칸)
with st.sidebar.expander("후보칸 레이아웃/비율 튜닝", expanded=True):
    layout = st.selectbox("레이아웃", ["상단 10칸(작은 정사각형)", "중앙 5칸(스킨 줄)"])
    if layout == "상단 10칸(작은 정사각형)":
        default = dict(x0=0.37, x1=0.86, y0=0.060, y1=0.118, gap=0.008, slots=10)  # ← 이 스샷 기준
    else:
        default = dict(x0=0.335, x1=0.735, y0=0.072, y1=0.132, gap=0.012, slots=5)

    bar_x0_ratio = st.slider("x0", 0.00, 1.00, default["x0"], 0.001)
    bar_x1_ratio = st.slider("x1", 0.00, 1.00, default["x1"], 0.001)
    bar_y0_ratio = st.slider("y0", 0.00, 1.00, default["y0"], 0.001)
    bar_y1_ratio = st.slider("y1", 0.00, 1.00, default["y1"], 0.001)
    col_gap_ratio= st.slider("칸 간격",0.000,0.050, default["gap"], 0.001)
    slot_count    = st.number_input("칸 수", 1, 12, default["slots"])
    icon_match_size = st.slider("아이콘 매칭 크기(px)", 32, 128, 48, 8)
    x_shift = st.slider("X 이동(±)", -0.20, 0.20, 0.00, 0.001)
    y_shift = st.slider("Y 이동(±)", -0.20, 0.20, 0.00, 0.001)
    w_scale = st.slider("가로 스케일", 0.6, 1.4, 1.00, 0.01)
    h_scale = st.slider("세로 스케일", 0.6, 1.4, 1.00, 0.01)

class CandidateBarConfig:
    def __init__(self, x0,x1,y0,y1,gap, slots=10, x_shift=0.0,y_shift=0.0,w_scale=1.0,h_scale=1.0):
        self.bar_x0_ratio=x0; self.bar_x1_ratio=x1; self.bar_y0_ratio=y0; self.bar_y1_ratio=y1
        self.col_gap_ratio=gap; self.slots=int(slots)
        self.x_shift=x_shift; self.y_shift=y_shift; self.w_scale=w_scale; self.h_scale=h_scale

def auto_trim_letterbox(img: Image.Image, thresh: int=8)->Image.Image:
    arr=np.asarray(img.convert("L")); H,W=arr.shape
    top=0;   while top<H and arr[top].mean()<thresh: top+=1
    bot=H-1; while bot>top and arr[bot].mean()<thresh: bot-=1
    left=0;  while left<W and arr[:,left].mean()<thresh: left+=1
    right=W-1; 
    while right>left and arr[:,right].mean()<thresh: right-=1
    if right-left < W*0.5 or bot-top < H*0.5: return img
    return img.crop((left,top,right+1,bot+1))

def _apply_adjust(x0r,x1r,y0r,y1r,xs,ys,ws,hs):
    cx=(x0r+x1r)/2; cy=(y0r+y1r)/2; w=(x1r-x0r)*ws; h=(y1r-y0r)*hs
    nx0=cx-w/2+xs; nx1=cx+w/2+xs; ny0=cy-h/2+ys; ny1=cy+h/2+ys
    return max(0.0,nx0), min(1.0,nx1), max(0.0,ny0), min(1.0,ny1)

def crop_candidate_slots(img:Image.Image, cfg:CandidateBarConfig)->Tuple[Image.Image,List[Image.Image],List[Tuple[int,int,int,int]]]:
    base=auto_trim_letterbox(img); W,H=base.size
    x0r,x1r,y0r,y1r=_apply_adjust(cfg.bar_x0_ratio,cfg.bar_x1_ratio,cfg.bar_y0_ratio,cfg.bar_y1_ratio,
                                  cfg.x_shift,cfg.y_shift,cfg.w_scale,cfg.h_scale)
    x0=int(round(W*x0r)); x1=int(round(W*x1r))
    y0=int(round(H*y0r)); y1=int(round(H*y1r))
    bar_w=max(1, x1-x0)
    gap=int(round(W*cfg.col_gap_ratio))
    total_gap=gap*(cfg.slots-1)
    slot_w=max(1, (bar_w - total_gap)//cfg.slots)
    slots=[]; rects=[]; cur=x0
    for _ in range(cfg.slots):
        r=(cur, y0, cur+slot_w, y1)
        rects.append(r); slots.append(base.crop(r))
        cur = r[2] + gap
    return base, slots, rects

def draw_overlay(img:Image.Image, rects:list[tuple[int,int,int,int]]):
    im=img.copy(); dr=ImageDraw.Draw(im, "RGBA")
    for i,(x0,y0,x1,y1) in enumerate(rects,1):
        dr.rectangle([x0,y0,x1,y1], outline=(0,255,255,220), width=3)
        dr.text((x0+4,y0+4), f"{i}", fill=(255,255,255,230))
    return im

@st.cache_data(show_spinner=False)
def _download_icon_as_arr(url:str, size:int=48):
    try:
        r=requests.get(url,timeout=10); r.raise_for_status()
        im=Image.open(io.BytesIO(r.content)).convert("RGB").resize((size,size))
        return np.asarray(im).astype(np.float32)/255.0
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def build_icon_bank(size:int=48)->Dict[str,np.ndarray]:
    bank={}
    for r in champ_df.itertuples():
        arr=_download_icon_as_arr(id2icon.get(r.championId,""), size=size)
        if arr is not None: bank[r.name]=arr
    return bank

def mse(a:np.ndarray,b:np.ndarray)->float:
    if a.shape!=b.shape: return 1e9
    d=a-b; return float(np.mean(d*d))

def predict_champion_from_icon(crop:Image.Image, bank:Dict[str,np.ndarray], size:int=48):
    arr=np.asarray(crop.convert("RGB").resize((size,size))).astype(np.float32)/255.0
    best,dist=None,1e9
    for name,t in bank.items():
        d=mse(arr,t)
        if d<dist: dist, best = d, name
    conf=max(0.0, 1.0 - min(dist,0.12)/0.12)  # 상단칸 작아서 임계 완화
    return best, conf

# --- 팀(아군) 자동 인식(Gemini)
def init_vertex():
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception:
        return None
    proj=st.secrets.get("GCP_PROJECT",""); loc=st.secrets.get("GCP_LOCATION","us-central1"); sa=st.secrets.get("GCP_SA_JSON","")
    if not (proj and sa): return None
    try:
        with open("/tmp/gcp_key.json","w",encoding="utf-8") as f: f.write(sa)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/tmp/gcp_key.json"
        import vertexai; vertexai.init(project=proj, location=loc)
        from vertexai.generative_models import GenerativeModel
        return GenerativeModel(st.secrets.get("GEMINI_MODEL","gemini-1.5-flash-002"))
    except Exception:
        return None

def detect_allies_from_gemini(img:Image.Image)->List[str]|None:
    m=init_vertex()
    if m is None: return None
    from vertexai.generative_models import Part
    sys_prompt=("Return JSON only like {\"ally_champions\":[\"초가스\",\"타릭\",\"벨베스\",\"요네\"]} "
                "from a Korean ARAM pick-phase screenshot. Extract ONLY the four ally champion names from the left column.")
    buf=io.BytesIO(); img.save(buf,format="PNG")
    resp=m.generate_content([sys_prompt, Part.from_data(mime_type="image/png", data=buf.getvalue())],
                            generation_config={"temperature":0.1, "max_output_tokens":256})
    raw=getattr(resp,"text","") or ""
    if not raw and getattr(resp,"candidates",None):
        for c in resp.candidates:
            for p in getattr(c,"content",{}).parts or []:
                if getattr(p,"text",None): raw+=p.text
    m=re.search(r"\{[\s\S]*\}", raw.strip() if raw else "")
    if not m: return None
    try:
        data=json.loads(m.group(0)); allies=[x for x in data.get("ally_champions", []) if x in name2id]
        return allies
    except Exception:
        return None

# ===== 업로더 + 실행 =====
up_img = st.file_uploader("픽창 스크린샷 업로드 (PNG/JPG)", type=["png","jpg","jpeg"])
if up_img and st.button("🔍 스샷 인식 & 추천 (팀 자동 + 후보칸)"):
    img=Image.open(up_img).convert("RGB")

    # 후보칸 인식
    cfg=CandidateBarConfig(bar_x0_ratio,bar_x1_ratio,bar_y0_ratio,bar_y1_ratio,col_gap_ratio,
                           slots=slot_count, x_shift=x_shift,y_shift=y_shift,w_scale=w_scale,h_scale=h_scale)
    base, crops, rects = crop_candidate_slots(img, cfg)
    st.image(draw_overlay(base, rects), caption=f"후보칸 오버레이 (칸수={slot_count})", use_container_width=True)

    bank=build_icon_bank(size=icon_match_size)
    cand_names=[]
    for c in crops:
        name, conf = predict_champion_from_icon(c, bank, size=icon_match_size)
        # 너무 빈 칸(검은/하늘색 바 등)을 걸러내기 위한 간단 기준
        if name and name in name2id and conf >= 0.35:
            cand_names.append(name)
    # 중복제거 후 상한(10 또는 5)에 맞춰 자르기
    cand_names = list(dict.fromkeys(cand_names))[:slot_count]
    if not cand_names:
        st.warning("후보칸 인식 실패: 비율/간격 슬라이더를 조금 조정해 보세요.")
        st.stop()

    # 팀(아군) 자동 인식 → 실패 시 폴백
    ally_auto = detect_allies_from_gemini(img)
    if ally_auto and len(ally_auto) >= 4:
        ally_names = ally_auto[:4]
        st.success(f"아군 자동 인식: {ally_names}")
    else:
        ally_names = st.session_state.get("ally_names_manual", [])
        if len(ally_names) != 4:
            st.warning("아군 4명 자동 인식 실패. '픽창 입력(폴백)'에서 4명을 선택하세요.")
            st.stop()

    enemy_names = st.session_state.get("enemy_names_manual", [])

    # ===== 추천 계산 =====
    ally_ids=[name2id[n] for n in ally_names]
    enemy_ids=[name2id[n] for n in enemy_names] if enemy_names else []
    per=build_personal_stats(df_pre) if df_pre is not None else pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
    per_map=per.set_index("championId").to_dict(orient="index") if len(per)>0 else {}
    misc_modes=per_champ_misc_modes(df_pre) if df_pre is not None else {}
    alpha=st.sidebar.slider("α 모델 가중치",0.0,1.0,0.60,0.01)
    beta =st.sidebar.slider("β 개인 성향 가중치",0.0,1.0,0.35,0.01)
    gamma=st.sidebar.slider("γ 조합 보너스 가중치",0.0,0.5,0.05,0.01)
    min_games_used=st.sidebar.number_input("개인 성향 최소 표본",0,50,5,step=1)

    rows=[]
    for cname in cand_names:
        cid=name2id[cname]
        meta=per_map.get(cid, {"games":0,"wins":0,"wr":np.nan,"personal_score":-0.5})
        ps=meta["personal_score"] - (0.3 if meta["games"]<min_games_used else 0.0)
        mode=misc_modes.get(cid,{})
        misc_row={"spell_pair":str(mode.get("spell_pair","__UNK__")),
                  "primaryStyle":str(mode.get("primaryStyle","__UNK__")),
                  "subStyle":str(mode.get("subStyle","__UNK__")),
                  "keystone":str(mode.get("keystone","__UNK__")),
                  "patch":str(mode.get("patch","__UNK__"))}
        prob=predict_prob_comp(bundle, cid, ally_ids, enemy_ids, misc_row)
        bonus=comp_bonus_score(cid, ally_ids, id2tags)
        rune=suggest_runes_from_modes(cid, misc_modes)
        spells=personal_spell_from_df(df_pre, cid, min_games=min_games_used) or \
               suggest_spells_for_champ(cid, id2tags, ally_ids, enemy_ids)
        rows.append({
            "icon": id2icon.get(cid,""),
            "챔피언": cname,
            "예측승률α(%)": round(prob*100,2),
            "개인_게임수": meta.get("games",0),
            "개인_승률(%)": round(meta.get("wr",0)*100,2) if pd.notna(meta.get("wr")) else None,
            "추천_스펠": " + ".join(spells),
            "추천_룬": f"주{rune['primaryStyle']} · 부{rune['subStyle']} · 핵심{rune['keystone']}",
            "점수": alpha*prob + beta*ps + gamma*bonus
        })

    out=pd.DataFrame(rows).sort_values("점수",ascending=False).reset_index(drop=True)
    st.subheader("후보 챔피언 추천 순위")
    top3=out.head(3); cols=st.columns(len(top3))
    for col,(_,r) in zip(cols, top3.iterrows()):
        with col:
            if r["icon"]: st.image(r["icon"], width=64)
            st.markdown(f"**{r['챔피언']}**")
            st.write(f"예측 {r['예측승률α(%)']}%")
            st.write(f"스펠: {r['추천_스펠']}"); st.write(r["추천_룬"])

    st.markdown("### 전체 표")
    st.dataframe(out.drop(columns=["점수"]),
                 column_config={"icon": st.column_config.ImageColumn(" ", help="챔피언 아이콘", width="small")},
                 use_container_width=True)
