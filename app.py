# app.py
# 🎯 ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP)
# 팀(아군 4명) = 스크린샷 자동 인식(Gemini, 실패 시 수동 폴백)
# 후보칸 = 비율 크롭(상단 10칸 or 중앙 5칸) + DDragon 아이콘 템플릿 매칭

import os, io, re, json, requests, numpy as np, pandas as pd, streamlit as st, torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict

# gdown (있으면 사용)
try:
    import gdown
    HAS_GDOWN = True
except Exception:
    HAS_GDOWN = False

# sklearn 객체 안전 로드 등록
from torch.serialization import add_safe_globals
add_safe_globals([OrdinalEncoder])

st.set_page_config(page_title="ARAM 픽창 개인화 추천", page_icon="🎯", layout="wide")
st.title("🎯 ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP)")

LANG = "ko_KR"
LOCAL_MODEL_PATH = "model/pregame_mlp_comp.pt"
os.makedirs("model", exist_ok=True)

# ---------------- DDragon ----------------
@st.cache_data(show_spinner=False)
def ddragon_latest_version():
    try:
        return requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=10).json()[0]
    except Exception:
        return "14.14.1"

@st.cache_data(show_spinner=True)
def load_champion_static(lang=LANG):
    ver = ddragon_latest_version()
    url = f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/{lang}/champion.json"
    data = requests.get(url, timeout=20).json()["data"]
    rows = []
    for _, v in data.items():
        rows.append({
            "championId": int(v["key"]),
            "name": v["name"],
            "id": v["id"],
            "tags": v.get("tags", []),
            "icon": f"https://ddragon.leagueoflegends.com/cdn/{ver}/img/champion/{v['id']}.png",
        })
    df = pd.DataFrame(rows).sort_values("championId")
    id2name = {r.championId: r.name for r in df.itertuples()}
    id2icon = {r.championId: r.icon for r in df.itertuples()}
    id2tags = {r.championId: r.tags for r in df.itertuples()}
    name2id = {r.name: r.championId for r in df.itertuples()}
    return df, id2name, id2icon, id2tags, name2id

@st.cache_data(show_spinner=True)
def load_runes_static(lang=LANG):
    ver = ddragon_latest_version()
    url = f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/{lang}/runesReforged.json"
    data = requests.get(url, timeout=20).json()
    style_id2name, keystone_id2name = {}, {}
    for style in data:
        style_id2name[style["id"]] = style["name"]
        if style.get("slots"):
            for r in style["slots"][0].get("runes", []):
                keystone_id2name[r["id"]] = r["name"]
    return style_id2name, keystone_id2name

champ_df, id2name, id2icon, id2tags, name2id = load_champion_static()
style_id2name, keystone_id2name = load_runes_static()

# ---------------- Model loader ----------------
class CompMLP_Exact(nn.Module):
    def __init__(self, n_champ, d_champ,
                 n_sp, d_sp, n_pri, d_pri, n_sub, d_sub, n_key, d_key, n_pat, d_pat,
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
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, 1),
            )
    def forward(self, my_idx, ally_lists, enem_lists, misc_idx):
        me = self.emb_champ(my_idx)
        allies = [self.emb_champ(a) for a in ally_lists[: self.n_allies]]
        for _ in range(max(0, self.n_allies - len(allies))):
            allies.append(self.emb_champ(torch.zeros_like(my_idx)))
        enemies = [self.emb_champ(e) for e in enem_lists[: self.n_enemies]]
        for _ in range(max(0, self.n_enemies - len(enemies))):
            enemies.append(self.emb_champ(torch.zeros_like(my_idx)))
        sp  = self.emb_sp(misc_idx[:, 0]); pri = self.emb_pri(misc_idx[:, 1])
        sub = self.emb_sub(misc_idx[:, 2]); key = self.emb_key(misc_idx[:, 3])
        pat = self.emb_pat(misc_idx[:, 4]); misc = torch.cat([sp,pri,sub,key,pat], -1)
        x = torch.cat([me, *allies, *enemies, misc], -1)
        expect = None
        for mod in self.mlp:
            if isinstance(mod, nn.Linear):
                expect = int(mod.in_features); break
        if expect is not None:
            cur = int(x.size(-1))
            if cur != expect:
                if cur < expect:
                    pad = torch.zeros(x.size(0), expect-cur, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], -1)
                else:
                    x = x[..., :expect]
        return self.mlp(x).squeeze(-1)

def _infer_model_from_state(sd):
    n_champ, d_champ = sd["emb_champ.weight"].shape
    n_sp, d_sp   = sd["emb_sp.weight"].shape
    n_pri, d_pri = sd["emb_pri.weight"].shape
    n_sub, d_sub = sd["emb_sub.weight"].shape
    n_key, d_key = sd["emb_key.weight"].shape
    n_pat, d_pat = sd["emb_pat.weight"].shape
    in_dim = sd["mlp.0.weight"].shape[1]
    h1     = sd["mlp.0.weight"].shape[0]
    use_dropout = ("mlp.3.weight" in sd and "mlp.2.weight" not in sd)
    h2 = sd["mlp.3.weight"].shape[0] if use_dropout else sd["mlp.2.weight"].shape[0]
    misc_sum = d_sp + d_pri + d_sub + d_key + d_pat
    best = None
    for allies in range(0, 6):
        for enemies in range(0, 10):
            expect = d_champ*(1+allies+enemies)+misc_sum
            if expect == in_dim:
                score = -abs(allies-4)*10 + enemies
                if best is None or score > best[0]:
                    best = (score, allies, enemies)
    if best is None:
        total_slots = (in_dim - misc_sum)//d_champ
        allies, enemies = 4, max(total_slots-1-4, 0)
    else:
        _, allies, enemies = best
    return dict(n_champ=n_champ,d_champ=d_champ,n_sp=n_sp,d_sp=d_sp,n_pri=n_pri,d_pri=d_pri,
                n_sub=n_sub,d_sub=d_sub,n_key=n_key,d_key=d_key,n_pat=n_pat,d_pat=d_pat,
                in_dim=in_dim,h1=h1,h2=h2,use_dropout=use_dropout,allies=allies,enemies=enemies)

def enc_misc_row(enc: OrdinalEncoder, row: dict):
    vals = [[
        row.get("spell_pair", "__UNK__"),
        row.get("primaryStyle", "__UNK__"),
        row.get("subStyle", "__UNK__"),
        row.get("keystone", "__UNK__"),
        row.get("patch", "__UNK__"),
    ]]
    arr = enc.transform(vals).astype(int)
    for j in range(arr.shape[1]):
        if arr[0, j] < 0:
            arr[0, j] = len(enc.categories_[j])
    return torch.tensor(arr, dtype=torch.long)

def _extract_drive_file_id(url: str) -> str | None:
    if not url: return None
    for pat in [r"/d/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})"]:
        m = re.search(pat, url)
        if m: return m.group(1)
    return None

def ensure_model_file(local_path: str, url: str):
    if os.path.exists(local_path): return local_path
    if not url: return None
    fid = _extract_drive_file_id(url)
    try:
        if fid and HAS_GDOWN:
            gdown.download(f"https://drive.google.com/uc?id={fid}", local_path, quiet=False)
        else:
            dl = f"https://drive.google.com/uc?id={fid}&confirm=t" if fid else url
            with requests.get(dl, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path,"wb") as f:
                    for ch in r.iter_content(1024*1024):
                        if ch: f.write(ch)
    except Exception as e:
        st.error(f"모델 다운로드 실패: {e}"); return None
    try:
        with open(local_path,"rb") as f:
            head=f.read(32)
        if head.strip().startswith(b"<"):
            raise ValueError("HTML이 내려왔습니다.")
    except Exception as e:
        st.error(f"모델 파일 검증 실패: {e}")
        try: os.remove(local_path)
        except Exception: pass
        return None
    return local_path

@st.cache_resource(show_spinner=True)
def load_model(local_path: str):
    if not os.path.exists(local_path): return None
    obj = torch.load(local_path, map_location="cpu", weights_only=False)
    sd   = obj["state_dict"]; c2i = obj["champ_id2idx"]; enc = obj["enc_misc"]
    spec = _infer_model_from_state(sd)
    model = CompMLP_Exact(spec["n_champ"],spec["d_champ"],
                          spec["n_sp"],spec["d_sp"],spec["n_pri"],spec["d_pri"],
                          spec["n_sub"],spec["d_sub"],spec["n_key"],spec["d_key"],
                          spec["n_pat"],spec["d_pat"],spec["in_dim"],spec["h1"],
                          spec["h2"],spec["use_dropout"],spec["allies"],spec["enemies"])
    model.load_state_dict(sd); model.eval()
    return {"model":model,"champ_id2idx":c2i,"enc_misc":enc,"allies":spec["allies"],"enemies":spec["enemies"]}

@st.cache_resource(show_spinner=True)
def get_bundle():
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            with open(LOCAL_MODEL_PATH,"rb") as f:
                if f.read(32).strip().startswith(b"<"):
                    os.remove(LOCAL_MODEL_PATH)
        except Exception: pass
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            b = load_model(LOCAL_MODEL_PATH)
            if b: return b
        except Exception:
            try: os.remove(LOCAL_MODEL_PATH)
            except Exception: pass
    url = (os.environ.get("MODEL_URL","") or st.secrets.get("MODEL_URL","")).strip()
    if url:
        path = ensure_model_file(LOCAL_MODEL_PATH, url)
        if path:
            try:
                b = load_model(path)
                if b: return b
            except Exception as e:
                st.error(f"다운로드 모델 로드 실패: {e}")
    st.error("모델 준비 실패 (로컬 없음 & MODEL_URL 미설정)")
    return None

bundle = get_bundle()
if bundle: st.sidebar.success("모델 준비 완료 ✅")
else:      st.sidebar.error("모델 미로딩 ❌")

def predict_prob_comp(bundle, my_cid, ally_ids, enemy_ids, misc_row):
    if bundle is None: return 0.5
    model, c2i, enc = bundle["model"], bundle["champ_id2idx"], bundle["enc_misc"]
    na, ne = bundle.get("allies",4), bundle.get("enemies",5)
    device = torch.device("cpu"); unk_idx = len(c2i)
    def pad(ids, need):
        ids = [int(x) for x in ids][:need]
        while len(ids)<need: ids.append(0)
        return ids
    my = torch.tensor([c2i.get(int(my_cid), unk_idx)], dtype=torch.long).to(device)
    ally = torch.tensor([c2i.get(i, unk_idx) for i in pad(ally_ids, na)], dtype=torch.long).unsqueeze(0).to(device)
    enem = torch.tensor([c2i.get(i, unk_idx) for i in pad(enemy_ids, ne)], dtype=torch.long).unsqueeze(0).to(device)
    misc = enc_misc_row(enc, misc_row).to(device)
    with torch.no_grad():
        out = model(my, [ally[:,i] for i in range(ally.shape[1])], [enem[:,i] for i in range(enem.shape[1])], misc)
        prob = torch.sigmoid(out).cpu().item()
    return float(prob)

# ---------------- 개인 성향/룬/스펠 ----------------
def build_personal_stats(df: pd.DataFrame):
    if df is None or len(df)==0:
        return pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
    g = df.groupby("championId").agg(games=("win","size"), wins=("win","sum")).reset_index()
    g["wr"] = g["wins"]/g["games"]; g["personal_score"] = g["wr"] + 0.1*np.log1p(g["games"])
    return g

def per_champ_misc_modes(df: pd.DataFrame):
    if df is None or len(df)==0: return {}
    cols = ["championId","spell_pair","primaryStyle","subStyle","keystone","patch"]
    for c in cols:
        if c not in df.columns: df[c] = "__UNK__"
    modes = {}
    for cid, sub in df.groupby("championId"):
        mode = {}
        for c in ["spell_pair","primaryStyle","subStyle","keystone","patch"]:
            s = sub[c].mode(dropna=True); mode[c] = s.iloc[0] if not s.empty else "__UNK__"
        modes[int(cid)] = mode
    return modes

def suggest_spells_for_champ(cid: int, id2tags: dict, ally_ids: list[int], enemy_ids: list[int]):
    tags = set(id2tags.get(cid, [])); second = "유체화"
    if "Assassin" in tags or "Mage" in tags: second = "점화"
    if any("Assassin" in id2tags.get(e, []) for e in enemy_ids): second = "탈진"
    if "Marksman" in tags and any("Assassin" in id2tags.get(e, []) for e in enemy_ids): second = "탈진"
    return ["눈덩이", second]

def suggest_runes_from_modes(cid: int, misc_modes: dict):
    m = misc_modes.get(cid, {})
    def to_int(x):
        try: return int(x)
        except Exception: return None
    return {
        "primaryStyle": style_id2name.get(to_int(m.get("primaryStyle","")), str(m.get("primaryStyle",""))),
        "subStyle":     style_id2name.get(to_int(m.get("subStyle","")),    str(m.get("subStyle",""))),
        "keystone":     keystone_id2name.get(to_int(m.get("keystone","")), str(m.get("keystone",""))),
    }

def personal_spell_from_df(df: pd.DataFrame, cid: int, min_games: int = 3):
    if df is None or len(df)==0 or "spell_pair" not in df.columns: return None
    sub = df[(df["championId"]==cid) & (df["spell_pair"].notna())]
    if sub.empty: return None
    cnt = sub.groupby("spell_pair").size().sort_values(ascending=False)
    top_pair = cnt.index[0]
    if cnt.iloc[0] < min_games: return None
    parts = [p.strip() for p in str(top_pair).split("+") if p.strip()]
    return parts if len(parts)==2 else None

def comp_bonus_score(my_cid, ally_ids, id2tags):
    tags_me = set(id2tags.get(my_cid, []))
    allies = [set(id2tags.get(i, [])) for i in ally_ids]
    score = 0.0
    if "Tank" in tags_me and not any("Tank" in t for t in allies): score += 0.05
    if "Support" in tags_me and not any("Support" in t for t in allies): score += 0.05
    ap_like={"Mage","Support"}; ad_like={"Marksman","Fighter","Assassin"}
    ally_ap=sum(any(tt in ap_like for tt in t) for t in allies)
    ally_ad=sum(any(tt in ad_like for tt in t) for t in allies)
    if "Mage" in tags_me and ally_ap<=1: score+=0.03
    if "Marksman" in tags_me and ally_ad<=1: score+=0.03
    return min(score,0.15)

# ---------------- 수동 입력 폴백 ----------------
st.markdown("## 3) 픽창 입력(폴백)")
c1, c2 = st.columns(2)
with c1:
    ally_names_manual = st.multiselect("아군 챔피언(폴백)", champ_df["name"].tolist(), max_selections=4)
with c2:
    enemy_names_manual = st.multiselect("상대 챔피언 (선택)", champ_df["name"].tolist(), max_selections=5)

# ---------------- 후보칸 비율 크롭 ----------------
class CandidateBarConfig:
    def __init__(self, x0, x1, y0, y1, gap, slots=10, x_shift=0.0, y_shift=0.0, w_scale=1.0, h_scale=1.0):
        self.bar_x0_ratio=float(x0); self.bar_x1_ratio=float(x1)
        self.bar_y0_ratio=float(y0); self.bar_y1_ratio=float(y1)
        self.col_gap_ratio=float(gap); self.slots=int(slots)
        self.x_shift=float(x_shift); self.y_shift=float(y_shift)
        self.w_scale=float(w_scale); self.h_scale=float(h_scale)

def auto_trim_letterbox(img: Image.Image, thresh: int = 8) -> Image.Image:
    arr = np.asarray(img.convert("L")); H, W = arr.shape
    top=0
    while top<H and arr[top].mean()<thresh: top+=1
    bot=H-1
    while bot>top and arr[bot].mean()<thresh: bot-=1
    left=0
    while left<W and arr[:,left].mean()<thresh: left+=1
    right=W-1
    while right>left and arr[:,right].mean()<thresh: right-=1
    if (right-left)<W*0.5 or (bot-top)<H*0.5: return img
    return img.crop((left, top, right+1, bot+1))

def _apply_adjust(x0r,x1r,y0r,y1r,xs,ys,ws,hs):
    cx=(x0r+x1r)/2; cy=(y0r+y1r)/2
    w=(x1r-x0r)*ws; h=(y1r-y0r)*hs
    nx0=max(0.0,cx-w/2+xs); nx1=min(1.0,cx+w/2+xs)
    ny0=max(0.0,cy-h/2+ys); ny1=min(1.0,cy+h/2+ys)
    return nx0,nx1,ny0,ny1

def crop_candidate_slots(img: Image.Image, cfg: CandidateBarConfig, use_trim: bool=False)\
        -> Tuple[Image.Image, List[Image.Image], List[Tuple[int,int,int,int]]]:
    base = auto_trim_letterbox(img) if use_trim else img
    W,H = base.size
    x0r,x1r,y0r,y1r=_apply_adjust(cfg.bar_x0_ratio,cfg.bar_x1_ratio,cfg.bar_y0_ratio,cfg.bar_y1_ratio,
                                  cfg.x_shift,cfg.y_shift,cfg.w_scale,cfg.h_scale)
    bar_x0,bar_x1=int(round(W*x0r)),int(round(W*x1r))
    bar_y0,bar_y1=int(round(H*y0r)),int(round(H*y1r))
    bar_w=max(1,bar_x1-bar_x0)
    gap_px=int(round(W*cfg.col_gap_ratio))
    total_gap=gap_px*max(0,cfg.slots-1)
    slot_w=max(1,(bar_w-total_gap)//max(1,cfg.slots))
    slots,rects=[],[]
    cur_x=bar_x0
    for _ in range(cfg.slots):
        x0=cur_x; x1=x0+slot_w
        rects.append((x0,bar_y0,x1,bar_y1))
        slots.append(base.crop((x0,bar_y0,x1,bar_y1)))
        cur_x=x1+gap_px
    return base,slots,rects

def draw_overlay(base_img: Image.Image, rects: list[tuple[int,int,int,int]]):
    im=base_img.copy(); dr=ImageDraw.Draw(im,"RGBA")
    for i,(x0,y0,x1,y1) in enumerate(rects,1):
        dr.rectangle([x0,y0,x1,y1], outline=(0,255,255,220), width=3)
        dr.text((x0+4,y0+4), str(i), fill=(255,255,255,230))
    return im

# ---------------- 아이콘 매칭 ----------------
@st.cache_data(show_spinner=False)
def _download_icon_as_arr(url: str, size: int=48):
    try:
        r=requests.get(url,timeout=10); r.raise_for_status()
        im=Image.open(io.BytesIO(r.content)).convert("RGB").resize((size,size))
        return np.asarray(im).astype(np.float32)/255.0
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def build_icon_bank(size: int=48) -> Dict[str,np.ndarray]:
    bank={}
    for r in champ_df.itertuples():
        arr=_download_icon_as_arr(id2icon.get(r.championId,""), size=size)
        if arr is not None: bank[r.name]=arr
    return bank

def mse(a: np.ndarray,b: np.ndarray)->float:
    if a.shape!=b.shape: return 1e9
    d=a-b; return float(np.mean(d*d))

def predict_champion_from_icon(crop_img: Image.Image, bank: Dict[str,np.ndarray], size:int=48):
    arr=np.asarray(crop_img.convert("RGB").resize((size,size))).astype(np.float32)/255.0
    best_name,best_dist=None,1e9
    for name,icon_arr in bank.items():
        d=mse(arr,icon_arr)
        if d<best_dist: best_dist,best_name=d,name
    conf=max(0.0, 1.0 - min(best_dist,0.12)/0.12)
    return best_name,conf

# ---------------- Vertex (팀 자동 인식: 기존 로직 유지) ----------------
def init_vertex():
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception:
        return None
    project = st.secrets.get("GCP_PROJECT",""); location = st.secrets.get("GCP_LOCATION","us-central1")
    sa_json = st.secrets.get("GCP_SA_JSON","")
    if not (project and sa_json): return None
    key_path="/tmp/gcp_key.json"
    try:
        with open(key_path,"w",encoding="utf-8") as f:
            f.write(sa_json if isinstance(sa_json,str) else json.dumps(sa_json))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=key_path
        import vertexai as vx; vx.init(project=project, location=location)
        from vertexai.generative_models import GenerativeModel
    except Exception:
        return None
    prefer=(st.secrets.get("GEMINI_MODEL","") or "").strip()
    candidates=[m for m in [prefer or None, "gemini-1.5-flash","gemini-1.5-flash-001","gemini-1.0-pro-vision-001"] if m]
    for mid in candidates:
        try:
            model=GenerativeModel(mid)
            test=model.generate_content(["ping"], generation_config={"max_output_tokens":1})
            if getattr(test,"candidates",None): st.caption(f"Using Vertex model: **{mid}**"); return model
        except Exception:
            continue
    return None

def detect_allies_from_gemini(img: Image.Image) -> List[str] | None:
    model = init_vertex()
    if model is None: return None
    from vertexai.generative_models import Part
    sys_prompt = (
        'Return STRICT JSON only: {"ally_champions":["초가스","타릭","벨베스","요네"]}. '
        "Extract ONLY the four ally champions shown in the left column from a Korean ARAM pick-phase screenshot. "
        "Use exact Korean names as in the League client."
    )
    buf=io.BytesIO(); img.save(buf, format="PNG")
    resp=model.generate_content([sys_prompt, Part.from_data(mime_type="image/png", data=buf.getvalue())],
                                generation_config={"temperature":0.1,"max_output_tokens":256})
    raw=getattr(resp,"text","") or ""
    if not raw and getattr(resp,"candidates",None):
        parts=[]
        for c in resp.candidates:
            try:
                for p in getattr(c,"content",{}).parts or []:
                    if getattr(p,"text",None): parts.append(p.text)
            except Exception: pass
        raw="\n".join(parts)
    m=re.search(r"\{[\s\S]*\}", raw.strip() if raw else "")
    if not m: return None
    try:
        data=json.loads(m.group(0))
        allies=[x for x in data.get("ally_champions",[]) if x in name2id]
        return allies
    except Exception:
        return None

# ---------------- UI: CSV/가중치/진단 ----------------
st.sidebar.header("1) 내 전적 CSV")
csv_mode=st.sidebar.radio("불러오기 방식",["GitHub RAW URL","파일 업로드"],horizontal=True)
df_pre=None
if csv_mode=="GitHub RAW URL":
    url=st.sidebar.text_input("RAW CSV URL",value="")
    if url:
        try: df_pre=pd.read_csv(url); st.sidebar.success(f"CSV 로드: {len(df_pre)}행")
        except Exception as e: st.sidebar.error(f"로드 실패: {e}")
else:
    up=st.sidebar.file_uploader("Drag & drop CSV", type=["csv"])
    if up:
        try: df_pre=pd.read_csv(up); st.sidebar.success(f"CSV 로드: {len(df_pre)}행")
        except Exception as e: st.sidebar.error(f"로드 실패: {e}")

with st.sidebar.expander("추천 가중치", expanded=True):
    alpha=st.slider("α 모델 가중치",0.0,1.0,0.60,0.01)
    beta =st.slider("β 개인 성향 가중치",0.0,1.0,0.35,0.01)
    gamma=st.slider("γ 조합 보너스 가중치",0.0,0.5,0.05,0.01)
    min_games_used=st.number_input("개인 성향 최소 표본",0,50,5,step=1)

# ---------------- 스샷 인식 ----------------
st.markdown("---")
st.header("🖼️ 스크린샷 자동 추천 (팀 자동 + 후보칸 비율크롭)")

with st.sidebar.expander("후보칸 레이아웃/비율 튜닝", expanded=True):
    layout = st.selectbox("레이아웃", ["상단 10칸(작은 정사각형)", "중앙 5칸(스킨 줄)"])
    # ✅ 네 스샷 기준: 12시 맨 위 10칸 (오른쪽/아래 쏠림 보정)
    if layout == "상단 10칸(작은 정사각형)":
        defaults = dict(x0=0.355, x1=0.944, y0=0.035, y1=0.085, gap=0.008, slots=10)
    else:
        defaults = dict(x0=0.335, x1=0.735, y0=0.072, y1=0.132, gap=0.012, slots=5)

    use_trim = st.checkbox("레터박스(검은 띠) 자동 제거", value=False)

    bar_x0_ratio = st.slider("x0", 0.00, 1.00, defaults["x0"], 0.001)
    bar_x1_ratio = st.slider("x1", 0.00, 1.00, defaults["x1"], 0.001)
    bar_y0_ratio = st.slider("y0", 0.00, 1.00, defaults["y0"], 0.001)
    bar_y1_ratio = st.slider("y1", 0.00, 1.00, defaults["y1"], 0.001)
    col_gap_ratio = st.slider("칸 간격", 0.000, 0.050, defaults["gap"], 0.001)
    slot_count    = st.number_input("칸 수", 1, 12, defaults["slots"])
    icon_match_size = st.slider("아이콘 매칭 크기(px)", 32, 128, 48, 8)
    x_shift = st.slider("X 이동(±)", -0.20, 0.20, 0.00, 0.001)
    y_shift = st.slider("Y 이동(±)", -0.20, 0.20, 0.00, 0.001)
    w_scale = st.slider("가로 스케일", 0.6, 1.4, 1.00, 0.01)
    h_scale = st.slider("세로 스케일", 0.6, 1.4, 1.00, 0.01)

up_img=st.file_uploader("픽창 스크린샷 업로드 (PNG/JPG)", type=["png","jpg","jpeg"])

if up_img and st.button("🔍 스샷 인식 & 추천 (팀 자동 + 후보칸)"):
    img=Image.open(up_img).convert("RGB")

    # 1) 후보칸 오버레이
    cfg=CandidateBarConfig(bar_x0_ratio,bar_x1_ratio,bar_y0_ratio,bar_y1_ratio,
                           col_gap_ratio, slots=slot_count,
                           x_shift=x_shift,y_shift=y_shift,w_scale=w_scale,h_scale=h_scale)
    base,crops,rects=crop_candidate_slots(img,cfg,use_trim=use_trim)
    st.image(draw_overlay(base,rects), caption=f"후보칸 오버레이 (칸수={slot_count})", use_container_width=True)

    # 2) 후보칸 아이콘 매칭
    bank=build_icon_bank(size=icon_match_size)
    cand_names=[]
    for c in crops:
        name,conf=predict_champion_from_icon(c, bank, size=icon_match_size)
        if name and name in name2id and conf>=0.35:
            cand_names.append(name)
    cand_names=list(dict.fromkeys(cand_names))[:slot_count]
    if not cand_names:
        st.warning("후보칸 인식 실패: x0/x1/y0/y1, 간격을 0.002~0.004 정도 미세조정해 보세요.")
        # 후보 인식이 없어도 팀 추천은 계산할 수 없으니 중단
        st.stop()

    # 3) 팀 자동 인식(이전 코드 유지) → 실패 시 수동 선택 폴백
    ally_auto = detect_allies_from_gemini(img)
    ally_names = ally_auto[:4] if (ally_auto and len(ally_auto)>=4) else ally_names_manual
    if len(ally_names)!=4:
        st.warning("아군 4명 자동 인식 실패. '픽창 입력(폴백)'에서 4명을 선택하면 진행합니다.")
        st.stop()
    enemy_names = enemy_names_manual or []

    # 4) 추천 계산
    per = build_personal_stats(df_pre) if df_pre is not None else pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
    per_map = per.set_index("championId").to_dict(orient="index") if len(per)>0 else {}
    misc_modes = per_champ_misc_modes(df_pre) if df_pre is not None else {}
    ally_ids  = [name2id[n] for n in ally_names]
    enemy_ids = [name2id[n] for n in enemy_names] if enemy_names else []

    rows=[]
    for cname in cand_names:
        cid=name2id[cname]
        meta=per_map.get(cid,{"games":0,"wins":0,"wr":np.nan,"personal_score":-0.5})
        ps  =meta["personal_score"] - (0.3 if meta["games"]<min_games_used else 0.0)
        mode=misc_modes.get(cid,{})
        misc_row={"spell_pair":str(mode.get("spell_pair","__UNK__")),
                  "primaryStyle":str(mode.get("primaryStyle","__UNK__")),
                  "subStyle":str(mode.get("subStyle","__UNK__")),
                  "keystone":str(mode.get("keystone","__UNK__")),
                  "patch":str(mode.get("patch","__UNK__"))}
        prob = predict_prob_comp(bundle, cid, ally_ids, enemy_ids, misc_row)
        bonus= comp_bonus_score(cid, ally_ids, id2tags)
        score= alpha*prob + beta*ps + gamma*bonus
        rune = suggest_runes_from_modes(cid, misc_modes)
        spells = personal_spell_from_df(df_pre, cid, min_games=min_games_used) or \
                 suggest_spells_for_champ(cid, id2tags, ally_ids, enemy_ids)
        rows.append({"icon":id2icon.get(cid,""),"챔피언":cname,
                     "예측승률α(%)":round(prob*100,2),
                     "개인_게임수":meta.get("games",0),
                     "개인_승률(%)":round(meta.get("wr",0)*100,2) if pd.notna(meta.get("wr")) else None,
                     "추천_스펠":" + ".join(spells),
                     "추천_룬":f"주{rune['primaryStyle']} · 부{rune['subStyle']} · 핵심{rune['keystone']}",
                     "점수":score})

    out=pd.DataFrame(rows).sort_values("점수",ascending=False).reset_index(drop=True)
    st.subheader("후보 챔피언 추천 순위")
    top3=out.head(3)
    cols=st.columns(len(top3))
    for col,(_,r) in zip(cols,top3.iterrows()):
        with col:
            if r["icon"]: st.image(r["icon"], width=64)
            st.markdown(f"**{r['챔피언']}**")
            st.write(f"예측 {r['예측승률α(%)']}%")
            st.write(f"스펠: {r['추천_스펠']}"); st.write(r["추천_룬"])
    st.markdown("### 전체 표")
    table=out.drop(columns=["점수"]).copy()
    st.dataframe(table, column_config={"icon":st.column_config.ImageColumn(" ", help="챔피언 아이콘", width="small")},
                 use_container_width=True)
