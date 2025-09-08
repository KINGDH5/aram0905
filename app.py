# app.py
# ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP)
# 팀(아군 4명): 스크린샷 자동 인식(Gemini)  / 후보 5칸: 비율 크롭 + 아이콘 템플릿 매칭

import os, io, re, json, requests, numpy as np, pandas as pd, streamlit as st, torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict

# -------- gdown (있으면 사용, 없으면 requests 폴백)
try:
    import gdown
    HAS_GDOWN = True
except Exception:
    HAS_GDOWN = False

# -------- sklearn 객체 안전 로드
from torch.serialization import add_safe_globals
add_safe_globals([OrdinalEncoder])

st.set_page_config(page_title="ARAM 픽창 개인화 추천", page_icon="🎯", layout="wide")
st.title("🎯 ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP)")

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
LANG = "ko_KR"
LOCAL_MODEL_PATH = "model/pregame_mlp_comp.pt"
os.makedirs("model", exist_ok=True)

# ==================================================================
# Data Dragon 정적 정보 (챔피언/룬)
# ==================================================================
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

# ==================================================================
# 체크포인트 모양을 그대로 복원하는 모델 로더
# ==================================================================
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
                nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1),
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
        sub = self.emb_sub(misc_idx[:, 2]); key = self.emb_key(misc_idx[:, 3]); pat = self.emb_pat(misc_idx[:, 4])
        misc = torch.cat([sp, pri, sub, key, pat], dim=-1)
        x = torch.cat([me, *allies, *enemies, misc], dim=-1)
        try:
            expect = int(self.mlp[0].in_features)
        except Exception:
            expect = None
            for mod in self.mlp:
                if isinstance(mod, torch.nn.Linear):
                    expect = int(mod.in_features); break
        if expect is not None:
            cur = int(x.size(-1))
            if cur != expect:
                if not hasattr(self, "_dim_warned"):
                    st.warning(f"[입력 차원 자동 보정] cur={cur} expect={expect} (allies={self.n_allies}, enemies={self.n_enemies})")
                    self._dim_warned = True
                if cur < expect:
                    pad = torch.zeros(x.size(0), expect - cur, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=-1)
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
            expect = d_champ * (1 + allies + enemies) + misc_sum
            if expect == in_dim:
                score = -abs(allies - 4) * 10 + enemies
                cand = (score, allies, enemies)
                if best is None or cand > best:
                    best = cand
    if best is None:
        total_slots = (in_dim - misc_sum) // d_champ
        allies = 4; enemies = max(total_slots - 1 - allies, 0)
    else:
        allies, enemies = best[1], best[2]
    return dict(n_champ=n_champ, d_champ=d_champ, n_sp=n_sp, d_sp=d_sp, n_pri=n_pri, d_pri=d_pri,
                n_sub=n_sub, d_sub=d_sub, n_key=n_key, d_key=d_key, n_pat=n_pat, d_pat=d_pat,
                in_dim=in_dim, h1=h1, h2=h2, use_dropout=use_dropout, allies=allies, enemies=enemies)

def enc_misc_row(enc: OrdinalEncoder, row: dict):
    vals = [[row.get("spell_pair","__UNK__"), row.get("primaryStyle","__UNK__"),
             row.get("subStyle","__UNK__"), row.get("keystone","__UNK__"),
             row.get("patch","__UNK__")]]
    arr = enc.transform(vals).astype(int)
    for j in range(arr.shape[1]):
        if arr[0,j] < 0: arr[0,j] = len(enc.categories_[j])
    return torch.tensor(arr, dtype=torch.long)

# ---- Google Drive 헬퍼
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
            dl_url = f"https://drive.google.com/uc?id={fid}&confirm=t" if fid else url
            with requests.get(dl_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk: f.write(chunk)
    except Exception as e:
        st.error(f"모델 다운로드 실패: {e}"); return None
    try:
        with open(local_path, "rb") as f: head = f.read(32)
        if head.strip().startswith(b"<"): raise ValueError("다운로드된 내용이 HTML입니다.")
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
    state_dict, champ_id2idx, enc_misc = obj["state_dict"], obj["champ_id2idx"], obj["enc_misc"]
    spec = _infer_model_from_state(state_dict)
    model = CompMLP_Exact(spec["n_champ"], spec["d_champ"],
                          spec["n_sp"], spec["d_sp"], spec["n_pri"], spec["d_pri"],
                          spec["n_sub"], spec["d_sub"], spec["n_key"], spec["d_key"],
                          spec["n_pat"], spec["d_pat"], spec["in_dim"], spec["h1"],
                          spec["h2"], spec["use_dropout"], spec["allies"], spec["enemies"])
    model.load_state_dict(state_dict); model.eval()
    return {"model": model, "champ_id2idx": champ_id2idx, "enc_misc": enc_misc,
            "allies": spec["allies"], "enemies": spec["enemies"]}

@st.cache_resource(show_spinner=True)
def get_bundle():
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            with open(LOCAL_MODEL_PATH, "rb") as f:
                if f.read(32).strip().startswith(b"<"): os.remove(LOCAL_MODEL_PATH)
        except Exception: pass
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            b = load_model(LOCAL_MODEL_PATH)
            if b: return b
        except Exception as e:
            try: os.remove(LOCAL_MODEL_PATH)
            except Exception: pass
            st.warning(f"로컬 모델 로드 실패, URL 재시도: {e}")
    url = os.environ.get("MODEL_URL", "")
    if not url and "MODEL_URL" in st.secrets:
        url = st.secrets["MODEL_URL"].strip()
    if url:
        path = ensure_model_file(LOCAL_MODEL_PATH, url)
        if path:
            try:
                b = load_model(path)
                if b: return b
            except Exception as e:
                st.error(f"다운로드 모델 로드 실패: {e}")
    st.error("모델 준비 실패 (로컬 파일 없음 & MODEL_URL 미설정)")
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
        while len(ids) < need: ids.append(0)
        return ids
    my = torch.tensor([c2i.get(int(my_cid), unk_idx)], dtype=torch.long).to(device)
    ally = torch.tensor([c2i.get(i, unk_idx) for i in pad(ally_ids, na)], dtype=torch.long).unsqueeze(0).to(device)
    enem = torch.tensor([c2i.get(i, unk_idx) for i in pad(enemy_ids, ne)], dtype=torch.long).unsqueeze(0).to(device)
    misc = enc_misc_row(enc, misc_row).to(device)
    with torch.no_grad():
        out = model(my, [ally[:,i] for i in range(ally.shape[1])],
                    [enem[:,i] for i in range(enem.shape[1])], misc)
        prob = torch.sigmoid(out).cpu().item()
    return float(prob)

# ==================================================================
# 내 전적 CSV 로드
# ==================================================================
st.sidebar.header("1) 내 전적 CSV")
csv_mode = st.sidebar.radio("불러오기 방식", ["GitHub RAW URL", "파일 업로드"], horizontal=True)
df_pre = None
if csv_mode == "GitHub RAW URL":
    url = st.sidebar.text_input("RAW CSV URL", value="")
    if url:
        try:
            df_pre = pd.read_csv(url); st.sidebar.success(f"CSV 로드: {len(df_pre)}행")
        except Exception as e:
            st.sidebar.error(f"로드 실패: {e}")
else:
    up = st.sidebar.file_uploader("Drag & drop CSV", type=["csv"])
    if up:
        try:
            df_pre = pd.read_csv(up); st.sidebar.success(f"CSV 로드: {len(df_pre)}행")
        except Exception as e:
            st.sidebar.error(f"로드 실패: {e}")

# ==================================================================
# 개인 성향/최빈 룬/스펠
# ==================================================================
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
            s = sub[c].mode(dropna=True)
            mode[c] = s.iloc[0] if not s.empty else "__UNK__"
        modes[int(cid)] = mode
    return modes

ARAM_SPELLS = {"Mark":"눈덩이","Exhaust":"탈진","Ignite":"점화","Ghost":"유체화",
               "Heal":"회복","Barrier":"방어막","Cleanse":"정화","Clarity":"총명"}

def suggest_spells_for_champ(cid: int, id2tags: dict, ally_ids: list[int], enemy_ids: list[int]):
    tags = set(id2tags.get(cid, []))
    second = "유체화"
    if "Assassin" in tags or "Mage" in tags: second = "점화"
    if any("Assassin" in id2tags.get(e, []) for e in enemy_ids): second = "탈진"
    if "Marksman" in tags and any("Assassin" in id2tags.get(e, []) for e in enemy_ids): second = "탈진"
    return ["눈덩이", second]

def suggest_runes_from_modes(cid: int, misc_modes: dict):
    m = misc_modes.get(cid, {})
    def to_int(x):
        try: return int(x)
        except Exception: return None
    psn = style_id2name.get(to_int(m.get("primaryStyle","")), str(m.get("primaryStyle","")))
    ssn = style_id2name.get(to_int(m.get("subStyle","")),    str(m.get("subStyle","")))
    ksn = keystone_id2name.get(to_int(m.get("keystone","")), str(m.get("keystone","")))
    return {"primaryStyle": psn, "subStyle": ssn, "keystone": ksn}

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
    ap_like = {"Mage","Support"}; ad_like = {"Marksman","Fighter","Assassin"}
    ally_ap = sum(any(tt in ap_like for tt in t) for t in allies)
    ally_ad = sum(any(tt in ad_like for tt in t) for t in allies)
    if "Mage" in tags_me and ally_ap<=1: score += 0.03
    if "Marksman" in tags_me and ally_ad<=1: score += 0.03
    return min(score, 0.15)

# ==================================================================
# [수동 입력] 픽창 UI (폴백 용)
# ==================================================================
st.markdown("## 3) 픽창 입력(폴백)")
c1, c2 = st.columns(2)
with c1:
    ally_names_manual = st.multiselect("아군 챔피언(폴백용)", champ_df["name"].tolist(), max_selections=4)
with c2:
    enemy_names_manual = st.multiselect("상대 챔피언 (선택)", champ_df["name"].tolist(), max_selections=5)

# 세션 저장
st.session_state["ally_names_manual"] = ally_names_manual
st.session_state["enemy_names_manual"] = enemy_names_manual

# ==================================================================
# 🧩 후보 바 비율 크롭 + 아이콘 템플릿 매칭  +  팀(아군) 자동 인식(Gemini)
# ==================================================================
st.markdown("---")
st.header("🖼️ 픽창 스크린샷으로 자동 추천 (팀 자동 + 후보 비율크롭)")

# ── 이 스크린샷(예시)에 맞춘 기본값 (원하면 미세조정)
with st.sidebar.expander("후보 바 비율(상단 5칸) 튜닝", expanded=True):
    bar_x0_ratio = st.slider("bar_x0_ratio", 0.00, 1.00, 0.335, 0.001)
    bar_x1_ratio = st.slider("bar_x1_ratio", 0.00, 1.00, 0.735, 0.001)
    bar_y0_ratio = st.slider("bar_y0_ratio", 0.00, 1.00, 0.072, 0.001)
    bar_y1_ratio = st.slider("bar_y1_ratio", 0.00, 1.00, 0.132, 0.001)
    col_gap_ratio= st.slider("col_gap_ratio",0.000,0.050,0.012, 0.001)
    icon_match_size = st.slider("아이콘 매칭 크기(px)", 32, 128, 64, 16)
    x_shift = st.slider("X 전역 이동(±)", -0.20, 0.20, 0.00, 0.001)
    y_shift = st.slider("Y 전역 이동(±)", -0.20, 0.20, 0.00, 0.001)
    w_scale = st.slider("가로 스케일", 0.6, 1.4, 1.00, 0.01)
    h_scale = st.slider("세로 스케일", 0.6, 1.4, 1.00, 0.01)

class CandidateBarConfig:
    def __init__(self, x0, x1, y0, y1, gap, x_shift=0.0, y_shift=0.0, w_scale=1.0, h_scale=1.0):
        self.bar_x0_ratio=x0; self.bar_x1_ratio=x1; self.bar_y0_ratio=y0; self.bar_y1_ratio=y1
        self.col_gap_ratio=gap; self.x_shift=x_shift; self.y_shift=y_shift
        self.w_scale=w_scale; self.h_scale=h_scale

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
    if right-left < W*0.5 or bot-top < H*0.5: return img
    return img.crop((left, top, right+1, bot+1))

def _apply_adjust(x0r, x1r, y0r, y1r, xs, ys, ws, hs):
    cx=(x0r+x1r)/2; cy=(y0r+y1r)/2
    w=(x1r-x0r)*ws; h=(y1r-y0r)*hs
    nx0=cx-w/2+xs; nx1=cx+w/2+xs; ny0=cy-h/2+ys; ny1=cy+h/2+ys
    return max(0.0,nx0), min(1.0,nx1), max(0.0,ny0), min(1.0,ny1)

def crop_candidate_slots(img: Image.Image, cfg: CandidateBarConfig) -> Tuple[Image.Image, List[Image.Image], List[Tuple[int,int,int,int]]]:
    base = auto_trim_letterbox(img); W,H = base.size
    x0r,x1r,y0r,y1r = _apply_adjust(cfg.bar_x0_ratio,cfg.bar_x1_ratio,cfg.bar_y0_ratio,cfg.bar_y1_ratio,
                                    cfg.x_shift,cfg.y_shift,cfg.w_scale,cfg.h_scale)
    bar_x0,bar_x1 = int(round(W*x0r)), int(round(W*x1r))
    bar_y0,bar_y1 = int(round(H*y0r)), int(round(H*y1r))
    bar_w = max(1, bar_x1-bar_x0)
    gap_px = int(round(W*cfg.col_gap_ratio))
    total_gap = gap_px*4
    slot_w = max(1, (bar_w - total_gap)//5)
    slots, rects = [], []
    cur_x = bar_x0
    for _ in range(5):
        x0 = cur_x; x1 = x0+slot_w; y0 = bar_y0; y1 = bar_y1
        rects.append((x0,y0,x1,y1))
        slots.append(base.crop((x0,y0,x1,y1)))
        cur_x = x1 + gap_px
    return base, slots, rects

def draw_overlay(base_img: Image.Image, rects: list[tuple[int,int,int,int]]):
    im = base_img.copy(); dr = ImageDraw.Draw(im, "RGBA")
    for i,(x0,y0,x1,y1) in enumerate(rects,1):
        dr.rectangle([x0,y0,x1,y1], outline=(0,255,255,200), width=4)
        dr.text((x0+6,y0+6), f"{i}", fill=(255,255,255,220))
    return im

# ── DDragon 아이콘 캐싱 & 템플릿 매칭
@st.cache_data(show_spinner=False)
def _download_icon_as_arr(url: str, size: int = 64):
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB").resize((size,size))
        arr = np.asarray(im).astype(np.float32)/255.0
        return arr
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def build_icon_bank(size: int = 64) -> Dict[str, np.ndarray]:
    bank = {}
    for r in champ_df.itertuples():
        arr = _download_icon_as_arr(id2icon.get(r.championId,""), size=size)
        if arr is not None: bank[r.name] = arr
    return bank

def mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape: return 1e9
    d = (a-b); return float(np.mean(d*d))

def predict_champion_from_icon(crop_img: Image.Image, bank: Dict[str, np.ndarray], size: int = 64):
    arr = np.asarray(crop_img.convert("RGB").resize((size,size))).astype(np.float32)/255.0
    best_name, best_dist = None, 1e9
    for name, icon_arr in bank.items():
        d = mse(arr, icon_arr)
        if d < best_dist:
            best_dist, best_name = d, name
    conf = max(0.0, 1.0 - min(best_dist, 0.1)/0.1)  # 0~0.1 → 1.0~0.0
    return best_name, conf

# ── Gemini 초기화 & 팀(아군)만 JSON으로 추출
def init_vertex():
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception as e:
        st.info("Vertex AI 라이브러리가 없습니다. (pip 설치 또는 Secrets 설정 필요)")
        return None
    proj = st.secrets.get("GCP_PROJECT", "")
    loc  = st.secrets.get("GCP_LOCATION", "us-central1")
    sa_raw = st.secrets.get("GCP_SA_JSON", "")
    if not (proj and sa_raw):
        st.info("Secrets에 GCP_PROJECT, GCP_LOCATION, GCP_SA_JSON을 설정하세요.")
        return None
    try:
        sa_obj = json.loads(sa_raw)
    except Exception as e:
        st.error(f"GCP_SA_JSON 파싱 실패: {e}")
        return None
    key_path = "/tmp/gcp_key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(sa_obj, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    import vertexai
    vertexai.init(project=proj, location=loc)
    from vertexai.generative_models import GenerativeModel
    prefer = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash-002")
    model = GenerativeModel(prefer)
    return model

def detect_allies_from_gemini(img: Image.Image) -> List[str] | None:
    model = init_vertex()
    if model is None: return None
    from vertexai.generative_models import Part
    sys_prompt = (
        "You are extracting ONLY ally champions (left 4 slots) from a Korean ARAM pick-phase screenshot. "
        "Return STRICT JSON: {\"ally_champions\": [\"...\",\"...\",\"...\",\"...\"]}. "
        "Names must exactly match Korean champion names used in the League client. "
        "If uncertain, leave that entry out. No extra text."
    )
    buf = io.BytesIO(); img.save(buf, format="PNG")
    resp = model.generate_content(
        [sys_prompt, Part.from_data(mime_type="image/png", data=buf.getvalue())],
        generation_config={"temperature": 0.1, "max_output_tokens": 256},
    )
    raw = getattr(resp, "text", "") or ""
    if not raw and getattr(resp, "candidates", None):
        parts = []
        for c in resp.candidates:
            try:
                for p in getattr(c, "content", {}).parts or []:
                    if getattr(p, "text", None):
                        parts.append(p.text)
            except Exception:
                pass
        raw = "\n".join(parts)
    m = re.search(r"\{[\s\S]*\}", (raw or "").strip())
    if not m: return None
    try:
        data = json.loads(m.group(0))
        allies = data.get("ally_champions", [])
        return [str(x) for x in allies if str(x) in name2id]
    except Exception:
        return None

# ===== 업로더 =====
up_img = st.file_uploader("픽창 스크린샷 업로드 (PNG/JPG)", type=["png","jpg","jpeg"])
if up_img and st.button("🔍 스샷 인식 & 추천 (팀 자동 + 후보 비율크롭)"):
    img = Image.open(up_img).convert("RGB")

    # 1) 후보 5칸 인식 (비율 크롭 + 아이콘 매칭)
    cfg = CandidateBarConfig(bar_x0_ratio, bar_x1_ratio, bar_y0_ratio, bar_y1_ratio,
                             col_gap_ratio, x_shift, y_shift, w_scale, h_scale)
    base, slots, rects = crop_candidate_slots(img, cfg)
    st.image(draw_overlay(base, rects), caption="후보칸 오버레이 프리뷰", use_container_width=True)

    cols = st.columns(5)
    for i, cimg in enumerate(slots):
        with cols[i]:
            st.image(cimg, caption=f"후보 {i+1}", use_container_width=True)

    bank = build_icon_bank(size=icon_match_size)
    cand_names = []
    for cimg in slots:
        name, conf = predict_champion_from_icon(cimg, bank, size=icon_match_size)
        if name and name in name2id and conf >= 0.35:
            cand_names.append(name)
    cand_names = list(dict.fromkeys(cand_names))[:5]
    if not cand_names:
        st.warning("후보 챔피언 인식 실패(비율 슬라이더/밝기 조정).")
        st.stop()

    # 2) 아군 4명 자동 인식 (Gemini) → 실패 시 수동값 폴백
    ally_names_auto = detect_allies_from_gemini(img)
    if ally_names_auto and len(ally_names_auto) >= 4:
        ally_names = ally_names_auto[:4]
        st.success(f"아군 자동 인식: {ally_names}")
    else:
        ally_names = st.session_state.get("ally_names_manual", [])
        if len(ally_names) != 4:
            st.warning("아군 4명 자동 인식 실패. 사이드바에서 수동으로 4명을 선택하세요.")
            st.stop()

    enemy_names = st.session_state.get("enemy_names_manual", [])

    # 3) 추천 계산
    ally_ids  = [name2id[n] for n in ally_names]
    enemy_ids = [name2id[n] for n in enemy_names] if enemy_names else []
    per = build_personal_stats(df_pre) if df_pre is not None else pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
    per_map = per.set_index("championId").to_dict(orient="index") if len(per)>0 else {}
    misc_modes = per_champ_misc_modes(df_pre) if df_pre is not None else {}
    alpha = st.session_state.get("alpha", 0.60)
    beta  = st.session_state.get("beta", 0.35)
    gamma = st.session_state.get("gamma", 0.05)
    min_games_used = st.session_state.get("min_games", 5)

    rows = []
    for cname in cand_names:
        cid = name2id[cname]
        meta = per_map.get(cid, {"games":0,"wins":0,"wr":np.nan,"personal_score":-0.5})
        ps   = meta["personal_score"] - (0.3 if meta["games"]<min_games_used else 0.0)
        mode = misc_modes.get(cid, {})
        misc_row = {
            "spell_pair": str(mode.get("spell_pair","__UNK__")),
            "primaryStyle": str(mode.get("primaryStyle","__UNK__")),
            "subStyle": str(mode.get("subStyle","__UNK__")),
            "keystone": str(mode.get("keystone","__UNK__")),
            "patch": str(mode.get("patch","__UNK__")),
        }
        prob  = predict_prob_comp(bundle, cid, ally_ids, enemy_ids, misc_row)
        bonus = comp_bonus_score(cid, ally_ids, id2tags)
        score = alpha*prob + beta*ps + gamma*bonus
        rune = suggest_runes_from_modes(cid, misc_modes)
        spells = personal_spell_from_df(df_pre, cid, min_games=min_games_used) \
                 or suggest_spells_for_champ(cid, id2tags, ally_ids, enemy_ids)
        rows.append({
            "icon": id2icon.get(cid,""),
            "챔피언": cname,
            "예측승률α(%)": round(prob*100,2),
            "개인_게임수": meta.get("games",0),
            "개인_승률(%)": round(meta.get("wr",0)*100,2) if pd.notna(meta.get("wr")) else None,
            "추천_스펠": " + ".join(spells),
            "추천_룬": f"주{rune['primaryStyle']} · 부{rune['subStyle']} · 핵심{rune['keystone']}",
            "점수": score
        })

    out = pd.DataFrame(rows).sort_values("점수", ascending=False).reset_index(drop=True)
    st.subheader("후보 챔피언 추천 순위")
    top3 = out.head(3); cols = st.columns(len(top3))
    for col, (_, r) in zip(cols, top3.iterrows()):
        with col:
            if r["icon"]: st.image(r["icon"], width=64)
            st.markdown(f"**{r['챔피언']}**")
            st.write(f"예측 {r['예측승률α(%)']}%")
            st.write(f"스펠: {r['추천_스펠']}"); st.write(r["추천_룬"])

    st.markdown("### 전체 표")
    table = out.drop(columns=["점수"]).copy()
    st.dataframe(table, column_config={"icon": st.column_config.ImageColumn(" ", help="챔피언 아이콘", width="small")}, use_container_width=True)
