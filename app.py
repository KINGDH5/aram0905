# app.py
# ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP) + 스크린샷 인식(β: 비율크롭+아이콘매칭)

import os, io, re, json, requests, numpy as np, pandas as pd, streamlit as st, torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict

# gdown (있으면 사용, 없으면 requests 폴백)
try:
    import gdown
    HAS_GDOWN = True
except Exception:
    HAS_GDOWN = False

# PyTorch가 sklearn 객체를 안전 로드할 수 있게 등록
from torch.serialization import add_safe_globals
add_safe_globals([OrdinalEncoder])

st.set_page_config(page_title="ARAM 픽창 개인화 추천", page_icon="🎯", layout="wide")
st.title("🎯 ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP)")

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
LANG = "ko_KR"
LOCAL_MODEL_PATH = "model/pregame_mlp_comp.pt"  # 레포에 없으면 MODEL_URL로 다운로드
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
    """
    runesReforged.json을 불러 ID→이름 매핑을 만든다.
    - style_id2name: 8000/8100/... → "정밀"/"지배"/...
    - keystone_id2name: 8128/8010/... → "감전"/"정복자"/...
    """
    ver = ddragon_latest_version()
    url = f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/{lang}/runesReforged.json"
    data = requests.get(url, timeout=20).json()

    style_id2name = {}
    keystone_id2name = {}
    for style in data:
        style_id2name[style["id"]] = style["name"]
        if style.get("slots"):
            # 슬롯0 = 키스톤
            for r in style["slots"][0].get("runes", []):
                keystone_id2name[r["id"]] = r["name"]
    return style_id2name, keystone_id2name

champ_df, id2name, id2icon, id2tags, name2id = load_champion_static()
style_id2name, keystone_id2name = load_runes_static()

# ==================================================================
# 체크포인트 모양을 그대로 복원하는 모델 로더 (크기 mismatch 방지)
# ==================================================================
class CompMLP_Exact(nn.Module):
    """
    체크포인트(state_dict)에서 임베딩/레이어 차원을 읽어 '그대로' 복원.
    allies/enemies 슬롯 개수도 입력차원에서 역산.
    """
    def __init__(self, n_champ, d_champ,
                 n_sp, d_sp, n_pri, d_pri, n_sub, d_sub, n_key, d_key, n_pat, d_pat,
                 in_dim, h1, h2, use_dropout, allies, enemies):
        super().__init__()
        # Embeddings
        self.emb_champ = nn.Embedding(n_champ, d_champ)
        self.emb_sp  = nn.Embedding(n_sp,  d_sp)
        self.emb_pri = nn.Embedding(n_pri, d_pri)
        self.emb_sub = nn.Embedding(n_sub, d_sub)
        self.emb_key = nn.Embedding(n_key, d_key)
        self.emb_pat = nn.Embedding(n_pat, d_pat)

        self.n_allies = allies
        self.n_enemies = enemies

        # MLP
        if use_dropout:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, h1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Linear(h2, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, h1),
                nn.ReLU(),
                nn.Linear(h1, h2),
                nn.ReLU(),
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

        sp  = self.emb_sp(misc_idx[:, 0])
        pri = self.emb_pri(misc_idx[:, 1])
        sub = self.emb_sub(misc_idx[:, 2])
        key = self.emb_key(misc_idx[:, 3])
        pat = self.emb_pat(misc_idx[:, 4])
        misc = torch.cat([sp, pri, sub, key, pat], dim=-1)

        x = torch.cat([me, *allies, *enemies, misc], dim=-1)

        # 입력 차원 보정 (예방)
        try:
            expect = int(self.mlp[0].in_features)
        except Exception:
            expect = None
            for mod in self.mlp:
                if isinstance(mod, torch.nn.Linear):
                    expect = int(mod.in_features)
                    break

        if expect is not None:
            cur = int(x.size(-1))
            if cur != expect:
                if not hasattr(self, "_dim_warned"):
                    import streamlit as st
                    st.warning(
                        f"[입력 차원 자동 보정] cur_dim={cur}, expect={expect} "
                        f"(allies={self.n_allies}, enemies={self.n_enemies})"
                    )
                    self._dim_warned = True
                if cur < expect:
                    pad = torch.zeros(x.size(0), expect - cur, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=-1)
                else:
                    x = x[..., :expect]

        return self.mlp(x).squeeze(-1)

def _infer_model_from_state(sd):
    # 임베딩 모양
    n_champ, d_champ = sd["emb_champ.weight"].shape
    n_sp, d_sp   = sd["emb_sp.weight"].shape
    n_pri, d_pri = sd["emb_pri.weight"].shape
    n_sub, d_sub = sd["emb_sub.weight"].shape
    n_key, d_key = sd["emb_key.weight"].shape
    n_pat, d_pat = sd["emb_pat.weight"].shape

    # MLP 크기/드롭아웃
    in_dim = sd["mlp.0.weight"].shape[1]
    h1     = sd["mlp.0.weight"].shape[0]
    use_dropout = ("mlp.3.weight" in sd and "mlp.2.weight" not in sd)
    h2 = sd["mlp.3.weight"].shape[0] if use_dropout else sd["mlp.2.weight"].shape[0]

    misc_sum = d_sp + d_pri + d_sub + d_key + d_pat

    # allies/enemies 자동 탐색 (정확히 in_dim 일치)
    best = None
    for allies in range(0, 6):
        for enemies in range(0, 10):
            expect = d_champ * (1 + allies + enemies) + misc_sum
            if expect == in_dim:
                score = -abs(allies - 4) * 10 + enemies  # allies=4 선호
                cand = (score, allies, enemies)
                if best is None or cand > best:
                    best = cand
    if best is None:
        total_slots = (in_dim - misc_sum) // d_champ
        allies = 4
        enemies = max(total_slots - 1 - allies, 0)
    else:
        allies = best[1]
        enemies = best[2]

    return dict(
        n_champ=n_champ, d_champ=d_champ,
        n_sp=n_sp, d_sp=d_sp, n_pri=n_pri, d_pri=d_pri,
        n_sub=n_sub, d_sub=d_sub, n_key=n_key, d_key=d_key,
        n_pat=n_pat, d_pat=d_pat,
        in_dim=in_dim, h1=h1, h2=h2, use_dropout=use_dropout,
        allies=allies, enemies=enemies
    )

def enc_misc_row(enc: OrdinalEncoder, row: dict):
    vals = [[
        row.get("spell_pair", "__UNK__"),
        row.get("primaryStyle", "__UNK__"),
        row.get("subStyle", "__UNK__"),
        row.get("keystone", "__UNK__"),
        row.get("patch", "__UNK__"),
    ]]
    arr = enc.transform(vals).astype(int)  # -1 포함 가능
    for j in range(arr.shape[1]):
        if arr[0, j] < 0:
            arr[0, j] = len(enc.categories_[j])  # UNK = 마지막 인덱스
    return torch.tensor(arr, dtype=torch.long)

# ---- Google Drive 헬퍼 ----
def _extract_drive_file_id(url: str) -> str | None:
    if not url:
        return None
    for pat in [r"/d/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})"]:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None

def ensure_model_file(local_path: str, url: str):
    if os.path.exists(local_path):
        return local_path
    if not url:
        return None

    fid = _extract_drive_file_id(url)

    try:
        if fid and HAS_GDOWN:
            gdown.download(f"https://drive.google.com/uc?id={fid}", local_path, quiet=False)
        else:
            dl_url = f"https://drive.google.com/uc?id={fid}&confirm=t" if fid else url
            with requests.get(dl_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
    except Exception as e:
        st.error(f"모델 다운로드 실패: {e}")
        return None

    # 파일 검증: HTML 받았는지 체크
    try:
        with open(local_path, "rb") as f:
            head = f.read(32)
        if head.strip().startswith(b"<"):
            raise ValueError("다운로드된 내용이 HTML입니다. 드라이브 공유 설정 또는 링크를 확인하세요.")
    except Exception as e:
        st.error(f"모델 파일 검증 실패: {e}")
        try:
            os.remove(local_path)
        except Exception:
            pass
        return None

    return local_path

@st.cache_resource(show_spinner=True)
def load_model(local_path: str):
    if not os.path.exists(local_path):
        return None
    # sklearn 객체 포함 → weights_only=False
    obj = torch.load(local_path, map_location="cpu", weights_only=False)

    state_dict   = obj["state_dict"]
    champ_id2idx = obj["champ_id2idx"]
    enc_misc     = obj["enc_misc"]

    spec = _infer_model_from_state(state_dict)
    model = CompMLP_Exact(
        spec["n_champ"], spec["d_champ"],
        spec["n_sp"], spec["d_sp"], spec["n_pri"], spec["d_pri"],
        spec["n_sub"], spec["d_sub"], spec["n_key"], spec["d_key"],
        spec["n_pat"], spec["d_pat"], spec["in_dim"], spec["h1"],
        spec["h2"], spec["use_dropout"], spec["allies"], spec["enemies"]
    )
    model.load_state_dict(state_dict)
    model.eval()
    return {"model": model, "champ_id2idx": champ_id2idx, "enc_misc": enc_misc,
            "allies": spec["allies"], "enemies": spec["enemies"]}

@st.cache_resource(show_spinner=True)
def get_bundle():
    # 0) 로컬 파일이 HTML/깨진 파일인지 사전 검증 → 있으면 삭제
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            with open(LOCAL_MODEL_PATH, "rb") as f:
                head = f.read(32)
            if head.strip().startswith(b"<"):
                os.remove(LOCAL_MODEL_PATH)
        except Exception:
            pass

    # 1) 로컬 우선 로드 시도
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            b = load_model(LOCAL_MODEL_PATH)
            if b:
                return b
        except Exception as e:
            try:
                os.remove(LOCAL_MODEL_PATH)
            except Exception:
                pass
            st.warning(f"로컬 모델 로드 실패, URL 재시도: {e}")

    # 2) Secrets/ENV에서 MODEL_URL 읽어 다운로드
    url = os.environ.get("MODEL_URL", "")
    if not url and "MODEL_URL" in st.secrets:
        url = st.secrets["MODEL_URL"].strip()

    if url:
        dl_path = LOCAL_MODEL_PATH
        path = ensure_model_file(dl_path, url)
        if path:
            try:
                b = load_model(path)
                if b:
                    return b
            except Exception as e:
                st.error(f"다운로드 모델 로드 실패: {e}")

    st.error("모델 준비 실패 (로컬 파일 없음 & MODEL_URL 미설정)")
    return None

bundle = get_bundle()
if bundle: st.sidebar.success("모델 준비 완료 ✅")
else:      st.sidebar.error("모델 미로딩 ❌")

def predict_prob_comp(bundle, my_cid, ally_ids, enemy_ids, misc_row):
    """체크포인트가 기대하는 allies/enemies 개수에 맞춰 패딩/트렁케이트"""
    if bundle is None:
        return 0.5

    model = bundle["model"]
    c2i   = bundle["champ_id2idx"]
    enc   = bundle["enc_misc"]
    na    = bundle.get("allies", 4)
    ne    = bundle.get("enemies", 5)

    device = torch.device("cpu")
    unk_idx = len(c2i)  # 미등록 id → 마지막 인덱스 사용

    def pad(ids, need):
        ids = [int(x) for x in ids][:need]
        while len(ids) < need:
            ids.append(0)
        return ids

    my = torch.tensor([c2i.get(int(my_cid), unk_idx)], dtype=torch.long).to(device)
    ally = torch.tensor([c2i.get(i, unk_idx) for i in pad(ally_ids, na)], dtype=torch.long).unsqueeze(0).to(device)
    enem = torch.tensor([c2i.get(i, unk_idx) for i in pad(enemy_ids, ne)], dtype=torch.long).unsqueeze(0).to(device)
    misc = enc_misc_row(enc, misc_row).to(device)

    with torch.no_grad():
        out = model(
            my,
            [ally[:, i] for i in range(ally.shape[1])],
            [enem[:, i] for i in range(enem.shape[1])],
            misc
        )
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
            df_pre = pd.read_csv(url)
            st.sidebar.success(f"CSV 로드: {len(df_pre)}행")
        except Exception as e:
            st.sidebar.error(f"로드 실패: {e}")
else:
    up = st.sidebar.file_uploader("Drag & drop CSV", type=["csv"])
    if up:
        try:
            df_pre = pd.read_csv(up)
            st.sidebar.success(f"CSV 로드: {len(df_pre)}행")
        except Exception as e:
            st.sidebar.error(f"로드 실패: {e}")

# ==================================================================
# 개인 성향/최빈 룬/스펠
# ==================================================================
def build_personal_stats(df: pd.DataFrame):
    if df is None or len(df)==0:
        return pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
    g = df.groupby("championId").agg(games=("win","size"), wins=("win","sum")).reset_index()
    g["wr"] = g["wins"]/g["games"]
    g["personal_score"] = g["wr"] + 0.1*np.log1p(g["games"])  # 간단 보정
    return g

def per_champ_misc_modes(df: pd.DataFrame):
    if df is None or len(df)==0:
        return {}
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

# 간단 스펠/룬 추천 휴리스틱
ARAM_SPELLS = {
    "Mark":"눈덩이", "Exhaust":"탈진", "Ignite":"점화", "Ghost":"유체화",
    "Heal":"회복", "Barrier":"방어막", "Cleanse":"정화", "Clarity":"총명"
}
def suggest_spells_for_champ(cid: int, id2tags: dict, ally_ids: list[int], enemy_ids: list[int]):
    tags = set(id2tags.get(cid, []))
    second = "유체화"
    if "Assassin" in tags or "Mage" in tags: second = "점화"
    if any("Assassin" in id2tags.get(e, []) for e in enemy_ids): second = "탈진"
    if "Marksman" in tags and any("Assassin" in id2tags.get(e, []) for e in enemy_ids): second = "탈진"
    return ["눈덩이", second]

def suggest_runes_from_modes(cid: int, misc_modes: dict):
    m = misc_modes.get(cid, {})
    ps = m.get("primaryStyle", "")
    ss = m.get("subStyle", "")
    ks = m.get("keystone", "")

    def to_int(x):
        try: return int(x)
        except Exception: return None

    psn = style_id2name.get(to_int(ps), str(ps))          # 스타일 이름
    ssn = style_id2name.get(to_int(ss), str(ss))          # 보조 스타일 이름
    ksn = keystone_id2name.get(to_int(ks), str(ks))       # 키스톤 이름
    return {"primaryStyle": psn, "subStyle": ssn, "keystone": ksn}

def personal_spell_from_df(df: pd.DataFrame, cid: int, min_games: int = 3):
    """
    내 CSV에서 해당 챔피언의 최빈 스펠 조합을 가져와 추천.
    - 표본 수가 min_games 미만이면 None → 휴리스틱 폴백
    - CSV 'spell_pair'가 "눈덩이+점화" 형식이라고 가정
    """
    if df is None or len(df) == 0 or "spell_pair" not in df.columns:
        return None
    sub = df[(df["championId"] == cid) & (df["spell_pair"].notna())]
    if sub.empty:
        return None
    cnt = sub.groupby("spell_pair").size().sort_values(ascending=False)
    top_pair = cnt.index[0]
    if cnt.iloc[0] < min_games:
        return None
    parts = [p.strip() for p in str(top_pair).split("+") if p.strip()]
    if len(parts) == 2:
        return parts
    return None

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
# [수동 입력] 픽창 UI
# ==================================================================
st.markdown("## 3) 픽창 입력")
c1, c2 = st.columns(2)
with c1:
    ally_names = st.multiselect("아군 챔피언", champ_df["name"].tolist(), max_selections=4)
with c2:
    enemy_names = st.multiselect("상대 챔피언 (선택)", champ_df["name"].tolist(), max_selections=5)

# 세션에 저장(스크린샷 섹션에서 재사용)
st.session_state["ally_names"] = ally_names
st.session_state["enemy_names"] = enemy_names

cand_names = st.multiselect("후보 챔피언 (선택)", champ_df["name"].tolist(), help="여기에 넣은 후보들만 점수화합니다.")
alpha = st.slider("α 모델 가중치", 0.0, 1.0, 0.60, 0.01)
beta  = st.slider("β 개인 성향 가중치", 0.0, 1.0, 0.35, 0.01)
gamma = st.slider("γ 조합 보너스 가중치", 0.0, 0.5, 0.05, 0.01)
min_games = st.number_input("개인 성향 최소 표본", 0, 50, 5, step=1)

st.session_state["alpha"] = alpha
st.session_state["beta"]  = beta
st.session_state["gamma"] = gamma
st.session_state["min_games"] = min_games

if st.button("🚀 추천 실행"):
    if len(ally_names) != 4 or len(cand_names)==0:
        st.warning("아군 4명과 후보를 선택해주세요.")
    else:
        ally_ids = [name2id[n] for n in ally_names]
        enemy_ids = [name2id[n] for n in enemy_names] if enemy_names else []

        per = build_personal_stats(df_pre) if df_pre is not None else pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
        per_map = per.set_index("championId").to_dict(orient="index") if len(per)>0 else {}
        misc_modes = per_champ_misc_modes(df_pre) if df_pre is not None else {}

        rows = []
        for cname in cand_names:
            cid = name2id[cname]
            meta = per_map.get(cid, {"games":0,"wins":0,"wr":np.nan,"personal_score":-0.5})
            ps   = meta["personal_score"] - (0.3 if meta["games"]<min_games else 0.0)

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
            spells = personal_spell_from_df(df_pre, cid, min_games=min_games) \
                     or suggest_spells_for_champ(cid, id2tags, ally_ids, enemy_ids)

            rows.append({
                "icon": id2icon.get(cid,""),
                "챔피언": cname,
                "예측승률α(%)": round(prob*100,2),
                "조합보너스γ(%)": round(bonus*100,2),
                "개인_게임수": meta.get("games",0),
                "개인_승률(%)": round(meta.get("wr",0)*100,2) if pd.notna(meta.get("wr")) else None,
                "추천_스펠": " + ".join(spells),
                "추천_룬": f"주{rune['primaryStyle']} · 부{rune['subStyle']} · 핵심{rune['keystone']}",
                "점수": score
            })

        out = pd.DataFrame(rows).sort_values("점수", ascending=False).reset_index(drop=True)

        st.subheader("추천 결과")
        top3 = out.head(3)
        cols = st.columns(len(top3))
        for col, (_, r) in zip(cols, top3.iterrows()):
            with col:
                if r["icon"]: st.image(r["icon"], width=64)
                st.markdown(f"**{r['챔피언']}**")
                st.write(f"예측 {r['예측승률α(%)']}% | 보너스 {r['조합보너스γ(%)']}%")
                st.write(f"스펠: {r['추천_스펠']}")
                st.write(r["추천_룬"])

        # === 전체 표 (아이콘 실제 이미지 렌더) ===
        st.subheader("전체 표")
        table = out.drop(columns=["점수"]).copy()
        st.dataframe(
            table,
            column_config={
                "icon": st.column_config.ImageColumn(" ", help="챔피언 아이콘", width="small")
            },
            use_container_width=True,
        )

# ==================================================================
# 🧩 비율 좌표 크롭 + 아이콘 템플릿 매칭 (후보 5칸)
# ==================================================================
st.markdown("---")
st.header("🖼️ 픽창 스크린샷으로 자동 추천 (β · 비율크롭)")

# ── 1) 후보 바 비율 파라미터 (슬라이더로 즉시 튜닝)
with st.sidebar.expander("후보 바 비율(상단 5칸) 튜닝", expanded=False):
    bar_x0_ratio = st.slider("bar_x0_ratio", 0.00, 1.00, 0.18, 0.001)
    bar_x1_ratio = st.slider("bar_x1_ratio", 0.00, 1.00, 0.82, 0.001)
    bar_y0_ratio = st.slider("bar_y0_ratio", 0.00, 1.00, 0.08, 0.001)
    bar_y1_ratio = st.slider("bar_y1_ratio", 0.00, 1.00, 0.18, 0.001)
    col_gap_ratio= st.slider("col_gap_ratio",0.000,0.050,0.008, 0.001)
    icon_match_size = st.slider("아이콘 매칭 크기(px)", 32, 128, 64, 16)

class CandidateBarConfig:
    def __init__(self, x0, x1, y0, y1, gap):
        self.bar_x0_ratio = x0
        self.bar_x1_ratio = x1
        self.bar_y0_ratio = y0
        self.bar_y1_ratio = y1
        self.col_gap_ratio = gap

def auto_trim_letterbox(img: Image.Image, thresh: int = 8) -> Image.Image:
    arr = np.asarray(img.convert("L"))
    H, W = arr.shape
    top = 0
    while top < H and arr[top].mean() < thresh: top += 1
    bot = H - 1
    while bot > top and arr[bot].mean() < thresh: bot -= 1
    left = 0
    while left < W and arr[:, left].mean() < thresh: left += 1
    right = W - 1
    while right > left and arr[:, right].mean() < thresh: right -= 1
    if right - left < W * 0.5 or bot - top < H * 0.5:
        return img
    return img.crop((left, top, right + 1, bot + 1))

def crop_candidate_slots(img: Image.Image, cfg: CandidateBarConfig) -> List[Image.Image]:
    base = auto_trim_letterbox(img)
    W, H = base.size
    bar_x0 = int(round(W * cfg.bar_x0_ratio))
    bar_x1 = int(round(W * cfg.bar_x1_ratio))
    bar_y0 = int(round(H * cfg.bar_y0_ratio))
    bar_y1 = int(round(H * cfg.bar_y1_ratio))
    bar_w  = max(1, bar_x1 - bar_x0)
    gap_px = int(round(W * cfg.col_gap_ratio))
    total_gap = gap_px * 4
    slot_w = max(1, (bar_w - total_gap) // 5)
    slots = []
    cur_x = bar_x0
    for _ in range(5):
        x0 = cur_x
        x1 = x0 + slot_w
        y0 = bar_y0
        y1 = bar_y1
        slots.append(base.crop((x0, y0, x1, y1)))
        cur_x = x1 + gap_px
    return slots

# ── 2) DDragon 아이콘을 내려받아 매칭용 배열로 캐시
@st.cache_data(show_spinner=False)
def _download_icon_as_arr(url: str, size: int = 64):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB").resize((size, size))
        arr = np.asarray(im).astype(np.float32) / 255.0
        return arr
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def build_icon_bank(size: int = 64) -> Dict[str, np.ndarray]:
    bank = {}
    for r in champ_df.itertuples():
        arr = _download_icon_as_arr(id2icon.get(r.championId, ""), size=size)
        if arr is not None:
            bank[r.name] = arr
    return bank

def mse(a: np.ndarray, b: np.ndarray) -> float:
    # 두 배열 모양이 다르면 즉시 큰 값 반환
    if a.shape != b.shape:
        return 1e9
    diff = (a - b)
    return float(np.mean(diff * diff))

def predict_champion_from_icon(crop_img: Image.Image, bank: Dict[str, np.ndarray], size: int = 64):
    """간단 템플릿 매칭(MSE). 필요시 SSIM/임베딩으로 교체 가능."""
    arr = np.asarray(crop_img.convert("RGB").resize((size, size))).astype(np.float32) / 255.0
    best_name, best_dist = None, 1e9
    for name, icon_arr in bank.items():
        d = mse(arr, icon_arr)
        if d < best_dist:
            best_dist = d
            best_name = name
    # 신뢰도 스케일(경험적): 0~0.1 구간을 1.0~0.0로 역변환
    conf = max(0.0, 1.0 - min(best_dist, 0.1) / 0.1)
    return best_name, conf

up_img2 = st.file_uploader("픽창 스크린샷 업로드 (PNG/JPG)", type=["png","jpg","jpeg"], key="ratio_uploader")
if up_img2 and st.button("🔍 스샷 인식 & 추천 (비율크롭)"):
    img = Image.open(up_img2).convert("RGB")
    st.image(img, caption="입력 이미지", use_container_width=True)

    cfg = CandidateBarConfig(bar_x0_ratio, bar_x1_ratio, bar_y0_ratio, bar_y1_ratio, col_gap_ratio)
    slots = crop_candidate_slots(img, cfg)

    cols = st.columns(5)
    for i, cimg in enumerate(slots):
        with cols[i]:
            st.image(cimg, caption=f"후보 {i+1}", use_container_width=True)

    bank = build_icon_bank(size=icon_match_size)
    cand_names_detected = []
    for cimg in slots:
        name, conf = predict_champion_from_icon(cimg, bank, size=icon_match_size)
        if name and name in name2id and conf >= 0.35:  # 너무 낮으면 버림
            cand_names_detected.append(name)

    # 중복 제거, 최대 5개
    cand_names_detected = list(dict.fromkeys(cand_names_detected))[:5]
    st.write("인식된 후보:", cand_names_detected if cand_names_detected else "없음")

    # 아군은 '수동 입력' 섹션의 선택을 사용
    ally_names_ss = st.session_state.get("ally_names", [])
    enemy_names_ss = st.session_state.get("enemy_names", [])

    if len(ally_names_ss) != 4 or not cand_names_detected:
        st.warning("아군 4명 선택(상단 '픽창 입력' 섹션) 및 후보 인식이 필요합니다.")
        st.stop()

    ally_ids = [name2id[n] for n in ally_names_ss]
    enemy_ids = [name2id[n] for n in enemy_names_ss] if enemy_names_ss else []

    per = build_personal_stats(df_pre) if df_pre is not None else pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
    per_map = per.set_index("championId").to_dict(orient="index") if len(per)>0 else {}
    misc_modes = per_champ_misc_modes(df_pre) if df_pre is not None else {}

    alpha = st.session_state.get("alpha", 0.60)
    beta  = st.session_state.get("beta", 0.35)
    gamma = st.session_state.get("gamma", 0.05)
    min_games_used = st.session_state.get("min_games", 5)

    rows = []
    for cname in cand_names_detected:
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
    top3 = out.head(3)
    cols = st.columns(len(top3))
    for col, (_, r) in zip(cols, top3.iterrows()):
        with col:
            if r["icon"]: st.image(r["icon"], width=64)
            st.markdown(f"**{r['챔피언']}**")
            st.write(f"예측 {r['예측승률α(%)']}%")
            st.write(f"스펠: {r['추천_스펠']}")
            st.write(r["추천_룬"])

    st.markdown("### 전체 표")
    table = out.drop(columns=["점수"]).copy()
    st.dataframe(
        table,
        column_config={
            "icon": st.column_config.ImageColumn(" ", help="챔피언 아이콘", width="small")
        },
        use_container_width=True,
    )
