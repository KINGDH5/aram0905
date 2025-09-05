import os, io, json, time, requests, numpy as np, pandas as pd, streamlit as st, torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder  # enc_misc 로드용

# 기존 import 들 아래에 추가
from torch.serialization import add_safe_globals
from sklearn.preprocessing import OrdinalEncoder  # 이미 있을 수도 있음
add_safe_globals([OrdinalEncoder])  # sklearn 객체 로드 허용

LANG = "ko_KR"
LOCAL_MODEL_PATH = "model/pregame_mlp_comp.pt"  # 우선 여기서 찾고, 없으면 MODEL_URL에서 다운로드
CSV_DEFAULT_RAW_URL = ""  # 선택: 내 CSV RAW URL을 여기에 기본값으로 지정 가능

# --------- Helper: download & cache model ---------
@st.cache_resource(show_spinner=True)
def ensure_model_file(local_path: str, url: str | None):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return local_path
    if not url:
        return None
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = local_path + ".downloading"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1<<20):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, local_path)
        return local_path
    except Exception as e:
        st.error(f"모델 다운로드 실패: {e}")
        return None

# --------- Data Dragon ---------
@st.cache_data(show_spinner=False)
def get_dd_version() -> str:
    return requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]

@st.cache_data(show_spinner=True)
def load_champion_full(ver: str, lang: str):
    url = f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/{lang}/championFull.json"
    data = requests.get(url).json()["data"]
    rows = []
    for k, c in data.items():
        cid = int(c.get("key","0"))
        rows.append({
            "championId": cid,
            "championKey": c.get("id",""),
            "championName": c.get("name",""),
            "tags": c.get("tags", []) or [],
            "icon_url": f"https://ddragon.leagueoflegends.com/cdn/{ver}/img/champion/{c.get('image',{}).get('full','')}"
        })
    df = pd.DataFrame(rows).sort_values("championName")
    id2name = dict(zip(df.championId, df.championName))
    id2icon = dict(zip(df.championId, df.icon_url))
    id2tags = dict(zip(df.championId, df.tags))
    name2id = {v:k for k,v in id2name.items()}
    return df, id2name, id2icon, id2tags, name2id

def role_tags():
    return {"Front":{"Tank","Fighter"}, "AP":{"Mage"}, "AD":{"Marksman","Assassin"}, "Support":{"Support"}}

def comp_bonus_score(cid: int, ally_ids: list[int], id2tags: dict[int, list[str]]) -> float:
    ROLE = role_tags()
    def role_vec(_cid):
        t = set(id2tags.get(_cid, []))
        return {"Front":int(len(t&ROLE["Front"])>0), "AP":int(len(t&ROLE["AP"])>0),
                "AD":int(len(t&ROLE["AD"])>0), "Support":int(len(t&ROLE["Support"])>0)}
    summ = {"Front":0,"AP":0,"AD":0,"Support":0}
    for aid in ally_ids:
        rv = role_vec(aid)
        for k in summ: summ[k]+=rv[k]
    me = role_vec(cid)
    bonus = 0.0
    if summ["Front"]==0 and me["Front"]: bonus+=0.04
    if summ["AD"]==0   and me["AD"]:    bonus+=0.03
    if summ["AP"]==0   and me["AP"]:    bonus+=0.03
    if summ["Support"]==0 and me["Support"]: bonus+=0.02
    if summ["AD"]>=3 and summ["AP"]<=1 and me["AP"]: bonus+=0.03
    if summ["AP"]>=3 and summ["AD"]<=1 and me["AD"]: bonus+=0.03
    return bonus

def patch2major(ver: str) -> str:
    if not isinstance(ver, str): return ""
    parts = ver.split(".")
    return ".".join(parts[:2]) if len(parts)>=2 else ver

# --------- Personal CSV ---------
@st.cache_data(show_spinner=True)
def load_personal_csv_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def build_personal_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["championId","win"]).copy()
    df["championId"] = df["championId"].astype(int)
    g = df.groupby("championId")["win"].agg(games="count", wins="sum").reset_index()
    g["wr"] = g["wins"] / g["games"]
    def norm(x):
        m, s = x.mean(), x.std()
        return (x - m) / (s + 1e-9) if len(x)>1 else (x*0)
    g["games_z"] = norm(g["games"]); g["wr_z"] = norm(g["wr"])
    g["personal_score"] = 0.4*g["games_z"] + 0.6*g["wr_z"]
    return g

def per_champ_misc_modes(df: pd.DataFrame):
    df = df.copy()
    df["patch"] = df["gameVersion"].fillna("").map(patch2major)
    cols = ["spell_pair","primaryStyle","subStyle","keystone","patch"]
    for c in ["primaryStyle","subStyle","keystone"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    modes = {}
    for cid, grp in df.groupby("championId"):
        row = {}
        for c in cols:
            if grp[c].notna().sum()>0:
                row[c] = grp[c].mode(dropna=True).iloc[0]
            else:
                row[c] = "__UNK__"
        modes[int(cid)] = row
    return modes

# --------- Model (CompMLP) ---------
class CompMLP(nn.Module):
    def __init__(self, n_champ:int, dim_champ=64, n_misc_cards=(0,0,0,0,0), dim_misc=16):
        super().__init__()
        self.emb_champ = nn.Embedding(n_champ+1, dim_champ)  # +1 unknown
        (c_sp, c_pri, c_sub, c_key, c_pat) = n_misc_cards
        self.emb_sp  = nn.Embedding(c_sp +1, dim_misc)
        self.emb_pri = nn.Embedding(c_pri+1, dim_misc)
        self.emb_sub = nn.Embedding(c_sub+1, dim_misc)
        self.emb_key = nn.Embedding(c_key+1, dim_misc)
        self.emb_pat = nn.Embedding(c_pat+1, dim_misc)
        in_dim = dim_champ*(1+1+1) + dim_misc*5
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, my_idx, ally_lists, enem_lists, misc_idx):
        my_idx = my_idx.clone()
        my_idx[my_idx<0] = self.emb_champ.num_embeddings - 1
        my_emb = self.emb_champ(my_idx)
        B, D = my_emb.size(0), self.emb_champ.embedding_dim
        ally_sum = torch.zeros(B, D, device=my_emb.device)
        enem_sum = torch.zeros(B, D, device=my_emb.device)
        for i in range(B):
            a = ally_lists[i].clone(); e = enem_lists[i].clone()
            a[a<0] = self.emb_champ.num_embeddings - 1
            e[e<0] = self.emb_champ.num_embeddings - 1
            if a.numel()>0: ally_sum[i] = self.emb_champ(a).sum(0)
            if e.numel()>0: enem_sum[i] = self.emb_champ(e).sum(0)
        m = misc_idx.clone()
        sp  = m[:,0]; sp[sp  < 0] = self.emb_sp.num_embeddings  - 1
        pri = m[:,1]; pri[pri < 0] = self.emb_pri.num_embeddings - 1
        sub = m[:,2]; sub[sub < 0] = self.emb_sub.num_embeddings - 1
        key = m[:,3]; key[key < 0] = self.emb_key.num_embeddings - 1
        pat = m[:,4]; pat[pat < 0] = self.emb_pat.num_embeddings - 1
        emb_misc = torch.cat([self.emb_sp(sp), self.emb_pri(pri), self.emb_sub(sub),
                              self.emb_key(key), self.emb_pat(pat)], dim=1)
        z = torch.cat([my_emb, ally_sum, enem_sum, emb_misc], dim=1)
        return self.mlp(z).squeeze(1)

@st.cache_resource(show_spinner=True)
def load_model(local_path: str):
    if not os.path.exists(local_path):
        return None
    # 중요: weights_only=False (객체 포함 체크포인트를 읽기 위해 필요)
    obj = torch.load(local_path, map_location="cpu", weights_only=False)
    state_dict   = obj["state_dict"]
    champ_id2idx = obj["champ_id2idx"]
    enc_misc     = obj["enc_misc"]
    meta         = obj.get("meta", {"dim_champ":64,"dim_misc":16})

    n_sp  = len(enc_misc.categories_[0])
    n_pri = len(enc_misc.categories_[1])
    n_sub = len(enc_misc.categories_[2])
    n_key = len(enc_misc.categories_[3])
    n_pat = len(enc_misc.categories_[4])

    model = CompMLP(n_champ=len(champ_id2idx), dim_champ=meta["dim_champ"],
                    n_misc_cards=(n_sp,n_pri,n_sub,n_key,n_pat), dim_misc=meta["dim_misc"])
    model.load_state_dict(state_dict); model.eval()
    return {"model": model, "champ_id2idx": champ_id2idx, "enc_misc": enc_misc}


def encode_misc(enc_misc: OrdinalEncoder, row_dict: dict):
    X = pd.DataFrame([row_dict], columns=["spell_pair","primaryStyle","subStyle","keystone","patch"])
    X_enc = enc_misc.transform(X).astype(np.int64)
    return torch.tensor(X_enc, dtype=torch.long)

def map_ids_to_idx(champ_id2idx: dict, ids: list[int]) -> torch.Tensor:
    return torch.tensor([champ_id2idx.get(int(x), -1) for x in ids], dtype=torch.long)

def predict_prob_comp(bundle, cid: int, ally_ids: list[int], enemy_ids: list[int], misc_row: dict):
    if bundle is None: return 0.5
    model = bundle["model"]; enc_misc = bundle["enc_misc"]; cid2idx = bundle["champ_id2idx"]
    my_idx   = torch.tensor([cid2idx.get(int(cid), -1)], dtype=torch.long)
    ally_idx = [map_ids_to_idx(cid2idx, ally_ids)]
    enem_idx = [map_ids_to_idx(cid2idx, enemy_ids)]
    misc_idx = encode_misc(enc_misc, misc_row)  # (1,5)
    with torch.no_grad():
        prob = torch.sigmoid(model(my_idx, ally_idx, enem_idx, misc_idx)).cpu().numpy()[0].item()
    return float(prob)

# -------------------- UI --------------------
st.set_page_config(page_title="🎯 ARAM 픽창 개인화 추천 (CompMLP)", page_icon="🎯", layout="wide")
st.title("🎯 ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP)")

with st.sidebar:
    st.subheader("1) 내 전적 CSV")
    mode = st.radio("불러오기 방식", ["GitHub RAW URL", "파일 업로드"], index=0)
    df_pre = None
    if mode == "GitHub RAW URL":
        url = st.text_input("RAW CSV URL", value=CSV_DEFAULT_RAW_URL,
                            placeholder="https://raw.githubusercontent.com/<user>/<repo>/main/pre_game_me_2025.csv")
        if url:
            try:
                df_pre = load_personal_csv_from_url(url); st.success(f"CSV 로드: {len(df_pre)}행")
            except Exception as e:
                st.error(f"로드 실패: {e}")
    else:
        up = st.file_uploader("pre_game_me_2025.csv 업로드", type=["csv"])
        if up:
            try:
                df_pre = pd.read_csv(up); st.success(f"CSV 로드: {len(df_pre)}행")
            except Exception as e:
                st.error(f"로드 실패: {e}")

    st.subheader("2) 모델 로드")
    # 2-1) 로컬 파일 있으면 사용, 없으면 MODEL_URL에서 다운로드
    model_url = st.secrets.get("MODEL_URL", os.environ.get("MODEL_URL", "")).strip()
    path = ensure_model_file(LOCAL_MODEL_PATH, model_url if model_url else None)
    if path is None:
        st.error("모델이 없습니다. 레포에 model/pregame_mlp_comp.pt를 넣거나, secrets에 MODEL_URL을 설정하세요.")
        bundle = None
    else:
        bundle = load_model(path)
        if bundle is None:
            st.error("모델 로드 실패: 파일이 손상되었을 수 있습니다.")
        else:
            st.success("모델 준비 완료 ✅")

st.markdown("### 3) 픽창 입력")

ver = get_dd_version()
champ_df, id2name, id2icon, id2tags, name2id = load_champion_full(ver, LANG)
names = champ_df["championName"].tolist()

c1, c2, c3 = st.columns(3)
with c1:
    ally_names = st.multiselect("아군 확정 4명", names, max_selections=4)
with c2:
    enemy_names = st.multiselect("상대 챔피언 (선택)", names)
with c3:
    candidate_names = st.multiselect("후보 챔피언 (선택)", names, help="없으면 자동 추천(내 전적+역할보완)")

ally_ids = [name2id[n] for n in ally_names]
enemy_ids = [name2id[n] for n in enemy_names]
candidates = [name2id[n] for n in candidate_names]

alpha = st.slider("α 모델 가중치", 0.0, 1.0, 0.60, 0.05)
beta  = st.slider("β 개인 성향 가중치", 0.0, 1.0, 0.35, 0.05)
gamma = st.slider("γ 조합 보너스 가중치", 0.0, 1.0, 0.05, 0.01)
min_games = st.number_input("개인 성향 최소 표본", 0, 50, 5)

run = st.button("🚀 추천 실행")

def per_champ_misc_modes(df: pd.DataFrame):
    df = df.copy()
    df["patch"] = df["gameVersion"].fillna("").map(patch2major)
    cols = ["spell_pair","primaryStyle","subStyle","keystone","patch"]
    for c in ["primaryStyle","subStyle","keystone"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    modes = {}
    for cid, grp in df.groupby("championId"):
        row = {}
        for c in cols:
            if grp[c].notna().sum()>0:
                row[c] = grp[c].mode(dropna=True).iloc[0]
            else:
                row[c] = "__UNK__"
        modes[int(cid)] = row
    return modes

if run:
    if df_pre is None:
        st.error("먼저 내 CSV를 불러와 주세요."); st.stop()
    if len(ally_ids) != 4:
        st.error("아군 확정 챔피언을 **정확히 4명** 선택하세요."); st.stop()
    if bundle is None:
        st.error("모델이 준비되지 않았습니다."); st.stop()

    per = build_personal_stats(df_pre)
    per_map = per.set_index("championId").to_dict(orient="index")
    misc_modes = per_champ_misc_modes(df_pre)

    if candidates:
        pool = sorted(set(candidates))
    else:
        base = per.sort_values(["games","wr"], ascending=False)["championId"].tolist()
        pool_ids = set(base[:80])
        ROLE = role_tags()
        summ = {"Front":0,"AP":0,"AD":0,"Support":0}
        for aid in ally_ids:
            t = set(id2tags.get(aid, []))
            summ["Front"]+= int(len(t & ROLE["Front"])>0)
            summ["AP"]+= int(len(t & ROLE["AP"])>0)
            summ["AD"]+= int(len(t & ROLE["AD"])>0)
            summ["Support"]+= int(len(t & ROLE["Support"])>0)
        needs=[]
        if summ["Front"]==0: needs.append("Front")
        if summ["AD"]==0:    needs.append("AD")
        if summ["AP"]==0:    needs.append("AP")
        if summ["Support"]==0: needs.append("Support")
        if needs:
            for cid in champ_df["championId"]:
                tags = set(id2tags.get(int(cid), []))
                if ("Front" in needs and len(tags & ROLE["Front"])>0) \
                   or ("AD" in needs and len(tags & ROLE["AD"])>0) \
                   or ("AP" in needs and len(tags & ROLE["AP"])>0) \
                   or ("Support" in needs and len(tags & ROLE["Support"])>0):
                    pool_ids.add(int(cid))
        pool = sorted(list(pool_ids))

    rows = []
    for cid in pool:
        meta = per_map.get(cid, {"games":0,"wins":0,"wr":np.nan,"personal_score":-0.5})
        personal_adj = meta["personal_score"] - (0.3 if meta["games"] < min_games else 0.0)
        mode = misc_modes.get(cid, {})
        misc_row = {
            "spell_pair": str(mode.get("spell_pair", "__UNK__")),
            "primaryStyle": str(mode.get("primaryStyle", "__UNK__")),
            "subStyle": str(mode.get("subStyle", "__UNK__")),
            "keystone": str(mode.get("keystone", "__UNK__")),
            "patch": str(mode.get("patch", "__UNK__")),
        }
        prob  = predict_prob_comp(bundle, cid, ally_ids, enemy_ids, misc_row)
        bonus = comp_bonus_score(cid, ally_ids, id2tags)
        final_score = alpha*prob + beta*personal_adj + gamma*bonus
        rows.append({
            "icon": id2icon.get(cid,""),
            "championId": cid,
            "챔피언": id2name.get(cid,str(cid)),
            "개인_게임수": meta["games"],
            "개인_승률": round(meta["wr"]*100,2) if pd.notna(meta["wr"]) else None,
            "예측승률α": round(prob*100,2),
            "조합보너스γ": round(bonus*100,2),
            "최종점수": final_score
        })

    out = pd.DataFrame(rows).sort_values("최종점수", ascending=False).reset_index(drop=True)

    st.subheader("추천 Top 5")
    top5 = out.head(5)
    cols = st.columns(len(top5))
    for col, (_, r) in zip(cols, top5.iterrows()):
        with col:
            if r["icon"]: st.image(r["icon"], width=64)
            st.markdown(f"**{r['챔피언']}**")
            st.caption(f"개인 {r['개인_게임수']}판 / {r['개인_승률'] or '-'}%")
            st.caption(f"예측 {r['예측승률α']}% · 보너스 {r['조합보너스γ']}%")

    st.markdown("### 전체 순위")
    st.dataframe(out.drop(columns=["최종점수"]), use_container_width=True)
