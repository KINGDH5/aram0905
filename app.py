# app.py
# ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP) + 스크린샷 인식(β)
# ---------------------------------------------------------------
# 사용법
# 1) 사이드바에서 내 전적 CSV 업로드(또는 RAW URL)
# 2) [픽창 입력] 탭에서 아군4/상대(선택)/후보 입력 → 추천 실행
# 3) [스크린샷(β)] 탭에서 픽창 스샷 업로드 → 자동 인식 & 추천
#
# 사전 준비
# - requirements.txt: streamlit, torch, pandas, numpy, requests, scikit-learn, pillow, google-cloud-aiplatform
# - Secrets (Streamlit Cloud):
#     MODEL_URL = "https://drive.google.com/uc?export=download&id=......"
#     GCP_PROJECT  = "elite-crossbar-471202-p5"
#     GCP_LOCATION = "us-central1"
#     GCP_SA_JSON  = """{ ... 서비스계정 키 JSON 전체 ... }"""
# ---------------------------------------------------------------

import os, io, json, time, math, requests
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder

# PyTorch 보안 로드(피클 안전 목록) – 체크포인트에 포함된 sklearn 객체 허용
from torch.serialization import add_safe_globals
add_safe_globals([OrdinalEncoder])

st.set_page_config(page_title="ARAM 픽창 개인화 추천", page_icon="🎯", layout="wide")
st.title("🎯 ARAM 픽창 개인화 추천 (내 2025 전적 + CompMLP)")

# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------
LANG = "ko_KR"
LOCAL_MODEL_PATH = "model/pregame_mlp_comp.pt"   # 레포에 깨진 pt가 있으면 다른 이름으로 바꿔도 됨
os.makedirs("model", exist_ok=True)

@st.cache_data(show_spinner=False)
def ddragon_latest_version():
    try:
        v = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=10).json()[0]
        return v
    except Exception:
        return "14.14.1"

@st.cache_data(show_spinner=True)
def load_champion_static(lang=LANG):
    ver = ddragon_latest_version()
    url = f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/{lang}/champion.json"
    data = requests.get(url, timeout=20).json()["data"]
    rows = []
    for k, v in data.items():
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

champ_df, id2name, id2icon, id2tags, name2id = load_champion_static()

# ------------------------------------------------------------------
# 모델 정의 & 유틸
# ------------------------------------------------------------------
class CompMLP(nn.Module):
    def __init__(self, n_champ, dim_champ=64, n_misc_cards=(64,64,64,64,64), dim_misc=16, hidden=128):
        super().__init__()
        self.emb_champ = nn.Embedding(n_champ+1, dim_champ)  # 마지막 인덱스=UNK
        sp, pri, sub, key, pat = n_misc_cards
        self.emb_sp  = nn.Embedding(sp+1, 8)   # +1 for UNK
        self.emb_pri = nn.Embedding(pri+1, 8)
        self.emb_sub = nn.Embedding(sub+1, 8)
        self.emb_key = nn.Embedding(key+1, 8)
        self.emb_pat = nn.Embedding(pat+1, 4)

        in_dim = dim_champ*(1+4+5) + (8+8+8+8+4)  # me + 4 ally + 5 enemy + misc
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
        )

    def forward(self, my_idx, ally_lists, enem_lists, misc_idx):
        # 챔피언 임베딩
        me = self.emb_champ(my_idx)  # [B, D]
        allies = [self.emb_champ(a) for a in ally_lists]  # 4 * [B, D]
        enemies = [self.emb_champ(e) for e in enem_lists] # 5 * [B, D] (빈 자리 0 또는 UNK로 채움)

        # misc: [B, 5]
        sp = self.emb_sp(misc_idx[:,0])
        pri = self.emb_pri(misc_idx[:,1])
        sub = self.emb_sub(misc_idx[:,2])
        key = self.emb_key(misc_idx[:,3])
        pat = self.emb_pat(misc_idx[:,4])

        misc = torch.cat([sp, pri, sub, key, pat], dim=-1)
        x = torch.cat([me, *allies, *enemies, misc], dim=-1)
        out = self.mlp(x).squeeze(-1)
        return out

def tensorize_ids(id_list, champ_id2idx, unk_idx):
    idxs = [champ_id2idx.get(int(i), unk_idx) for i in id_list]
    return torch.tensor([idxs], dtype=torch.long)

def enc_misc_row(enc: OrdinalEncoder, row: dict):
    # enc.categories_ 순서: [spell_pair, primaryStyle, subStyle, keystone, patch]
    vals = [[
        row.get("spell_pair", "__UNK__"),
        row.get("primaryStyle", "__UNK__"),
        row.get("subStyle", "__UNK__"),
        row.get("keystone", "__UNK__"),
        row.get("patch", "__UNK__"),
    ]]
    arr = enc.transform(vals).astype(int)  # -1 포함 가능
    # -1 → 해당 카드의 UNK(=last index)로 치환
    for j in range(arr.shape[1]):
        if arr[0, j] < 0:
            arr[0, j] = len(enc.categories_[j])  # 마지막 인덱스가 UNK
    return torch.tensor(arr, dtype=torch.long)

@st.cache_resource(show_spinner=True)
def load_model(local_path: str):
    if not os.path.exists(local_path):
        return None
    # 객체 포함 체크포인트 → weights_only=False
    obj = torch.load(local_path, map_location="cpu", weights_only=False)
    state_dict   = obj["state_dict"]
    champ_id2idx = obj["champ_id2idx"]
    enc_misc     = obj["enc_misc"]
    meta = obj.get("meta", {"dim_champ":64, "dim_misc":16})

    n_sp  = len(enc_misc.categories_[0])
    n_pri = len(enc_misc.categories_[1])
    n_sub = len(enc_misc.categories_[2])
    n_key = len(enc_misc.categories_[3])
    n_pat = len(enc_misc.categories_[4])

    model = CompMLP(
        n_champ=len(champ_id2idx),
        dim_champ=meta["dim_champ"],
        n_misc_cards=(n_sp, n_pri, n_sub, n_key, n_pat),
        dim_misc=meta["dim_misc"]
    )
    model.load_state_dict(state_dict)
    model.eval()
    return {"model": model, "champ_id2idx": champ_id2idx, "enc_misc": enc_misc}

def ensure_model_file(local_path: str, url: str):
    if os.path.exists(local_path): return local_path
    if not url: return None
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
        return local_path
    except Exception as e:
        st.error(f"모델 다운로드 실패: {e}")
        return None

@st.cache_resource(show_spinner=True)
def get_bundle():
    # 1) 로컬 경로 우선
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            b = load_model(LOCAL_MODEL_PATH)
            if b: return b
        except Exception as e:
            st.warning(f"로컬 모델 로드 실패, URL 재시도: {e}")

    # 2) Secrets / env 의 MODEL_URL에서 받기
    url = os.environ.get("MODEL_URL", "")
    if not url and "MODEL_URL" in st.secrets:
        url = st.secrets["MODEL_URL"].strip()
    if url:
        dl_path = LOCAL_MODEL_PATH if not os.path.exists(LOCAL_MODEL_PATH) else "model/pregame_mlp_comp_dl.pt"
        path = ensure_model_file(dl_path, url)
        if path:
            try:
                b = load_model(path)
                if b: return b
            except Exception as e:
                st.error(f"다운로드 모델 로드 실패: {e}")
    st.error("모델 준비 실패(로컬 파일 없음 & MODEL_URL 미설정)")
    return None

bundle = get_bundle()
if bundle is not None:
    st.sidebar.success("모델 준비 완료 ✅")
else:
    st.sidebar.error("모델 미로딩 ❌")

# ------------------------------------------------------------------
# 개인 전적 CSV 불러오기
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 개인 성향/최빈 룬 계산
# ------------------------------------------------------------------
def build_personal_stats(df: pd.DataFrame):
    if df is None or len(df)==0:
        return pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
    g = df.groupby("championId").agg(games=("win","size"), wins=("win","sum")).reset_index()
    g["wr"] = g["wins"]/g["games"]
    # 개인 성향 점수: wr*1.0 + log(games) 보정
    g["personal_score"] = g["wr"] + 0.1*np.log1p(g["games"])
    return g

def per_champ_misc_modes(df: pd.DataFrame):
    if df is None or len(df)==0:
        return {}
    cols = ["championId","spell_pair","primaryStyle","subStyle","keystone","patch"]
    for c in cols:
        if c not in df.columns:
            df[c] = "__UNK__"
    modes = {}
    for cid, sub in df.groupby("championId"):
        mode = {}
        for c in ["spell_pair","primaryStyle","subStyle","keystone","patch"]:
            mode[c] = sub[c].mode(dropna=True).iloc[0] if not sub[c].mode(dropna=True).empty else "__UNK__"
        modes[int(cid)] = mode
    return modes

# 기본 스펠/룬 추천(간단)
ARAM_SPELLS = {
    "Mark": "눈덩이","Exhaust": "탈진","Ignite": "점화","Ghost": "유체화",
    "Heal": "회복","Barrier": "방어막","Cleanse": "정화","Clarity": "총명"
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
    return {
        "primaryStyle": str(m.get("primaryStyle","")),
        "subStyle": str(m.get("subStyle","")),
        "keystone": str(m.get("keystone","")),
    }

# ------------------------------------------------------------------
# 예측/점수 계산
# ------------------------------------------------------------------
def predict_prob_comp(bundle, my_cid, ally_ids, enemy_ids, misc_row):
    if bundle is None: return 0.5
    model = bundle["model"]
    c2i   = bundle["champ_id2idx"]
    enc   = bundle["enc_misc"]
    device = torch.device("cpu")

    unk_idx = len(c2i)  # Embedding에서 마지막이 UNK
    my = torch.tensor([ [c2i.get(int(my_cid), unk_idx)] ], dtype=torch.long).to(device)

    def pad_ids(ids, need):
        ids = [int(x) for x in ids][:need]
        while len(ids)<need: ids.append(0)  # 0 → 임의(없는 id) → unk로 매핑됨
        return ids

    ally = tensorize_ids(pad_ids(ally_ids,4), c2i, unk_idx).to(device)
    enem = tensorize_ids(pad_ids(enemy_ids,5), c2i, unk_idx).to(device)
    misc = enc_misc_row(enc, misc_row).to(device)

    with torch.no_grad():
        logit = model(my, [ally[:,i] for i in range(4)], [enem[:,i] for i in range(5)], misc)
        prob = torch.sigmoid(logit).cpu().item()
    return float(prob)

def comp_bonus_score(my_cid, ally_ids, id2tags):
    # 매우 단순한 태그 시너지 보너스 (0~0.15)
    tags_me = set(id2tags.get(my_cid, []))
    allies = [set(id2tags.get(i, [])) for i in ally_ids]
    score = 0.0
    # 탱/서포트 유무
    if "Tank" in tags_me and not any("Tank" in t for t in allies): score += 0.05
    if "Support" in tags_me and not any("Support" in t for t in allies): score += 0.05
    # AP/AD 밸런스
    ap_like = {"Mage","Support"}
    ad_like = {"Marksman","Fighter","Assassin"}
    ally_ap = sum(any(tt in ap_like for tt in t) for t in allies)
    ally_ad = sum(any(tt in ad_like for tt in t) for t in allies)
    if "Mage" in tags_me and ally_ap<=1: score += 0.03
    if "Marksman" in tags_me and ally_ad<=1: score += 0.03
    return min(score, 0.15)

# ------------------------------------------------------------------
# [픽창 입력] – 드롭다운 수동 입력 UI
# ------------------------------------------------------------------
st.markdown("## 3) 픽창 입력")
colL, colR = st.columns(2)
with colL:
    ally_names = st.multiselect("아군 챔피언", champ_df["name"].tolist(), max_selections=4)
with colR:
    enemy_names = st.multiselect("상대 챔피언 (선택)", champ_df["name"].tolist(), max_selections=5)

cand_names = st.multiselect("후보 챔피언 (선택)", champ_df["name"].tolist(), help="여기 넣은 후보들만 점수화합니다.")
alpha = st.slider("α 모델 가중치", 0.0, 1.0, 0.60, 0.01)
beta  = st.slider("β 개인 성향 가중치", 0.0, 1.0, 0.35, 0.01)
gamma = st.slider("γ 조합 보너스 가중치", 0.0, 0.5, 0.05, 0.01)
min_games = st.number_input("개인 성향 최소 표본", 0, 50, 5, step=1)

# 세션에 저장(스크린샷 탭에서도 사용)
st.session_state["alpha"] = alpha
st.session_state["beta"]  = beta
st.session_state["gamma"] = gamma

if st.button("🚀 추천 실행"):
    if len(ally_names) != 4 or len(cand_names)==0:
        st.warning("아군 4명과 후보를 선택해 주세요.")
    else:
        ally_ids = [name2id[n] for n in ally_names]
        enemy_ids = [name2id[n] for n in enemy_names] if enemy_names else []

        # 개인 통계/룬 모드
        per = build_personal_stats(df_pre) if df_pre is not None else pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
        per_map = per.set_index("championId").to_dict(orient="index") if len(per)>0 else {}
        misc_modes = per_champ_misc_modes(df_pre) if df_pre is not None else {}

        rows = []
        for cname in cand_names:
            cid = name2id[cname]
            meta = per_map.get(cid, {"games":0,"wins":0,"wr":np.nan,"personal_score":-0.5})
            ps = meta["personal_score"] - (0.3 if meta["games"]<min_games else 0.0)

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
            score = alpha*prob + beta*ps + gamma*bonus

            rune = suggest_runes_from_modes(cid, misc_modes)
            spells = suggest_spells_for_champ(cid, id2tags, ally_ids, enemy_ids)

            rows.append({
                "icon": id2icon.get(cid,""),
                "챔피언": cname,
                "예측승률α(%)": round(prob*100,2),
                "조합보너스γ(%)": round(bonus*100,2),
                "개인_게임수": meta.get("games",0),
                "개인_승률(%)": round(meta["wr"]*100,2) if pd.notna(meta.get("wr")) else None,
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
        st.dataframe(out.drop(columns=["점수"]), use_container_width=True)

# ------------------------------------------------------------------
# 🖼️ 스크린샷 업로드 → 자동 인식 (Vertex AI)
# ------------------------------------------------------------------
st.markdown("---")
st.header("🖼️ 픽창 스크린샷으로 자동 추천 (β)")

import io
from PIL import Image

def init_vertex():
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception as e:
        st.error(f"Vertex AI 라이브러리가 없습니다: {e}")
        return None
    proj = st.secrets.get("GCP_PROJECT", "")
    loc  = st.secrets.get("GCP_LOCATION", "us-central1")
    sa   = st.secrets.get("GCP_SA_JSON", "")
    if not (proj and sa):
        st.info("Secrets에 GCP_PROJECT, GCP_LOCATION, GCP_SA_JSON을 설정하세요.")
        return None
    key_path = "/tmp/gcp_key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(sa)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    import vertexai
    vertexai.init(project=proj, location=loc)
    from vertexai.generative_models import GenerativeModel
    return GenerativeModel("gemini-1.5-flash")

def _names_to_ids(names):
    return [int(name2id[n]) for n in names if n in name2id]

up_img = st.file_uploader("픽창 스크린샷 업로드 (PNG/JPG)", type=["png","jpg","jpeg"])
if up_img and st.button("🔍 스샷 인식 & 추천"):
    img = Image.open(up_img).convert("RGB")
    st.image(img, caption="입력 이미지", use_column_width=True)

    model = init_vertex()
    if model is None:
        st.stop()

    from vertexai.generative_models import Part
    sys_prompt = (
        "You extract ARAM pick-phase info from screenshots. "
        "Return STRICT JSON with keys: ally_champions (string[]), candidate_champions (string[]). "
        "Names must be Korean exactly as shown in the League client. If not visible, return empty arrays. JSON only."
    )
    user_prompt = (
        "이 이미지는 무작위 총력전(ARAM) 픽창입니다. 왼쪽의 아군 4명과 상단 후보 챔피언을 읽어 "
        "다음 JSON 형식으로만 출력하세요.\n"
        "{\n"
        '  "ally_champions": ["초가스","타릭","벨베스","요네"],\n'
        '  "candidate_champions": ["다이애나","럭스","제드"]\n'
        "}\n"
    )
    buf = io.BytesIO(); img.save(buf, format="PNG")

    try:
        resp = model.generate_content(
            [sys_prompt, Part.from_data(mime_type="image/png", data=buf.getvalue()), user_prompt],
            generation_config={"temperature": 0.1, "max_output_tokens": 512},
        )
        data = json.loads(resp.text.strip())
    except Exception as e:
        st.error(f"인식 실패: {e}")
        st.stop()

    st.subheader("인식 결과")
    st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")

    ally_ids = _names_to_ids(data.get("ally_champions", []))
    cand_ids = _names_to_ids(data.get("candidate_champions", []))

    if len(ally_ids) != 4 or not cand_ids:
        st.info("아군 4명 또는 후보가 충분히 인식되지 않았습니다. 이미지 해상도/밝기를 높여 다시 시도해 주세요.")
        st.stop()

    # 개인 통계/룬 모드
    per = build_personal_stats(df_pre) if df_pre is not None else pd.DataFrame(columns=["championId","games","wins","wr","personal_score"])
    per_map = per.set_index("championId").to_dict(orient="index") if len(per)>0 else {}
    misc_modes = per_champ_misc_modes(df_pre) if df_pre is not None else {}

    alpha = st.session_state.get("alpha", 0.60)
    beta  = st.session_state.get("beta", 0.35)
    gamma = st.session_state.get("gamma", 0.05)

    rows = []
    for cid in cand_ids:
        cname = id2name.get(cid, str(cid))
        meta = per_map.get(cid, {"games":0,"wins":0,"wr":np.nan,"personal_score":-0.5})
        ps = meta["personal_score"] - (0.3 if meta["games"]<5 else 0.0)

        mode = misc_modes.get(cid, {})
        misc_row = {
            "spell_pair": str(mode.get("spell_pair", "__UNK__")),
            "primaryStyle": str(mode.get("primaryStyle", "__UNK__")),
            "subStyle": str(mode.get("subStyle", "__UNK__")),
            "keystone": str(mode.get("keystone", "__UNK__")),
            "patch": str(mode.get("patch", "__UNK__")),
        }
        prob  = predict_prob_comp(bundle, cid, ally_ids, [], misc_row)
        bonus = comp_bonus_score(cid, ally_ids, id2tags)
        score = alpha*prob + beta*ps + gamma*bonus

        rune = suggest_runes_from_modes(cid, misc_modes)
        spells = suggest_spells_for_champ(cid, id2tags, ally_ids, [])

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
    st.dataframe(out.drop(columns=["점수"]), use_container_width=True)
