import pandas as pd
import numpy as np

year = "2025"

# -----------------------------
# CHARGEMENT
# -----------------------------
players = pd.read_csv(f"~/ValorantChampions/data/{year}/player_stats.csv")
dmatches = pd.read_csv(f"~/ValorantChampions/data/{year}/detailed_matches_player_stats.csv")

# -----------------------------
# FILTRER UNIQUEMENT LES LIGNES MAP (exclure 'overall')
# -----------------------------
if 'stat_type' in dmatches.columns:
    maps_df = dmatches[dmatches['stat_type'].astype(str).str.lower() == 'map'].copy()
else:
    # fallback si pas de colonne stat_type (mais tu as dit qu'il y en a)
    maps_df = dmatches[dmatches['map_name'].notna()].copy()

# normalise les noms
maps_df['player_name'] = maps_df['player_name'].astype(str).str.strip()
maps_df['player_team'] = maps_df['player_team'].astype(str).str.strip()
maps_df['agent'] = maps_df['agent'].astype(str)

# -----------------------------
# HELPERS
# -----------------------------
def parse_agents(value):
    """Retourne liste d'agents propres à partir du champ agent (gère 'A, B' ou "['A','B']")."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value)
    s = s.replace('[','').replace(']','').replace("'", "").replace('"','')
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return parts

def make_map_key(df):
    if 'match_id' in df.columns and 'map_name' in df.columns:
        return df['match_id'].astype(str).fillna('') + '||' + df['map_name'].astype(str).fillna('')
    else:
        return (
            df.get('team1','').astype(str).fillna('') + '||' +
            df.get('team2','').astype(str).fillna('') + '||' +
            df.get('map_name','').astype(str).fillna('') + '||' +
            df.get('match_date','').astype(str).fillna('')
        )

def determine_map_winner(row):
    mw = row.get('map_winner')
    if pd.notna(mw) and str(mw).strip() != '':
        return mw
    score = row.get('score_overall')
    team1 = row.get('team1'); team2 = row.get('team2')
    if pd.isna(score) or pd.isna(team1) or pd.isna(team2):
        return np.nan
    try:
        parts = [p.strip() for p in str(score).split('-')]
        if len(parts) != 2:
            return np.nan
        s1 = int(parts[0]); s2 = int(parts[1])
        if s1 > s2:
            return team1
        elif s2 > s1:
            return team2
    except Exception:
        return np.nan
    return np.nan

# -----------------------------
# PREP MAPS DF
# -----------------------------
# clé unique par map (robuste)
maps_df['_map_key'] = make_map_key(maps_df)

# parse agents en liste
maps_df['agent_list'] = maps_df['agent'].apply(parse_agents)

# déterminer gagnant de la map
maps_df['map_winner_determined'] = maps_df.apply(determine_map_winner, axis=1)

# colonne won (pour le joueur sur cette ligne map)
maps_df['won'] = (maps_df['player_team'] == maps_df['map_winner_determined']).astype(int)

# -----------------------------
# DÉDUPE : on veut des maps uniques par joueur
# -----------------------------
# table joueur+map unique (évite double comptage si plusieurs lignes identiques)
player_map_unique = maps_df[['player_name','_map_key','won','agent_list']].drop_duplicates(subset=['player_name','_map_key']).copy()

# map_played = nombre de maps uniques par joueur
player_map_stats = (
    player_map_unique.groupby('player_name')
    .agg(
        map_played = ('_map_key','count'),
        maps_won = ('won','sum')
    )
    .reset_index()
)
player_map_stats['map_winrate'] = np.where(
    player_map_stats['map_played'] > 0,
    (player_map_stats['maps_won'] / player_map_stats['map_played'] * 100).round(2),
    0.0
)

# -----------------------------
# CHOIX (stratégie PRIMARY AGENT) : prendre le premier agent listé par map
# -----------------------------
# extraire le "primary agent" (premier de la liste agent_list) par player+map
player_map_unique['primary_agent'] = player_map_unique['agent_list'].apply(
    lambda L: (L[0].strip() if isinstance(L, list) and len(L)>0 else np.nan)
)

# calcul par agent en utilisant primary_agent
agent_stats_primary = (
    player_map_unique.dropna(subset=['primary_agent'])
    .groupby(['player_name','primary_agent'])
    .agg(
        agent_maps_played = ('primary_agent','count'),
        agent_maps_won = ('won','sum')
    )
    .reset_index()
)

# dictionnaires par joueur
agent_played_dict_primary = agent_stats_primary.groupby('player_name').apply(
    lambda df: dict(zip(df['primary_agent'], df['agent_maps_played']))
).to_dict()

agent_win_dict_primary = agent_stats_primary.groupby('player_name').apply(
    lambda df: dict(zip(df['primary_agent'], df['agent_maps_won']))
).to_dict()

# -----------------------------
# MERGE AVEC TABLE PLAYERS + EXPORT
# -----------------------------
# nettoyer colonnes % dans players si besoin (kast, hs_percent, cl_percent)
for col in ['kast','hs_percent','cl_percent']:
    if col in players.columns:
        players[col] = players[col].astype(str).str.rstrip('%').replace('nan', np.nan)
        players[col] = pd.to_numeric(players[col], errors='coerce')

# colonnes à garder
cols_to_keep = [
    'player','player_name','team','roles','agents_played',
    'rating','acs','kd_ratio','kast','adr',
    'kpr','apr','fkpr','fdpr','hs_percent','cl_percent'
]
existing_cols = [c for c in cols_to_keep if c in players.columns]
df_moyennes = players[existing_cols].copy()
df_moyennes['player_name'] = df_moyennes['player_name'].astype(str).str.strip()

# merge des stats map
df_moyennes = df_moyennes.merge(
    player_map_stats[['player_name','map_played','maps_won','map_winrate']],
    on='player_name',
    how='left'
)

# remplacer agents_played par le dict primary (assure somme agents == map_played)
df_moyennes['agents_played'] = df_moyennes['player_name'].map(agent_played_dict_primary)
df_moyennes['agents_win_count'] = df_moyennes['player_name'].map(agent_win_dict_primary)

# valeurs par défaut
df_moyennes['map_played'] = df_moyennes['map_played'].fillna(0).astype(int)
df_moyennes['maps_won'] = df_moyennes['maps_won'].fillna(0).astype(int)
df_moyennes['map_winrate'] = df_moyennes['map_winrate'].fillna(0.0)

# remplacer NaN par {} pour colonnes dict
df_moyennes['agents_played'] = df_moyennes['agents_played'].apply(lambda x: x if isinstance(x, dict) else {})
df_moyennes['agents_win_count'] = df_moyennes['agents_win_count'].apply(lambda x: x if isinstance(x, dict) else {})

# arrondir numériques
numeric_cols = df_moyennes.select_dtypes(include='number').columns
df_moyennes[numeric_cols] = df_moyennes[numeric_cols].round(2)

# -----------------------------
# AJOUTER COLONNE ROLE PAR PLAYER
# -----------------------------
agents_df = pd.read_csv(f"/home/van-pc_ivan/Data_Ana/{year}/agent.csv")
agent_to_role = dict(zip(agents_df['Agent'], agents_df['Role']))

def agents_to_roles(agent_dict):
    """Convertit le dict d'agents joués en liste de rôles uniques"""
    roles_set = set()
    for agent in agent_dict.keys():
        role = agent_to_role.get(agent)
        if role:
            roles_set.add(role)
    return list(roles_set)

# créer colonne roles basée sur agents_played
df_moyennes['roles'] = df_moyennes['agents_played'].apply(agents_to_roles)


out_path = f"player_stats_moyennes_roles_{year}.csv"
df_moyennes.to_csv(out_path, index=False)

print(f"✅ Export : {out_path}")
print("Extrait (player_name, map_played, maps_won, map_winrate, agents_played, agents_win_count) :")
print(df_moyennes[['player_name','map_played','maps_won','map_winrate','agents_played','agents_win_count']].head(20).to_string(index=False))
