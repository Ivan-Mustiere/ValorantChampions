import pandas as pd
import numpy as np
import ast

# Chargement des données
df = pd.read_csv('player_stats_moyennes_roles_2024.csv')

# Pré-calcul des moyennes et écarts-types pour certaines statistiques (normalisation de score)
stats_cols = ['acs', 'adr', 'kpr', 'kast']
stats_mean = {c: df[c].mean() for c in stats_cols}
stats_std  = {c: df[c].std()  for c in stats_cols}

def dynamic_score(value, stat_name):
    """
    Calcule un score "z-score" sur 100 centré autour de la moyenne de la colonne
    Pour chaque statistique, renvoie 50 si std=0. Couvert à [0, 100].
    """
    mean = stats_mean[stat_name]
    std = stats_std[stat_name]
    if std == 0:
        return 50
    z = (value - mean) / std
    score = 50 + z * 15
    return max(0, min(100, score))

def compute_player_score_raw(row):
    """
    Calcule un score agrégé pour un joueur à partir de plusieurs stats brutes et des bonus.
    Les bonus incluent la polyvalence de rôle et le volume de parties jouées.
    """
    try:
        games = float(row['map_played'])          # nb de maps jouées
        winrate = float(row['map_winrate'])       # winrate en %
        kda = float(row['kd_ratio'])
        acs = float(row['acs'])
        adr = float(row['adr'])
        kpr = float(row['kpr'])
        kast = float(row['kast'])
    except Exception:
        return 1

    # Cadrage des valeurs extrêmes
    winrate = max(0, min(100, winrate))
    kda = max(0.3, min(4.0, kda))
    kpr = max(0.1, min(1.5, kpr))

    # Calcul des sous-scores basés sur des heuristiques
    winrate_score = (winrate - 40) * 2
    winrate_score = max(0, min(100, winrate_score))

    kda_score = (kda - 1.0) * 50
    kda_score = max(0, min(100, kda_score))

    acs_score  = dynamic_score(acs,  'acs')
    adr_score  = dynamic_score(adr,  'adr')
    kpr_score  = dynamic_score(kpr,  'kpr')
    kast_score = dynamic_score(kast, 'kast')

    # Pondération globale
    score = (
        0.30 * winrate_score +
        0.15 * kda_score +
        0.15 * acs_score +
        0.15 * adr_score +
        0.15 * kpr_score +
        0.10 * kast_score
    )

    # Extraction des rôles et détection du nombre de rôles joués (polyvalence)
    roles_col = row.get('roles', None)
    player_roles = []
    if isinstance(roles_col, str):
        try:
            parsed = ast.literal_eval(roles_col)
            if isinstance(parsed, str):
                player_roles = [parsed]
            elif isinstance(parsed, list):
                player_roles = parsed
            else:
                player_roles = []
        except Exception:
            player_roles = []
    elif isinstance(roles_col, list):
        player_roles = roles_col
    else:
        player_roles = []

    player_roles = [str(r).strip("'\"") for r in player_roles]
    unique_roles = set(player_roles)
    nb_roles = len(unique_roles)
    # Bonus en fonction du nombre de rôles joués
    if nb_roles == 1:
        bonus_points = 0
    elif nb_roles == 2:
        bonus_points = 3
    elif nb_roles == 3:
        bonus_points = 5
    elif nb_roles >= 4:
        bonus_points = 7
    else:
        bonus_points = 0
    score += bonus_points

    # Bonus en fonction du nombre de maps jouées (volume/expérience)
    part_min = 5    # seuil min sans bonus
    part_max = 20   # seuil max bonus max
    max_bonus = 20  # bonus maximal (additif) si part_max atteint ou dépassé
    if games <= part_min:
        bonus_ratio = 0
    elif games >= part_max:
        bonus_ratio = 1
    else:
        bonus_ratio = (games - part_min) / (part_max - part_min)
    score += bonus_ratio * max_bonus

    return score

# Calcul des scores (non normalisés) pour chaque joueur
df['score_raw'] = df.apply(compute_player_score_raw, axis=1)

# Normalisation pour centrer la moyenne à 50, bornage à [1,100]
mean_raw = df['score_raw'].mean()
def normalize_score(raw, mean_raw):
    adjusted = raw + (50 - mean_raw)
    return int(max(1, min(100, adjusted)))

df['score'] = df['score_raw'].apply(lambda x: normalize_score(x, mean_raw))
df = df.drop(columns=['score_raw'])

# Sauvegarde des données scorées
df.to_csv('player_stats_moyennes_roles_2024_scored.csv', index=False)
