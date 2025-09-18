import os
import unicodedata
import numpy as np
import pandas as pd

# Input / Output paths
IN_PATH = r"c:\Users\pdaadh\Desktop\Study 2\EQTd_DAi_25_cleaned 3_1 for Prince.xlsx"
OUT_CSV = os.path.join(os.path.dirname(IN_PATH), "EQTd_DAi_25_itemwise.csv")

# Helper to normalize diacritics for robust matching
norm = lambda s: unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii').lower() if isinstance(s,str) else s

# Load data
print(f"Reading: {IN_PATH}")
df = pd.read_excel(IN_PATH)

# Ensure orig_order follows Excel row order starting at 1
df['orig_order'] = np.arange(1, len(df) + 1)

# Identify item columns present
item_cols = [f"x{i}" for i in range(1, 58) if f"x{i}" in df.columns]

# Identify id/demographic columns to carry forward (only those present will be used)
id_cols_expected = [
    'IDCode','orig_order','grade','LANG1_school','LANG2_strong','sex','SEX2',
    'school_lang','home_lang','strong_lang','friend_lang',
    'self_perception_1','self_perception_2','self_perception_3','self_perception_4','self_perception_5','self_perception_6','MEAN_PERCPT',
    'OPLM_version','Missing_total','TOTAL','pTOTAL',
    'S1','S2','S3','S5','S6',
    'T10_theta_TOTAL','T10_theta_S1','T10_theta_S2','T10_theta_S3','T10_theta_S5','T10_theta_S6',
    'Total_Time_sum','Total_Time_count','Total_Time_sum_ms','Total_Time_sum_min'
]
id_cols = [c for c in id_cols_expected if c in df.columns]
assert 'IDCode' in id_cols, "IDCode column not found; cannot proceed"

# Build time group bases from columns
bases = [c[:-9] for c in df.columns if isinstance(c, str) and c.endswith('_Time_sum')]

# Desired mapping by normalized name to item index ranges
norm_to_items = {
    'rally': list(range(1, 11)),
    'vahennyslaskut1': list(range(11, 14)),
    'jakolaskut1': list(range(14, 17)),
    'geometriaperuskasitteet': list(range(17, 22)),
    'sadetilastot': list(range(22, 25)),
    'puuttuvalukuvahennyslaskut': list(range(25, 28)),
    'yhteenlaskut2': list(range(28, 32)),
    'vahennyslaskut2': list(range(32, 36)),
    'sanaarvoituspalkit2sana': list(range(36, 38)),
    'vahennyslaskut3': list(range(38, 41)),
    'kertolaskut2': list(range(41, 45)),
    'etaisyyskartalla': list(range(45, 48)),
    'puuttuvalukujakolasku': list(range(48, 50)),
    'looginenpaattely': list(range(50, 54)),
    'jakolaskut3': list(range(54, 57)),
    'korjaarobotinkoodi': [57],
}

# Canonical English names for each normalized base name
norm_to_canonical = {
    'rally': 'Rally',
    'vahennyslaskut1': 'Subtraction1',
    'jakolaskut1': 'Division1',
    'geometriaperuskasitteet': 'BasicGeometry',
    'sadetilastot': 'RainfallStatistics',
    'puuttuvalukuvahennyslaskut': 'MissingNumber_Subtraction',
    'yhteenlaskut2': 'Addition2',
    'vahennyslaskut2': 'Subtraction2',
    'sanaarvoituspalkit2sana': 'WordPuzzle',
    'vahennyslaskut3': 'Subtraction3',
    'kertolaskut2': 'Multiplication2',
    'etaisyyskartalla': 'MapDistance',
    'puuttuvalukujakolasku': 'MissingNumber_Division',
    'looginenpaattely': 'LogicalReasoning',
    'jakolaskut3': 'Division3',
    'korjaarobotinkoodi': 'DebugRobotCode',
}

# Map actual base name -> item indices and canonical name using normalized matching, skip Total
base_to_items = {}
base_to_canonical = {}
for b in bases:
    if b.lower().startswith('total'):
        continue
    nb = norm(b)
    matched = None
    if nb in norm_to_items:
        matched = nb
    else:
        for k in norm_to_items.keys():
            if nb.startswith(k) or (k in nb):
                matched = k
                break
    if matched is not None:
        base_to_items[b] = norm_to_items[matched]
        base_to_canonical[b] = norm_to_canonical.get(matched, b)

if not base_to_items:
    raise SystemExit("No time group base names matched to item ranges.")

print("Resolved group->items map (canonical names):")
for b, items in base_to_items.items():
    canon = base_to_canonical.get(b, b)
    print(f" - {canon}: x{items[0]}..x{items[-1]} ({len(items)} items)")

# Compute per-student per-group per-item time (even split within group)
rows = []
for b, items in base_to_items.items():
    sum_col = f"{b}_Time_sum"
    count_col = f"{b}_Time_count"
    s_sum = df[sum_col] if sum_col in df.columns else pd.Series([np.nan]*len(df))
    s_count = df[count_col] if count_col in df.columns else pd.Series([np.nan]*len(df))

    present_item_cols = [f"x{i}" for i in items if f"x{i}" in df.columns]
    group_size = len(present_item_cols)

    attempted = df[present_item_cols].notna().sum(axis=1) if group_size > 0 else pd.Series([0]*len(df))

    denom = np.where((~pd.isna(s_count)) & (s_count > 0), s_count,
                     np.where(attempted > 0, attempted,
                              np.where(group_size > 0, group_size, np.nan)))

    per_item_ms = np.where((~pd.isna(s_sum)) & (denom > 0), s_sum / denom, np.nan)

    tmp = pd.DataFrame({
        'IDCode': df['IDCode'],
        'group': base_to_canonical.get(b, b),
        'group_time_sum_ms': s_sum,
        'group_time_count': s_count,
        'group_attempted_items': attempted,
        'group_size_items': group_size,
        'per_item_time_ms': per_item_ms,
    })
    rows.append(tmp)

df_group = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['IDCode','group','per_item_time_ms'])

# Build item->group map for present items
item_map_rows = []
for b, items in base_to_items.items():
    for i in items:
        col = f"x{i}"
        if col in df.columns:
            item_map_rows.append({'item': col, 'item_index': i, 'group': base_to_canonical.get(b, b)})

df_item_map = pd.DataFrame(item_map_rows)

# Melt to long
df_items = df[id_cols + item_cols].melt(id_vars=id_cols, value_vars=item_cols, var_name='item', value_name='response')

# Merge item->group, then per-student per-group per-item time
df_items = df_items.merge(df_item_map, on='item', how='left')

df_items = df_items.merge(df_group[['IDCode','group','per_item_time_ms']], on=['IDCode','group'], how='left')

# Only assign time to non-missing responses
df_items['response_time_ms'] = np.where(df_items['response'].notna(), df_items['per_item_time_ms'], np.nan)

# Convenience seconds column
df_items['response_time_sec'] = df_items['response_time_ms'] / 1000.0

# Reorder/select columns
front_cols = ['IDCode'] + [c for c in id_cols if c != 'IDCode'] + ['item','item_index','group','response','response_time_ms','response_time_sec']
front_cols = [c for c in front_cols if c in df_items.columns]

df_out = df_items[front_cols]

# Group rows by student by sorting on orig_order (Excel row order) then item_index (keeps data the same)
df_out = df_out.sort_values(by=['orig_order', 'item_index']).reset_index(drop=True)

# Write CSV
print(f"Writing: {OUT_CSV}")
df_out.to_csv(OUT_CSV, index=False, encoding='utf-8')
print("Done.")
print(f"Rows: {len(df_out)}  Cols: {len(df_out.columns)}")
print(df_out.head(10).to_string())
