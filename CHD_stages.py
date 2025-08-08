"""
Created on Thu Jun 26 07:01:55 2025

@author: krodberg
@author: scaffold code from ChatGPT
"""
import pandas as pd

# Define the model parameters (similar to defineMFmodel)


def define_mf_model():
    data = {
        'MFmodel': ['ECFTX', 'NPALM', 'LWCSIM', 'ECFM', 'WCFM', 'ECSM'],
        'res': [1250, 704, 1000, 2400, 2400, 1000],
        'xmin': [24352.0, 680961.0, 218436.0, 565465.0, 20665.0, 671436.756],
        'ymin': [983097.0, 840454.0, 441788.0, -44448.0, -44448.0, 142788.280],
        'nlays': [11, 3, 9, 7, 7, 5],
        'nrows': [603, 292, 553, 552, 552, 1060],
        'ncols': [740, 408, 512, 236, 236, 313],
        'nsp': [192, 14975, 192, 288, 288, 11688],
        'startYr': [1999, 1965, 1999, 1989, 1989, 1985],
        'code': ['MF2005-NWT', 'MF2000', 'MF2005',
                 'SEAWAT-2000', 'SEAWAT-2000', 'ECSM'],
        'freq': ['Month', 'Daily', 'Month', 'Month', 'Month', 'Daily'],
        'mpath': [
            "//whqhpc01p/hpcc_shared/dbandara/CFWI/ECFTX/Model/Transient/*.*",
            "//whqhpc01p/hpcc_shared/jgidding/LECSR/LOX18/*.*",
            "//ad.sfwmd.gov/dfsroot/data/wsd/MOD/LWCSASIAS/model/*.*",
            "//ad.sfwmd.gov/dfsroot/data/wsd/MOD/ECFM/MB/*.*",
            "//whqhpc01p/hpcc_shared/jgidding/WCFM/SENS/ORG2/*.*",
            "//whqhpc01p/hpcc_shared/krodberg/SEAWAT/ECSM/RUN/*.*"
        ]
    }
    df = pd.DataFrame(data)
    df.set_index('MFmodel', inplace=True)
    return df

# Calculate cell centroids


def calc_centroids(df, model_name):
    models = define_mf_model()
    m = models.loc[model_name]

    res = m['res']
    llx = m['xmin']
    lly = m['ymin']
    nrows = m['nrows']
    uly = lly + res * nrows

    df['centx'] = llx + (df['COL'] * res) - (res / 2)
    df['centy'] = uly - (df['ROW'] * res) + (res / 2)
    return df

# --- Main Execution ---


model_path = r"\\ad.sfwmd.gov\dfsroot\data\wsd\SUP\devel\source\Python\LookAtTheWells\look_at_wells_TSC"
# Read CHD cells table
chd_path = model_path + "/l1_5NW_LakeO_l1E.chd"
chd_cells = pd.read_csv(
    chd_path,
    sep=r'\s+',
    skiprows=3,
    comment="-",
    usecols=[1, 2, 4],
    names=['ROW', 'COL', 'STATION'],
    dtype={0: int, 1: int, 4: str}
)

# Calculate coordinates
chd_coords = calc_centroids(chd_cells, 'ECSM')

# Aggregate mean centroids per station
chds = chd_coords.groupby('STATION')[['centx', 'centy']].mean().reset_index()

ugen_stg_path = model_path + "/UGEN_StagesTide_upd050525_AlexUpdated.csv"
# Step 1: Read CSV, dropping 'id'
ugen_stgs = pd.read_csv(ugen_stg_path)
df = ugen_stgs.drop(columns=['id'])

# Step 2: Melt into long format — from wide to long
stages = df.melt(id_vars=['year', 'month', 'day'],
                 var_name='station', value_name='stage')

# Step 3: Create a proper datetime column
#         and Drop original date components if not needed
#         and keep just the stations for CHDs
stages['date'] = pd.to_datetime(stages[['year', 'month', 'day']])
stages = stages.drop(columns=['year', 'month', 'day'])
valid_stations = chd_coords['STATION'].unique()
CHD_stages = stages[stages['station'].isin(valid_stations)].copy()

# Step 4: Aggregate: compute min, max, mean stage per station
agg_stats = stages.groupby('station')['stage'].agg(['min', 'max',
                                                    'mean']).reset_index()
agg_stats = agg_stats.rename(columns={'station': 'STATION'})

# merge the tables
CHD_Stg_Pnts = pd.merge(agg_stats, chds, on='STATION', how='inner')

# Round coordinates to 3 decimals
CHD_Stg_Pnts['centx'] = CHD_Stg_Pnts['centx'].round(3)
CHD_Stg_Pnts['centy'] = CHD_Stg_Pnts['centy'].round(3)

# Write to CSV without scientific notation
chd_stg_pnts_path = model_path + "/CHD/CHD_STG_pnts.csv"
CHD_Stg_Pnts.to_csv(chd_stg_pnts_path, index=False, float_format='%.3f')
chd_stg_pnts_path = model_path + "/CHD/CHD_STG_timeseries.csv"
CHD_stages.to_csv(chd_stg_pnts_path, index=False, float_format='%.3f')
