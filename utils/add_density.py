import pandas as pd


df = pd.read_csv('/home/abloom/projects/Global-NO2-Estimation/data/samples_S2S5P_hourly_data.csv')

pop_dens = pd.read_csv('/home/abloom/data/pop_density.csv')


df.head()
pop_dens.columns
# keep only AqsSiteCode PopulationDensity  LocationType
pop_dens = pop_dens[['AqsSiteCode', 'PopulationDensity', 'LocationType']].drop_duplicates()
# check for duplicate AqsSiteCode
assert pop_dens['AqsSiteCode'].duplicated().sum() == 0, "Duplicate AqsSiteCode found in pop_dens"

# Split the AirQualityStation into components and pad each part correctly
def format_aqs_code(station_str):
    parts = station_str.split('_')
    if len(parts) == 3:
        state = parts[0].zfill(2)    # State: 2 digits
        county = parts[1].zfill(3)   # County: 3 digits
        site = parts[2].zfill(4)     # Site: 4 digits
        full_code = state + county + site
        # convert to integer
        full_code_int = int(full_code)
        return full_code_int
    return None

df['AqsSiteCode'] = df['AirQualityStation'].apply(format_aqs_code)

# Now merge
merged_df = df.merge(pop_dens, on='AqsSiteCode', how='left')
# make location type categorical and numeric
merged_df['LocationType'] = merged_df['LocationType'].astype('category').cat.codes
# check for nulls in PopulationDensity
# check df for where PopulationDensity is null
merged_df[merged_df['PopulationDensity'].isnull()]
# get distinct AirQualityStation where PopulationDensity is null
len(merged_df[merged_df['PopulationDensity'].isnull()]['AqsSiteCode'].unique())

# single comma seperated string of ids
','.join([str(x).zfill(9) if len(str(x)) == 8 else str(x) for x in merged_df[merged_df['PopulationDensity'].isnull()]['AqsSiteCode'].unique()])
missing = [
'110010041,'
'110010043,'
'110010050,'
'110010051,'
'261630101,'
'260810023,'
'371190050,'
'480290052,'
'490190007,'
'490353018,'
'490030005,'
'560330007,'
'060190011,'
'060670012,'
'060731025,'
'081230015,'
'080690015'
]

other_list = pd.read_csv('/home/abloom/data/other_list.csv')

# get otherlist where 'AQS Site ID' is in missing
other_list[other_list['AQS Site ID'].isin(missing)]

# drop nulls from merged_df
merged_df = merged_df[~merged_df['PopulationDensity'].isnull()]

merged_df.to_csv('/home/abloom/projects/Global-NO2-Estimation/data/samples_S2S5P_hourly_data_with_pop_density.csv', index=False)