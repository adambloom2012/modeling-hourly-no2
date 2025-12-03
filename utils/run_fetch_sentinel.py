import os
from fetch_sentinel_2_data import get_sentinel_data
import geopandas as gpd
import pandas as pd
import geopandas as gpd
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from shapely.geometry import box
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret, include_client_id=True)

df = pd.read_csv("/Users/adambloom/Downloads/hourly_42602_2025.csv")
df = df[(
    df['State Code'] == 6)
    & (df['County Code'] == 37)
    & (df['Site Num'].isin([1201]))
]
df = df[['Latitude', 'Longitude', 'Site Num', 'State Code', 'County Code']]
df = df.drop_duplicates()
df.head()
df['geometry'] = gpd.points_from_xy(df.Longitude, df.Latitude)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
gdf.head()

# --- Start of new code ---

# 1. Choose a projected CRS appropriate for your data's location
# We'll use GeoPandas' handy function to estimate the best UTM zone
projected_crs = gdf.estimate_utm_crs()
gdf_proj = gdf.to_crs(projected_crs)

# 2. Define the box dimensions in meters
# For a 1.2 km square, we need to go 600 meters in each direction from the center
half_side = 600  # 1200 meters / 2

# 3. Create the bounding box for each point in the projected CRS
# We apply a function to each geometry (point) in the projected GeoDataFrame
boxes = gdf_proj.geometry.apply(lambda point: box(
    point.x - half_side,
    point.y - half_side,
    point.x + half_side,
    point.y + half_side
))

# 4. Create a new GeoDataFrame with these boxes and project it back to WGS84 (EPSG:4326)
boxes_gdf = gpd.GeoDataFrame(geometry=boxes, crs=projected_crs)
boxes_4326 = boxes_gdf.to_crs("EPSG:4326")

# 5. Extract the bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
# The .bounds attribute gives us the coordinates we need
gdf['bbox'] = boxes_4326.bounds.values.tolist()

# Display the final result with the new 'bbox' column
gdf['bbox'].iloc[0]
gdf

date_of_im = get_sentinel_data(oauth, token, gdf)

print(date_of_im)
