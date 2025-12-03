import os


def get_sentinel_data(
        oauth,
        token,
        gdf,  # GeoDataFrame with 'bbox' column ) -> None:
) -> None:
    # Create a session

    evalscript = """
//VERSION=3
function setup() {
  return {
    input: [
      {
        bands: [
          "B01",
          "B02",
          "B03",
          "B04",
          "B05",
          "B06",
          "B07",
          "B08",
          "B8A",
          "B09",
          "B11",
          "B12",
        ],
        units: "DN",
      },
    ],
    output: {
      id: "default",
      bands: 12,
      sampleType: SampleType.UINT16,
    },
  }
}

function evaluatePixel(sample) {
  return [
    sample.B01,
    sample.B02,
    sample.B03,
    sample.B04,
    sample.B05,
    sample.B06,
    sample.B07,
    sample.B08,
    sample.B8A,
    sample.B09,
    sample.B11,
    sample.B12,
  ]
}
"""
    for state_code, county_id, site_id, geometry in zip(gdf['State Code'], gdf['County Code'], gdf['Site Num'], gdf['bbox']):
        file_id = f"{state_code}_{county_id}_{site_id}"
        request = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "bbox": geometry
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                                "dataFilter": {
                                    "timeRange": {
                                        "from": "2024-01-01T00:00:00Z",
                                        "to": "2024-12-31T00:00:00Z",
                                    },
                                    "mosaickingOrder": "leastCC"
                                },
                        "processing": {"harmonizeValues": "false"},
                    }
                ],
            },
            "output": {
                "width": 120,
                "height": 120,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/tiff"},
                    }
                ],
            },
            "evalscript": evalscript,
        }

        url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        response = oauth.post(url, json=request)
        date_of_image = response.headers.get("Date")

        # process response
        # save response as geoTIFF if 200
        if response.status_code == 200:
            # create directory if it doesn't exist
            os.makedirs(f"../data/{file_id}", exist_ok=True)
            with open(f"../data/{file_id}/{file_id}.tiff", "wb") as f:
                f.write(response.content)
        # print error message if not 200
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
        return date_of_image
