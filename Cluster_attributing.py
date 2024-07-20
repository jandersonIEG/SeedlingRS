#Created by Jeff Anderson - Don't hesitate to email me with any questions - janderson@iegconsulting.com

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.warp

# Define file paths
MSOrtho = "Filepath/Multispectral_imagery.tif" 
classified_clusters = "Filepath/classified_clusters.shp"
all_clusters_updated = "Filepath/updated_clusters.shp"

epsg = "epsg:3155" # Not used, but good practice to have here 

def calculate_indices(red, nir, green):
    # Normalized Difference Vegetation Index (NDVI)
    ndvi = (nir - red) / (nir + red + 0.000001)  # added a small constant to avoid division by zero

    # Soil Adjusted Vegetation Index (SAVI)
    L = 0.16  # soil brightness correction factor - experiment in QGIS or Arc to find the right level
    savi = ((nir - red) / (nir + red + L)) * (1 + L)

    # Modified Soil Adjusted Vegetation Index (MSAVI)
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2

    # Green-Red Normalized Difference Vegetation Index (GRNVI)
    grnvi = ((green - red) / (green + red))

    # Green Normalized Difference Vegetation Index (GNDVI)
    gndvi = ((nir - green) / (nir + green))


    return ndvi, savi, msavi, grnvi, gndvi

def write_raster_data(data, file_path, file_meta):
    kwargs = file_meta.copy()
    kwargs.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(file_path, 'w', **kwargs) as dst:
        dst.write_band(1, data.astype(rasterio.float32))



def import_attributes_to_polygons(polygon_path, file_path, output):
    # Load polygons
    polygons = gpd.read_file(polygon_path)

    # Calculate size and shape attributes
    polygons['Area'] = polygons['geometry'].area
    polygons['Perimeter'] = polygons['geometry'].length
    # Example of additional shape metrics:
    polygons['Aspect_Ratio'] = polygons['geometry'].bounds.apply(
        lambda x: (x['maxx'] - x['minx']) / (x['maxy'] - x['miny']), axis=1)
    polygons['Compactness'] = (4 * np.pi * polygons['Area']) / (polygons['Perimeter'] ** 2)

    # Open raster
    with rasterio.open(file_path) as src:
        blue = src.read(1)  # assuming red is band 1
        green = src.read(2)  # assuming green is band 2
        red = src.read(3)  # assuming blue is band 3
        nir = src.read(5)  # assuming NIR is band 4
        red_edge = src.read(4)  # assuming RE is band 5
        # Get the dimensions of the raster
        height, width = src.shape

        # Calculate Indices
        ndvi, savi, msavi, grnvi, gndvi = calculate_indices(red, nir, green)

        # Assign raster values to polygons
        for index, polygon in polygons.iterrows():
            # Mask the raster with the polygon and calculate statistics
            mask = rasterio.features.geometry_mask([polygon['geometry']], transform=src.transform, out_shape=(height, width), invert=True, all_touched=True)
            red_mean = red[mask].mean()
            green_mean = green[mask].mean()
            blue_mean = blue[mask].mean()
            red_edge_mean = red_edge[mask].mean()  
            nir_mean = nir[mask].mean()
            ndvi_mean = ndvi[mask].mean()
            savi_mean = savi[mask].mean()
            msavi_mean = msavi[mask].mean()
            grnvi_mean = grnvi[mask].mean()
            gndvi_mean = gndvi[mask].mean()

            red_median = np.median(red[mask])
            green_median = np.median(green[mask])
            blue_median = np.median(blue[mask])
            red_edge_median = np.median(red_edge[mask])  
            nir_median = np.median(nir[mask])
            ndvi_median = np.median(ndvi[mask])
            savi_median = np.median(savi[mask])
            msavi_median = np.median(msavi[mask])
            grnvi_median = np.median(grnvi[mask])
            gndvi_median = np.median(gndvi[mask])
            
            red_75 = np.percentile(red[mask], 75)
            green_75 = np.percentile(green[mask], 75)
            blue_75 = np.percentile(blue[mask], 75)
            red_edge_75 = np.percentile(red_edge[mask], 75)
            nir_75 = np.percentile(nir[mask], 75)
            ndvi_75 = np.percentile(ndvi[mask], 75)
            savi_75 = np.percentile(savi[mask], 75)
            msavi_75 = np.percentile(msavi[mask], 75)
            grnvi_75 = np.percentile(grnvi[mask], 75)
            gndvi_75 = np.percentile(gndvi[mask], 75)

            red_90 = np.percentile(red[mask], 90)
            green_90 = np.percentile(green[mask], 90)
            blue_90 = np.percentile(blue[mask], 90)
            red_edge_90 = np.percentile(red_edge[mask], 90)
            nir_90 = np.percentile(nir[mask], 90)
            ndvi_90 = np.percentile(ndvi[mask], 90)
            savi_90 = np.percentile(savi[mask], 90)
            msavi_90 = np.percentile(msavi[mask], 90)
            grnvi_90 = np.percentile(grnvi[mask], 90)
            gndvi_90 = np.percentile(gndvi[mask], 90)

            red_max = red[mask].max()
            green_max = green[mask].max()
            blue_max = blue[mask].max()
            red_edge_max = red_edge[mask].max()
            nir_max = nir[mask].max()
            ndvi_max = ndvi[mask].max()
            savi_max = savi[mask].max()
            msavi_max = msavi[mask].max()
            grnvi_max = grnvi[mask].max()
            gndvi_max = gndvi[mask].max()

            # Assign values to polygon
            polygons.at[index, 'R_mean'] = red_mean
            polygons.at[index, 'G_mean'] = green_mean
            polygons.at[index, 'B_mean'] = blue_mean
            polygons.at[index, 'N_mean'] = nir_mean
            polygons.at[index, 'RE_mean'] = red_edge_mean
            polygons.at[index, 'NDVI_mean'] = ndvi_mean
            polygons.at[index, 'SAVI_mean'] = savi_mean
            polygons.at[index, 'MSAVI_mean'] = msavi_mean
            polygons.at[index, 'GRNVI_mean'] = grnvi_mean
            polygons.at[index, 'GNDVI_mean'] = gndvi_mean

            polygons.at[index, 'R_med'] = red_median
            polygons.at[index, 'G_med'] = green_median
            polygons.at[index, 'B_med'] = blue_median
            polygons.at[index, 'N_med'] = nir_median
            polygons.at[index, 'RE_med'] = red_edge_median
            polygons.at[index, 'NDVI_med'] = ndvi_median
            polygons.at[index, 'SAVI_med'] = savi_median
            polygons.at[index, 'MSAVI_med'] = msavi_median
            polygons.at[index, 'GRNVI_med'] = grnvi_median
            polygons.at[index, 'GNDVI_med'] = gndvi_median

            polygons.at[index, 'R_75'] = red_75
            polygons.at[index, 'G_75'] = green_75
            polygons.at[index, 'B_75'] = blue_75
            polygons.at[index, 'N_75'] = nir_75
            polygons.at[index, 'RE_75'] = red_edge_75
            polygons.at[index, 'NDVI_75'] = ndvi_75
            polygons.at[index, 'SAVI_75'] = savi_75
            polygons.at[index, 'MSAVI_75'] = msavi_75
            polygons.at[index, 'GRNVI_75'] = grnvi_75
            polygons.at[index, 'GNDVI_75'] = gndvi_75

            polygons.at[index, 'R_90'] = red_90
            polygons.at[index, 'G_90'] = green_90
            polygons.at[index, 'B_90'] = blue_90
            polygons.at[index, 'N_90'] = nir_90
            polygons.at[index, 'RE_90'] = red_edge_90
            polygons.at[index, 'NDVI_90'] = ndvi_90
            polygons.at[index, 'SAVI_90'] = savi_90
            polygons.at[index, 'MSAVI_90'] = msavi_90
            polygons.at[index, 'GRNVI_90'] = grnvi_90
            polygons.at[index, 'GNDVI_90'] = gndvi_90

            polygons.at[index, 'R_max'] = red_max
            polygons.at[index, 'G_max'] = green_max
            polygons.at[index, 'B_max'] = blue_max
            polygons.at[index, 'N_max'] = nir_max
            polygons.at[index, 'RE_max'] = red_edge_max
            polygons.at[index, 'NDVI_max'] = ndvi_max
            polygons.at[index, 'SAVI_max'] = savi_max
            polygons.at[index, 'MSAVI_max'] = msavi_max
            polygons.at[index, 'GRNVI_max'] = grnvi_max
            polygons.at[index, 'GNDVI_max'] = gndvi_max

    # Export polygons with new attributes
    polygons.to_file(output)

import_attributes_to_polygons(classified_clusters, MSOrtho, all_clusters_updated)
