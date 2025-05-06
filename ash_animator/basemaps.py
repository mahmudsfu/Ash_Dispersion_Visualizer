
# import contextily as ctx
# from mpl_toolkits.basemap import Basemap
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# def draw_etopo_basemap(ax, mode="basemap", zoom=7):
#     try:
#         if mode == "stock":
#             ax.stock_img()
#         elif mode == "contextily":
#             extent = ax.get_extent(ccrs.PlateCarree())
#             ax.set_extent(extent, crs=ccrs.PlateCarree())
#             ctx.add_basemap(ax, crs=ccrs.PlateCarree(), source=ctx.providers.CartoDB.Voyager, zoom=zoom)
#         elif mode == "basemap":
#             extent = ax.get_extent(ccrs.PlateCarree())
#             m = Basemap(projection='cyl',
#                         llcrnrlon=extent[0], urcrnrlon=extent[1],
#                         llcrnrlat=extent[2], urcrnrlat=extent[3],
#                         resolution='h', ax=ax)
#             m.shadedrelief()
#             m.drawcoastlines(linewidth=0.5)
#             m.drawcountries(linewidth=0.7)
#             m.drawmapboundary()
#         else:
#             raise ValueError(f"Unsupported basemap mode: {mode}")
#     except Exception as e:
#         print(f"[Relief Error - {mode} mode]:", e)
#         ax.add_feature(cfeature.LAND)
#         ax.add_feature(cfeature.OCEAN)

import os
import hashlib
import contextily as ctx
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
import matplotlib.pyplot as plt

# Define cache directories
# Optional: Set tile cache directory (must be done before contextily downloads tiles)
os.environ["XDG_CACHE_HOME"] = os.path.expanduser("~/.contextily_cache")

CTX_TILE_CACHE_DIR = os.path.expanduser("~/.contextily_cache")
BASEMAP_TILE_CACHE_DIR = os.path.expanduser("~/.basemap_cache")

os.makedirs(CTX_TILE_CACHE_DIR, exist_ok=True)
os.makedirs(BASEMAP_TILE_CACHE_DIR, exist_ok=True)

def draw_etopo_basemap(ax, mode="basemap", zoom=11):
    """
    Draws a high-resolution basemap background on the provided Cartopy GeoAxes.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib Axes object (with Cartopy projection) to draw the map background on.

    mode : str, optional
        The basemap mode to use:
        - "stock": Default stock image from Cartopy.
        - "contextily": Web tile background (CartoDB Voyager), with caching.
        - "basemap": High-resolution shaded relief using Basemap, with caching.
        Default is "basemap".

    zoom : int, optional
        Tile zoom level (only for "contextily"). Higher = more detail. Default is 7.

    Notes
    -----
    - Uses high resolution for Basemap (resolution='h') and saves figure at 300 DPI.
    - Cached images are reused using extent-based hashing to avoid re-rendering.
    - Basemap is deprecated; Cartopy with web tiles is recommended for new projects.
    """
    try:
        if mode == "stock":
            ax.stock_img()

        elif mode == "contextily":
            extent = ax.get_extent(crs=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ctx.add_basemap(
                ax,
                crs=ccrs.PlateCarree(),
                source=ctx.providers.CartoDB.Voyager,
                zoom=zoom                
            )

        elif mode == "basemap":
            extent = ax.get_extent(crs=ccrs.PlateCarree())

            # Create a hash key for this extent
            extent_str = f"{extent[0]:.4f}_{extent[1]:.4f}_{extent[2]:.4f}_{extent[3]:.4f}"
            cache_key = hashlib.md5(extent_str.encode()).hexdigest()
            cache_file = os.path.join(BASEMAP_TILE_CACHE_DIR, f"{cache_key}_highres.png")

            if os.path.exists(cache_file):
                img = Image.open(cache_file)
                ax.imshow(img, extent=extent, transform=ccrs.PlateCarree())
            else:
                # Create a high-resolution temporary figure
                temp_fig, temp_ax = plt.subplots(figsize=(12, 9),
                                                 subplot_kw={'projection': ccrs.PlateCarree()})
                temp_ax.set_extent(extent, crs=ccrs.PlateCarree())

                m = Basemap(projection='cyl',
                            llcrnrlon=extent[0], urcrnrlon=extent[1],
                            llcrnrlat=extent[2], urcrnrlat=extent[3],
                            resolution='f', ax=temp_ax)  # 'h' = high resolution

                m.shadedrelief()
                # m.drawcoastlines(linewidth=0.1)
                # m.drawcountries(linewidth=0.1)
                # m.drawmapboundary()

                # Save high-DPI figure for clarity
                temp_fig.savefig(cache_file, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(temp_fig)

                # Load and display the cached image
                img = Image.open(cache_file)
                ax.imshow(img, extent=extent, transform=ccrs.PlateCarree())

        else:
            raise ValueError(f"Unsupported basemap mode: {mode}")

    except Exception as e:
        print(f"[Relief Error - {mode} mode]:", e)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
