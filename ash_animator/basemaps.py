import os
import tempfile
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import contextily as ctx
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_cache_dir(app_name):
    """
    Try to create a cache directory under ./code first.
    If it fails (e.g., due to permissions), fallback to system temp directory.
    """
    try:
        preferred = os.path.join("/code", f"{app_name}_cache")
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except PermissionError:
        fallback = os.path.join(tempfile.gettempdir(), f"{app_name}_cache")
        os.makedirs(fallback, exist_ok=True)
        return fallback

# Define and create cache directories
CTX_TILE_CACHE_DIR = get_cache_dir("contextily")
BASEMAP_TILE_CACHE_DIR = get_cache_dir("basemap")
CARTOPY_CACHE_DIR = get_cache_dir("cartopy")

# Set environment variables for Cartopy
os.environ["CARTOPY_USER_BACKGROUNDS"] = CARTOPY_CACHE_DIR
os.environ["CARTOPY_CACHE_DIR"] = CARTOPY_CACHE_DIR

def draw_etopo_basemap(ax, mode="basemap", zoom=11):
    """
    Draws a background basemap image on a Cartopy GeoAxes.
    """
    try:
        if mode == "stock":
            ax.stock_img()

        elif mode == "contextily":
            extent = ax.get_extent(crs=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ctx.add_basemap(ax, crs=ccrs.PlateCarree(),
                            source=ctx.providers.CartoDB.Voyager, zoom=zoom)

        elif mode == "basemap":
            extent = ax.get_extent(crs=ccrs.PlateCarree())
            extent_str = "_".join(f"{v:.4f}" for v in extent)
            cache_key = hashlib.md5(extent_str.encode()).hexdigest()
            cache_file = os.path.join(BASEMAP_TILE_CACHE_DIR, f"{cache_key}_highres.png")

            if os.path.exists(cache_file):
                img = Image.open(cache_file)
                ax.imshow(img, extent=extent, transform=ccrs.PlateCarree())
            else:
                fig, temp_ax = plt.subplots(figsize=(12, 9),
                                            subplot_kw={'projection': ccrs.PlateCarree()})
                temp_ax.set_extent(extent, crs=ccrs.PlateCarree())

                m = Basemap(projection='cyl',
                            llcrnrlon=extent[0], urcrnrlon=extent[1],
                            llcrnrlat=extent[2], urcrnrlat=extent[3],
                            resolution='f', ax=temp_ax)

                m.shadedrelief()
                fig.savefig(cache_file, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

                img = Image.open(cache_file)
                ax.imshow(img, extent=extent, transform=ccrs.PlateCarree())

        else:
            raise ValueError(f"Unsupported basemap mode: {mode}")

    except Exception as e:
        print(f"[Basemap Error: {mode}] {e}")
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
