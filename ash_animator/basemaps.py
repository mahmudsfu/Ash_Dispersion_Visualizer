import os
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import contextily as ctx
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_cache_dir(app_name):
    """
    Returns a writable cache directory path depending on the OS.
    Linux: tries /code/<app>_cache, falls back to /tmp/<app>_cache
    Windows: uses LOCALAPPDATA/<app>_cache
    """
    if os.name == 'nt':
        # Windows
        base_dir = os.getenv('LOCALAPPDATA', os.getcwd())
    else:
        # Unix
        base_dir = "/code"
        try:
            test_path = os.path.join(base_dir, f"{app_name}_cache")
            os.makedirs(test_path, exist_ok=True)
            os.chmod(test_path, 0o777)
            return test_path
        except PermissionError:
            print(f"[PermissionError] Cannot use {base_dir}, falling back to /tmp.")
            base_dir = "/tmp"
    
    cache_path = os.path.join(base_dir, f"{app_name}_cache")
    os.makedirs(cache_path, exist_ok=True)
    return cache_path

# Setup cache directories
CTX_TILE_CACHE_DIR = get_cache_dir("contextily")
BASEMAP_TILE_CACHE_DIR = get_cache_dir("basemap")
CARTOPY_CACHE_DIR = get_cache_dir("cartopy")

# Set Cartopy environment variables
os.environ["CARTOPY_USER_BACKGROUNDS"] = CARTOPY_CACHE_DIR
os.environ["CARTOPY_CACHE_DIR"] = CARTOPY_CACHE_DIR

def draw_etopo_basemap(ax, mode="basemap", zoom=11):
    """
    Draws a basemap onto a Cartopy GeoAxes object.
    Parameters
    ----------
    ax : Cartopy GeoAxes
    mode : 'stock' | 'contextily' | 'basemap'
    zoom : int (contextily zoom level)
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
