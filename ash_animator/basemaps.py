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
    Attempt to create a writable cache directory.
    Return None if both preferred and fallback locations fail.
    """
    try:
        preferred = os.path.join("/code", f"{app_name}_cache")
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except Exception:
        try:
            fallback = os.path.join(tempfile.gettempdir(), f"{app_name}_cache")
            os.makedirs(fallback, exist_ok=True)
            return fallback
        except Exception:
            print(f"[Cache Disabled] Could not create cache for {app_name}.")
            return None

# Define cache dirs (may be None)
CTX_TILE_CACHE_DIR = get_cache_dir("contextily")
BASEMAP_TILE_CACHE_DIR = get_cache_dir("basemap")
CARTOPY_CACHE_DIR = get_cache_dir("cartopy")

# Set env vars if cartopy cache dir is usable
if CARTOPY_CACHE_DIR:
    os.environ["CARTOPY_USER_BACKGROUNDS"] = CARTOPY_CACHE_DIR
    os.environ["CARTOPY_CACHE_DIR"] = CARTOPY_CACHE_DIR

def draw_etopo_basemap(ax, mode="basemap", zoom=11):
    """
    Draws a background basemap image on a Cartopy GeoAxes.
    Falls back to simple features if caching or basemap rendering fails.
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
            cache_file = (os.path.join(BASEMAP_TILE_CACHE_DIR, f"{cache_key}_highres.png")
                          if BASEMAP_TILE_CACHE_DIR else None)

            if cache_file and os.path.exists(cache_file):
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

                if cache_file:
                    try:
                        fig.savefig(cache_file, dpi=300, bbox_inches='tight', pad_inches=0)
                    except Exception:
                        print("[Basemap Cache Write Failed] Skipping cache.")
                plt.close(fig)

                # img = Image.open(cache_file) if cache_file and os.path.exists(cache_file) else None
                # if img:
                #     ax.imshow(img, extent=extent, transform=ccrs.PlateCarree())

        else:
            raise ValueError(f"Unsupported basemap mode: {mode}")

    except Exception as e:
        print(f"[Basemap Error: {mode}] {e}")
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
