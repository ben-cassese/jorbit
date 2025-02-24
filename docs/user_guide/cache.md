# Cache management

``jorbit`` makes heavy use of Astropy's data caching services to download/store the files necessary for its operation. These include the files associated with the JPL DE 440 ephemeris (~765 MB), some basic mandatory files for the ``mpchecker`` functions (~660 MB), and additional files running the ``mpchecker`` functions at specific times (~250 MB / 30 day chunk). 

``jorbit`` will handle this downloading/caching automatically: when you first import ``jorbit``, it will automatically download and cache the necessary files, then when using the `mpchecker` functions, other files will similarly be automatically downloaded and cached. A warning will be issued each time a new file is downloaded, but if running on a shared system or if you have disk space concerns, be sure to keep track of your cache size.

By default, the ``mpchecker`` related files will "expire" 6 months after downloading, since by then there is potentially an updated version with more particles or more accurate orbits. The JPL DE 440 ephemeris files will not expire, since they are not updated and are necessary for all operations. ``jorbit`` will handle re-downloading expired files, but be aware that it's possible you'll need an internet connection even if you previously thought the files were cached.

```{note}
If you ever run into issues with the cache, either since files are corrupted or just because it's too large, you can clear all the jorbit-related files from your system with ``from jorbit.utils.cache import clear_jorbit_cache; clear_jorbit_cache()``.
```
