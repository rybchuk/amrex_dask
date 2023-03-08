# amrex_dask
This repo houses some code that can be used to read plotfiles from AMReX codes into numpy arrays or dask arrays, instead of interfacing with the data through e.g., [yt](https://amrex-codes.github.io/amrex/docs_html/Visualization.html#yt). At present, this repo is not a general purpose library, so users beware. My hope is that this code saves future AMReX analysts some time if they want to process data with numpy or dask. 

Some things to note:
* The code here was purpose-built to convert terabyte-scale AMR-Wind plotfiles to Zarr stores. These simulations were run using fixed refinement zones, which is an assumption that may not apply to output from your AMReX code. I also assume rectangular refinement zones.
* I refer to the small collections of data that AMReX writes out, sized e.g., (32,32,32), as "fabs". This probably isn't the most correct terminology, but in this context, I think this name is clearer than "grids" or "chunks".
* The code fundamentally works by assembling a bunch of [memory mapped](https://docs.dask.org/en/latest/array-creation.html#memory-mapping) numpy arrays into a numpy array or a dask array. I had problems getting this Dask code using a distributed scheduler, so as of now, it only works with a threaded scheduler. In theory, this means you might be limited to relatively smaller arrays, but I didn't have this issue. I believe this is a fixable problem. 
