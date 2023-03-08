import dask.array as da
import numpy as np
import pandas as pd
from pathlib import Path
import re
import xarray as xr

class AMReXDatasetMeta():
    '''
    A high level collection of metadata for the entire plotfile,
    
    Heavily based off the BoxlibDataset class from yt
    '''
    
    def __init__(self, fplt):
        '''
        Inputs
        * fplt: the path to the plotfile (str, Path)
        '''
        self.fplt = Path(fplt)
        self.fdataset_header = Path(fplt, 'Header')
        
        # Domain offset; doesn't necessarily need to equal (0,0,0)
        self.domain_offset = np.zeros(3, dtype="int64")  # from static_output.py
        self.geometry = "cartesian"  # TODO: hardcoded
        
        self._parse_dataset_header()
        
    def _parse_dataset_header(self):
        '''
        Parse the Header file
        '''
        ### Open the header file
        with open(self.fdataset_header) as header_file:
        
            ### Parse the header file, line-by-line
            orion_version = header_file.readline().rstrip()  # Not used
            self.n_fields = int(header_file.readline())
            self.field_list = [header_file.readline().strip() for i in range(self.n_fields)]
            self.dimensionality = int(header_file.readline())
            self.current_time = float(header_file.readline())
            self.max_level = int(header_file.readline())
            
            ## Domain edges
            domain_left_edge = np.zeros(self.dimensionality, dtype=float)
            domain_left_edge[:] = header_file.readline().split()
            self.domain_left_edge = domain_left_edge
            
            domain_right_edge = np.zeros(self.dimensionality, dtype=float)
            domain_right_edge[:] = header_file.readline().split()
            self.domain_right_edge = domain_right_edge

            ## Refinement factors (should always be None or 2 for AMR-Wind)
            ref_factors = np.array(header_file.readline().split(), dtype="int64")
            if ref_factors.size == 0:
                # We use a default of two
                ref_factors = [2] * (self.max_level + 1)
            # We can't vary refinement factors based on dimension, or whatever else
            # they are varied on.
            self.ref_factors = ref_factors
            
            if np.unique(ref_factors).size > 1:  # Edge case for varying refinement factors
                # We want everything to be a multiple of this.
                self.refine_by = min(ref_factors)
                # Check that they're all multiples of the minimum.
                if not all(
                    float(rf) / self.refine_by == int(float(rf) / self.refine_by)
                    for rf in ref_factors
                ):
                    raise RuntimeError
                base_log = np.log2(self.refine_by)
                self.level_offsets = [0]  # level 0 has to have 0 offset
                lo = 0
                for rf in self.ref_factors:
                    lo += int(np.log2(rf) / base_log) - 1
                    self.level_offsets.append(lo)
            else:
                self.refine_by = ref_factors[0]
                self.level_offsets = [0 for l in range(self.max_level + 1)]
            
            ## Global index space
            index_space = header_file.readline()
            # This will be of the form:
            #  ((0,0,0) (255,255,255) (0,0,0)) ((0,0,0) (511,511,511) (0,0,0))
            root_space = index_space.replace("(", "").replace(")", "").split()[:2]
            start = np.array(root_space[0].split(","), dtype="int64")
            stop = np.array(root_space[1].split(","), dtype="int64")
            dd = np.ones(3, dtype="int64")
            dd[: self.dimensionality] = stop - start + 1
            self.domain_offset[: self.dimensionality] = start  # Specifically, Level 0 offset; TODO: rename?
            self.domain_dimensions = dd  # Specifically, Level 0 dimensions; TODO: rename?
            
            header_file.readline()  # Skip timesteps per level
            
            ## Grid resolution per level
            grid_resolution = np.zeros((self.max_level+1, self.dimensionality))
            for i in range(self.max_level + 1):
                gridline = header_file.readline().split()
                if len(gridline) != self.dimensionality:
                    raise ValueError(f"Unable to parse line containing grid resolution info, '{gridline}'")
                grid_resolution[i,:] = gridline
            self.dx = grid_resolution
        
class AMReXFabsMetaSingleLevel():
    '''
    Collect the following metadata on the un-sorted fabs for a single level
    * low index, high index, associated name of Cell_D, and byte offset with Cell_D
    '''
    def __init__(self, fplt, n_fields, dimensionality, level):
        '''
        Inputs
        * fplt: the path to the plotfile (str, Path)
        * max_level: (int)
        '''
        self.fplt = Path(fplt)
        self.n_fields = n_fields
        self.dimensionality = dimensionality
        
        self.level = level
        self.fheader_file = Path(self.fplt, f'Level_{level}/Cell_H')
        
        self._parse_level_header()
        
        # for level in range(1):  # TODO: Support multiple levels
        #     self._parse_level_header(level)
        
    def _parse_level_header(self):
        '''
        Parse the Cell_H file for a single refinement level
        '''
        ### Open Cell_H
        with open(self.fheader_file) as header_file:
        
            ### Parse Cell_H
            ## Ski first two lines (header file version, how data was written)
            header_file.readline()
            header_file.readline()
            
            ## Double check number of fields
            cell_h_n_fields = int(header_file.readline())
            assert self.n_fields == cell_h_n_fields, f"Mismatch in number of fields, {self.n_fields} vs {cell_h_n_fields}"
            
            ## Number of ghost cells
            self.nghost = int(header_file.readline())
            
            ## Count number of FABs in this level
            self.nfabs = int(header_file.readline().split()[0][1:])
            
            ## Collect info on each fab
            # Set up regex
            _1dregx = r"-?\d+"
            _2dregx = r"-?\d+,-?\d+"
            _3dregx = r"-?\d+,-?\d+,-?\d+"
            _dim_finder = [
                re.compile(rf"\(\(({ndregx})\) \(({ndregx})\) \({ndregx}\)\)$")
                for ndregx in (_1dregx, _2dregx, _3dregx)
            ]
            _our_dim_finder = _dim_finder[self.dimensionality - 1]

            # Collect index info
            fab_inds_lo = np.zeros((self.nfabs, self.dimensionality), dtype=int)  # Initialize
            fab_inds_hi = np.zeros((self.nfabs, self.dimensionality), dtype=int)
            for fabnum in range(self.nfabs):
                start, stop = _our_dim_finder.match(header_file.readline()).groups()
                start = np.array(start.split(","), dtype=int)
                stop = np.array(stop.split(","), dtype=int)
                fab_inds_lo[fabnum,:] = start
                fab_inds_hi[fabnum,:] = stop + 1  # Offset by 1 because Python counting

            # Ensure we have read all fabs
            endcheck = header_file.readline()
            assert endcheck == ')\n', f"Did not collect all fab index info! Next line reads '{endcheck}'"
            header_file.readline()  # Skip next line
            
            # Collect filenames and byte offset info
            fab_filename = np.zeros(self.nfabs, dtype=object)
            fab_byte_offset = np.zeros(self.nfabs, dtype=int)
            for fabnum in range(self.nfabs):
                _, filename, byte_offset = header_file.readline().split()
                fab_filename[fabnum] = filename
                fab_byte_offset[fabnum] = byte_offset
                
            ## Store fab info in a dataframe
            df_cols = ['lo_i', 'lo_j', 'lo_k',
                    'hi_i', 'hi_j', 'hi_k',
                    'filename', 'byte_offset']
            df_meta = pd.DataFrame(columns=df_cols)
            df_meta.index.name = 'fab_id'
            df_meta['lo_i'] = fab_inds_lo[:,0]
            df_meta['lo_j'] = fab_inds_lo[:,1]
            df_meta['lo_k'] = fab_inds_lo[:,2]
            df_meta['hi_i'] = fab_inds_hi[:,0]
            df_meta['hi_j'] = fab_inds_hi[:,1]
            df_meta['hi_k'] = fab_inds_hi[:,2]
            df_meta['filename'] = fab_filename
            df_meta['byte_offset'] = fab_byte_offset
            
            ## Calculate some extra info about the fabs
            df_meta['di'] = df_meta['hi_i'] - df_meta['lo_i']
            df_meta['dj'] = df_meta['hi_j'] - df_meta['lo_j']
            df_meta['dk'] = df_meta['hi_k'] - df_meta['lo_k']
            df_meta['ncells'] = df_meta['di'] * df_meta['dj'] * df_meta['dk']
                
            # Pre-compute the offset that skips the line of metadata for the fab
            # TODO: This is the slowest block of code. Should we try and speed it up?
            fab_data_offset = np.zeros(self.nfabs, dtype=int)
            for i, fab in df_meta.iterrows():
                ffab = Path(self.fplt, f"Level_{self.level}/{fab['filename']}")
                with open(ffab, 'rb') as f:
                    f.seek(fab['byte_offset'])
                    f.readline()  # Always skip the first line
                    fab_data_offset[i] = f.tell()    
            df_meta['data_offset'] = fab_data_offset
                
            ## Save out fab info
            self.metadata = df_meta.copy()  # TODO: Unnecessary copy?
        
class AMReXArray():
    '''
    Read in a pandas DataFrame that contains metadata on all the fabs
      for a single field, e.g., temperature, and return a dask/numpy 
      array of assembled data
    '''

    def __init__(self, meta_df, fplt, level, fieldname, field_list, ilo, ihi, jlo, jhi, klo, khi, dx, dy, dz, to_dask=True):
        self.fplt = fplt
        self.level = level
        self.fieldnum = field_list.index(fieldname)

        self._calc_coords(ilo, ihi, jlo, jhi, klo, khi, dx, dy, dz)
        self._calc_fabsize_fabblock(meta_df)
        self._check_meta_inds(meta_df, ilo, jlo, klo)
        self._create_array(meta_df, fieldname, to_dask)

    def reshape_fortran(self, x, shape):
        '''
        Used to reshape in the Fortran order, which Dask can't do itself

        From https://stackoverflow.com/questions/45479325/reshaping-a-dask-array-in-fortran-contiguous-order
        '''
        return x.T.reshape(shape[::-1]).T
    
    def mmap_load_fab(self, meta_fab, fieldnum, level, fplt):
        '''
        Lazily load one fab through numpy memory mapping

        Parameters
        ----------

        metafab: metadata from one row of the DataFrame from AMReXFabsSingleLevel
        fieldnum: (int) the position of the desired data field relative to the other fields
        level: (int) the grid refinement level
        fplt: (str, Path) the directory to plt#####

        Returns
        -------
        fab: a memory mapped numpy array of data

        TODO
        '''
        # Helper info
        ncells = meta_fab['ncells']
        fieldoffset = ncells * fieldnum * 8  # The *8 comes from bytesize of float
        totaloffset = fieldoffset + meta_fab['data_offset']
                
        # Read one chunk of data
        ffab = Path(fplt, f"Level_{level}/{meta_fab['filename']}")
        with open(ffab, 'rb') as f:
            fab = np.memmap(f, mode='r', shape=ncells, dtype=float, offset=totaloffset, order='F')
            
        return fab, np.array((meta_fab['di'], meta_fab['dj'], meta_fab['dk']))

    def _calc_coords(self, ilo, ihi, jlo, jhi, klo, khi, dx, dy, dz):
        '''
        Set up (x,y,z) coordinates
        '''
        self.nx = ihi - ilo
        self.ny = jhi - jlo  # TODO: possible bug?
        self.nz = khi - klo

        xlo = ilo*dx
        xhi = ihi*dx
        ylo = jlo*dy
        yhi = jhi*dy
        zlo = klo*dz
        zhi = khi*dz

        xcoords = np.arange(xlo, xhi, dx)
        ycoords = np.arange(ylo, yhi, dy)
        zcoords = np.arange(zlo, zhi, dz)
        self.coords = {'x':xcoords, 'y':ycoords, 'z':zcoords}

        assert len(self.coords['x']) == self.nx, "Mismatch in x-coordinate length!"
        assert len(self.coords['y']) == self.ny, "Mismatch in y-coordinate length!"
        assert len(self.coords['z']) == self.nz, "Mismatch in z-coordinate length!"

    def _calc_fabsize_fabblock(self, meta_df):
        '''
        Calculate the size of each fab and the
          number of fabs in each direction
        Currently working towards accomodating non-uniform fab sizes
        ASSUME: If there are multiple di/dj/dk values,
          every fab aside from those one row/column
          are the maximum di/dj/dk (TODO: generalize and check)
        '''
        ## Fab size calculation
        #    We use a list to account for multiple possible fab sizes
        di_list = list(meta_df['di'].value_counts().index)
        dj_list = list(meta_df['dj'].value_counts().index)
        dk_list = list(meta_df['dk'].value_counts().index)

        di_list.sort(reverse=True)
        dj_list.sort(reverse=True)
        dk_list.sort(reverse=True)

        # ASSUME: the length of above lists is not longer than 2
        assert len(di_list) <= 2, "The number of unique di values must currently be no larger than 2!"
        assert len(dj_list) <= 2, "The number of unique dj values must currently be no larger than 2!"
        assert len(dk_list) <= 2, "The number of unique dk values must currently be no larger than 2!"

        di_max = max(di_list)
        dj_max = max(dj_list)
        dk_max = max(dk_list)

        self.fabsize_max = np.array((di_max,dj_max,dk_max))

        ## Number of fabs in each direction
        nfabblocks = np.array((self.nx, self.ny, self.nz)) / self.fabsize_max
        self.nfabblocks = np.ceil(nfabblocks).astype(int)  

    def _check_meta_inds(self, meta_df, ilo, jlo, klo):
        '''
        Check that the metadata dataframe is sorted in the required 
          order to work with da.block()
        '''
        expected_inds = []
        for k in range(self.nfabblocks[2]):
            k_contrib = k*self.fabsize_max[2] + klo
            for j in range(self.nfabblocks[1]):
                j_contrib = j*self.fabsize_max[1] + jlo
                for i in range(self.nfabblocks[0]):
                    i_contrib = i*self.fabsize_max[0] + ilo
                    expected_inds.append([i_contrib, j_contrib, k_contrib])
        expected_inds = np.array(expected_inds)

        expected_flag = np.sum(expected_inds != meta_df[['lo_i', 'lo_j', 'lo_k']].values)
        assert expected_flag == 0, f"The fabs in the metadata dataframe are sorted in an unsupported order! Expected inds: {expected_inds}"

    def _create_array(self, meta_df, fieldname, to_dask=True):
        '''
        Create the dask array (saved out to an Xarray Dataset) or 
          numpy array
        '''
        ## Get list of fabs
        lazy_fabs, fab_shapes = zip(*[self.mmap_load_fab(meta_fab,self.fieldnum,self.level,self.fplt) for _,meta_fab in meta_df.iterrows()])

        # Organize fab list into nested lists formatted for da.block()
        lazy_blocks = []
        counter_ind = 0
        for i in range(self.nfabblocks[0]):
            jblock = []
            i_contrib = i
            for j in range(self.nfabblocks[1]):
                kblock = []
                j_contrib = j*self.nfabblocks[0]
                for k in range(self.nfabblocks[2]):
                    k_contrib = k*self.nfabblocks[0]*self.nfabblocks[1]
                    counter_ind = i_contrib + j_contrib + k_contrib
                    kblock.append(self.reshape_fortran(lazy_fabs[counter_ind],fab_shapes[counter_ind]))
                jblock.append(kblock)
            lazy_blocks.append(jblock)

        # Create final array
        if to_dask:
            daskarr = da.block(lazy_blocks)
            ds = xr.Dataset(coords=self.coords)
            ds[fieldname] = (('x', 'y', 'z'), daskarr)
            self.data = ds
        else:
            outarr = np.array(lazy_blocks)  # TODO: Test this
            self.data = outarr