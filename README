# LBM-SWE-OBC-CUDA

COMMANDS:

Compilation:
make [-j [IN=<in>] [BN=<bn>] [PREC=<prec>] [EXE=<exe>]]
where
  <in>   = Method for class identification method. 
           Possible values = {1, 2, 3, 4}
           Default value = 1
  <bn>   = Method for branching.
           Possible values = {1, 2, 3}
           Default value = 1
  <prec> = Floating point precision.
           Possible values = {32, 64}
           Default value = 64
  <exe>  = Executable file name.
           Default value = LBM
           
Execution:
<exe> <config_file>
example
/user/example/path/bin/LBM /user/example/path/Config_40000.txt

INPUT FILES REQUIRED:

- Configuration file
 Located in the root directory.
 Contains the following data:

 Directory = Absolute path to the root folder.
 Scenario  = Name of the simulated scenario. We will refer to this as <SCE>.
 Test      = Label for a particular test. We will refer to this as <TEST>.
 MaxTime   = Number of time steps to be simulated.
 DtPlot    = How often (number of time steps) are output files generated.
 DtTS      = How often (number of time steps) are time series values registered.
 Tau       = LBM relaxation time.
 g         = Gravitational acceleration.
 Dt        = Seconds that a time step represents.
 Nblocks   = CUDA blocks size.

- Input file
 Located in the Inputs/ folder.
 Name of the file must be "<SCE>_<TEST>.txt".
 Format of the file:

 First line: Lx Ly Dx x0 y0
   Lx = Number of nodes along the x-axis.
   Ly = Number of nodes along the y-axis.
   Dx = Distance in meters between two adjacent nodes (both x-axis and y-axis).
   x0 = Coordinate in meters of the first node [0][0] in the x-axis.
   y0 = Coordinate in meters of the first node [0][0] in the y-axis.
 Next Ly*Lx lines (1 per node): B W Type
   B    = Bathymetry value.
   W    = Water level.
   Type = Type of node (dry, coast, wet).
  
- Time Series file
 Located in the Inputs/TS/ folder.
 Name of the file must be "<SCE>.txt"
 Format of the file:

 First line: NTS
   NTS = Number of points where time series will be registered.
 Next NTS lines (1 per time series point): x y
   x = Coordinate in meters of the point in the x-axis.
   y = Coordinate in meters of the point in the y-axis.
