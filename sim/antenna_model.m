%% Petal-loaded hexagonal metamaterial patch antenna -- openEMS reconstruction
%
% Independent re-simulation of the antenna described in
% "Petal-Loaded Hexagonal Metamaterial Patch Antenna for Connected Vehicles".
% The original study used ANSYS HFSS; this model uses openEMS (FDTD) and is
% driven from MATLAB/Octave.
%
% Geometry was reconstructed from a published figure by pixel measurement;
% dimensions the source does not state are marked ASSUMED below.
%
% Run:  octave --no-gui -q antenna_model.m      (or run in MATLAB)

close all; clear; clc;
addpath('/opt/openEMS/share/openEMS/matlab');
addpath('/opt/openEMS/share/CSXCAD/matlab');
physical_constants;
unit = 1e-3;                       % all lengths in mm

% run from the script's own directory so petal.csv / ringpoly.m resolve
scriptdir = fileparts(mfilename('fullpath'));
if ~isempty(scriptdir), cd(scriptdir); end

%% ------------------------------------------------------------------ geometry
sub.a         = 12.2;              % substrate side            (paper: a)
sub.h         = 1.5;               % substrate thickness       (paper)
sub.epsR      = 4.4;               % FR4 permittivity          (paper)
sub.tand      = 0.02;              % FR4 loss tangent          (paper)

out.R         = 4.96;              % outer split ring, outer radius   (measured)
out.w         = 0.60;              % ring trace width                 (measured)
out.gap       = 1.00;              % split width  (paper: l1)
out.gap_ang   = 180;               % split at left

hex.R         = 3.55;              % hexagonal split ring, circumradius (measured)
hex.w         = 0.57;              % ring trace width                   (measured)
hex.gap       = 1.00;              % split width  (paper: l1)
hex.gap_ang   = 0;                 % split at right (180 deg from outer ring)

feed.w        = 1.00;              % feed width   (paper: w3)
feed.l        = 1.40;              % feed length from substrate edge to ring (paper l2 = 1.17)

gnd.h         = 6.10;              % partial ground height (paper: l3)

srr.cx        = 3.00;              % SRR centre offset from substrate centre (measured)
srr.cy        = 2.90;              % SRR centre height above substrate centre (measured)
srr.R2        = 2.53;              % outer ring outer radius  (paper: R2 = 2.5)
srr.w         = 0.55;              % ring trace width                  (measured)
srr.R3in      = 0.98;              % inner ring inner radius  (paper: R3 = 1)
srr.Rin_out   = 1.53;              % inner ring outer radius  (= R2 - l4, paper l4 = 1)
srr.gap_out   = 0.80;              % outer ring split, at top          (ASSUMED, measured)
srr.gap_in    = 0.35;              % inner ring split, at bottom       (ASSUMED, measured)

petal = load('petal.csv');         % petal outline traced from Figure 1 (R1 region)
petal = petal.';                   % 2 x N

%% ------------------------------------------------------------------- FDTD
% Discretisation can be overridden from the environment for convergence
% studies; the defaults are what the reported result was produced with.
res    = envopt('ANT_RES',  0.10);     % in-plane cell size over the board [mm]
air    = envopt('ANT_AIR',  16);       % air padding [mm]
nrts   = envopt('ANT_NRTS', 60000);    % max timesteps
subcells = envopt('ANT_SUBZ', 5);      % mesh lines through the substrate
outfile  = getenv('ANT_OUT'); if isempty(outfile), outfile = 's11.csv'; end

f_min = 1e9; f_max = 14e9;
f0 = 7.5e9; fc = 6.5e9;
FDTD = InitFDTD('NrTs', nrts, 'EndCriteria', 1e-5);
FDTD = SetGaussExcite(FDTD, f0, fc);
FDTD = SetBoundaryCond(FDTD, {'MUR','MUR','MUR','MUR','MUR','MUR'});

CSX = InitCSX();
mesh.x = [-sub.a/2-air  sub.a/2+air];
mesh.y = [-sub.a/2-air  sub.a/2+air];
mesh.z = [-air          sub.h+air];

%% -------------------------------------------------------------- substrate
CSX = AddMaterial(CSX, 'FR4');
kappa = sub.tand * 2*pi*f0 * EPS0 * sub.epsR;
CSX = SetMaterialProperty(CSX, 'FR4', 'Epsilon', sub.epsR, 'Kappa', kappa);
CSX = AddBox(CSX, 'FR4', 0, [-sub.a/2 -sub.a/2 0], [sub.a/2 sub.a/2 sub.h]);
mesh.z = [mesh.z linspace(0, sub.h, subcells)];

%% ------------------------------------------------------------- top layer
CSX = AddMetal(CSX, 'patch');
CSX = AddPolygon(CSX, 'patch', 10, 2, sub.h, ringpoly(0,0,out.R,out.w,out.gap,out.gap_ang,0));
CSX = AddPolygon(CSX, 'patch', 10, 2, sub.h, ringpoly(0,0,hex.R,hex.w,hex.gap,hex.gap_ang,6));
CSX = AddPolygon(CSX, 'patch', 10, 2, sub.h, petal);
CSX = AddBox(CSX, 'patch', 10, [-feed.w/2 -sub.a/2 sub.h], [feed.w/2 -sub.a/2+feed.l sub.h]);

%% ---------------------------------------------------------- bottom layer
CSX = AddMetal(CSX, 'gnd');
CSX = AddBox(CSX, 'gnd', 10, [-sub.a/2 -sub.a/2 0], [sub.a/2 -sub.a/2+gnd.h 0]);

CSX = AddMetal(CSX, 'srr');
for s = [-1 1]
  cx = s*srr.cx;  cy = srr.cy;
  CSX = AddPolygon(CSX, 'srr', 10, 2, 0, ringpoly(cx,cy,srr.R2,      srr.w, srr.gap_out,  90, 0));
  CSX = AddPolygon(CSX, 'srr', 10, 2, 0, ringpoly(cx,cy,srr.Rin_out, srr.Rin_out-srr.R3in, srr.gap_in, 270, 0));
end

%% ---------------------------------------------------------------- feed port
start = [-feed.w/2  -sub.a/2  0];
stop  = [ feed.w/2  -sub.a/2  sub.h];
[CSX port] = AddLumpedPort(CSX, 5, 1, 50, start, stop, [0 0 1], true);

%% -------------------------------------------------------------------- mesh
% Uniform fine grid over the antenna footprint, graded out into the air box.
% (DetectEdges is deliberately not used on the ring polygons: their 240-point
% arcs would seed a mesh line per vertex.)
% 12.2 / 0.10 is an integer and the feed edges (+-0.5), the ground edge (y=0)
% and the feed end (y=-4.7) all fall exactly on grid lines, so no sliver cells
% are created -- a single thin cell would collapse the FDTD timestep.
mesh.x = unique([mesh.x, -sub.a/2:res:sub.a/2]);
mesh.y = unique([mesh.y, -sub.a/2:res:sub.a/2]);
mesh.z = unique([mesh.z, linspace(0, sub.h, subcells)]);
assert(min(diff(mesh.x)) > 0.9*res && min(diff(mesh.y)) > 0.9*res, 'sliver cell in mesh');
fprintf('meshing (%d/%d/%d lines)...\n', numel(mesh.x), numel(mesh.y), numel(mesh.z));
fflush(stdout);
mesh = SmoothMesh(mesh, 1.2);
fprintf('mesh done, min cell %.3f mm\n', min([diff(mesh.x) diff(mesh.y) diff(mesh.z)]));
fflush(stdout);
CSX = DefineRectGrid(CSX, unit, mesh);

nf2ff_start = [mesh.x(4) mesh.y(4) mesh.z(4)];
nf2ff_stop  = [mesh.x(end-3) mesh.y(end-3) mesh.z(end-3)];
[CSX nf2ff] = CreateNF2FFBox(CSX, 'nf2ff', nf2ff_start, nf2ff_stop);

fflush(stdout);
fprintf('mesh cells: %d x %d x %d = %.2f M\n', numel(mesh.x), numel(mesh.y), ...
       numel(mesh.z), numel(mesh.x)*numel(mesh.y)*numel(mesh.z)/1e6);

%% --------------------------------------------------------------- run solver
Sim_Path = ['tmp_' strrep(outfile,'.csv','')];  Sim_CSX = 'antenna.xml';
CleanupSimPath(Sim_Path);
WriteOpenEMS([Sim_Path '/' Sim_CSX], FDTD, CSX);
RunOpenEMS(Sim_Path, Sim_CSX, '--numThreads=4');

%% ------------------------------------------------------------- postprocess
freq = linspace(f_min, f_max, 1401);
port = calcPort(port, Sim_Path, freq);
s11  = port.uf.ref ./ port.uf.inc;
s11dB = 20*log10(abs(s11));
Zin  = port.uf.tot ./ port.if.tot;
csvwrite(outfile, [freq(:)/1e9, s11dB(:), real(Zin(:)), imag(Zin(:))]);

fprintf('\n===================== RESONANCES (S11 < -10 dB) =====================\n');
% Band edges are interpolated onto the -10 dB crossing. Taking the nearest
% sample instead biases the bandwidth by up to two frequency steps, which is
% tens of MHz here -- the same order as the bandwidths themselves.
for k = 2:numel(freq)-1
  if s11dB(k) < -10 && s11dB(k) < s11dB(k-1) && s11dB(k) <= s11dB(k+1)
    lo = k; while lo > 1 && s11dB(lo) < -10, lo = lo - 1; end
    hi = k; while hi < numel(freq) && s11dB(hi) < -10, hi = hi + 1; end
    flo = freq(lo) + (freq(lo+1)-freq(lo)) * (-10 - s11dB(lo)) / (s11dB(lo+1) - s11dB(lo));
    fhi = freq(hi-1) + (freq(hi)-freq(hi-1)) * (-10 - s11dB(hi-1)) / (s11dB(hi) - s11dB(hi-1));
    fprintf('  f = %6.2f GHz   S11 = %7.2f dB   BW(-10dB) = %5.0f MHz (%.1f%%)\n', ...
            freq(k)/1e9, s11dB(k), (fhi-flo)/1e6, 100*(fhi-flo)/freq(k));
  end
end
fprintf('====================================================================\n');
