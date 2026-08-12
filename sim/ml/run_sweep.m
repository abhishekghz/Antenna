%% Generate the surrogate training set.
%
% Loops over designs.csv, simulates each one in openEMS, and appends its S11
% response to sweep_s11.csv. Runs inside a single Octave process so the
% ~2 s interpreter start-up is paid once rather than 300 times.
%
% Resumable: any design already present in sweep_s11.csv is skipped, so the
% sweep can be stopped and restarted without losing work.
%
% Fidelity is deliberately lower than antenna_model.m (0.2 mm cells, 20k
% timesteps) to fit 300 designs into a few hours. calibrate.m quantifies the
% resulting offset against the converged reference.

addpath('/opt/openEMS/share/openEMS/matlab');
addpath('/opt/openEMS/share/CSXCAD/matlab');
addpath(fileparts(pwd));            % ringpoly.m lives one level up
physical_constants;
unit = 1e-3;

% Sweep fidelity is deliberately coarse; override from the environment to
% re-run selected designs (e.g. an inverse-design candidate) at the reference
% fidelity used by ../antenna_model.m before trusting the result.
function v = ev(name, dflt)
  s = getenv(name);
  if isempty(s), v = dflt; else, v = str2double(s); end
end
RES   = ev('SWEEP_RES',  0.2);    % in-plane cell size [mm]
AIR   = ev('SWEEP_AIR',  10);     % air padding [mm]
SUBZ  = ev('SWEEP_SUBZ', 5);      % mesh lines through the substrate
NRTS  = ev('SWEEP_NRTS', 20000);  % max timesteps
NF    = 201;        % frequency samples stored per design
FLO   = 3e9; FHI = 13e9;

% ---- fixed geometry (not design variables) --------------------------------
sub.a = 12.2; sub.h = 1.5; sub.epsR = 4.4; sub.tand = 0.02;
feed.w = 1.00; feed.l = 1.40;
srr.cx = 3.00; srr.cy = 2.90; srr.R2 = 2.53; srr.w = 0.55;
srr.Rin_out = 1.53; srr.R3in = 0.98; srr.gap_out = 0.80; srr.gap_in = 0.35;
petal0 = load('../petal.csv').';

dfile = getenv('SWEEP_DESIGNS'); if isempty(dfile), dfile = 'designs.csv'; end
designs = csvread(dfile, 1, 0);
ndes = rows(designs);

% Shard the sweep across independent Octave processes. Small meshes thread
% poorly, so N single-threaded processes beat one N-threaded process.
shard  = str2double(getenv('SWEEP_SHARD'));  if isnan(shard),  shard  = 0; end
nshard = str2double(getenv('SWEEP_NSHARD')); if isnan(nshard), nshard = 1; end
freq = linspace(FLO, FHI, NF);

otag = getenv('SWEEP_TAG'); if isempty(otag), otag = 'sweep'; end
outfile = sprintf('%s_s11_%d.csv', otag, shard);
done = [];
if exist(outfile, 'file')
    prev = csvread(outfile);
    if ~isempty(prev), done = prev(:, 1).'; end
end
fprintf('shard %d/%d: %d designs, %d already done\n', shard, nshard, ndes, numel(done)); fflush(stdout);

for k = 1:ndes
    if mod(k-1, nshard) ~= shard, continue; end
    if any(done == k), continue; end
    t0 = tic;
    p = designs(k, :);
    out_R = p(1); ring_w = p(2); hex_R = p(3); gap_l1 = p(4); petal_s = p(5); gnd_h = p(6);

    f0 = 8e9; fc = 5e9;
    FDTD = InitFDTD('NrTs', NRTS, 'EndCriteria', 1e-4);
    FDTD = SetGaussExcite(FDTD, f0, fc);
    FDTD = SetBoundaryCond(FDTD, {'MUR','MUR','MUR','MUR','MUR','MUR'});

    CSX = InitCSX();
    mesh.x = [-sub.a/2-AIR sub.a/2+AIR];
    mesh.y = [-sub.a/2-AIR sub.a/2+AIR];
    mesh.z = [-AIR sub.h+AIR];

    CSX = AddMaterial(CSX, 'FR4');
    kappa = sub.tand * 2*pi*f0 * EPS0 * sub.epsR;
    CSX = SetMaterialProperty(CSX, 'FR4', 'Epsilon', sub.epsR, 'Kappa', kappa);
    CSX = AddBox(CSX, 'FR4', 0, [-sub.a/2 -sub.a/2 0], [sub.a/2 sub.a/2 sub.h]);

    CSX = AddMetal(CSX, 'patch');
    CSX = AddPolygon(CSX, 'patch', 10, 2, sub.h, ringpoly(0,0,out_R,ring_w,gap_l1,180,0));
    CSX = AddPolygon(CSX, 'patch', 10, 2, sub.h, ringpoly(0,0,hex_R,ring_w,gap_l1,  0,6));
    CSX = AddPolygon(CSX, 'patch', 10, 2, sub.h, petal0 * petal_s);
    CSX = AddBox(CSX, 'patch', 10, [-feed.w/2 -sub.a/2 sub.h], [feed.w/2 -sub.a/2+feed.l sub.h]);

    CSX = AddMetal(CSX, 'gnd');
    CSX = AddBox(CSX, 'gnd', 10, [-sub.a/2 -sub.a/2 0], [sub.a/2 -sub.a/2+gnd_h 0]);

    CSX = AddMetal(CSX, 'srr');
    for s = [-1 1]
        CSX = AddPolygon(CSX,'srr',10,2,0, ringpoly(s*srr.cx,srr.cy,srr.R2,srr.w,srr.gap_out,90,0));
        CSX = AddPolygon(CSX,'srr',10,2,0, ringpoly(s*srr.cx,srr.cy,srr.Rin_out, ...
                                                    srr.Rin_out-srr.R3in,srr.gap_in,270,0));
    end

    [CSX port] = AddLumpedPort(CSX, 5, 1, 50, [-feed.w/2 -sub.a/2 0], ...
                               [feed.w/2 -sub.a/2 sub.h], [0 0 1], true);

    mesh.x = unique([mesh.x, -sub.a/2:RES:sub.a/2, sub.a/2]);
    mesh.y = unique([mesh.y, -sub.a/2:RES:sub.a/2, sub.a/2]);
    mesh.z = unique([mesh.z, linspace(0, sub.h, SUBZ)]);
    mesh = SmoothMesh(mesh, 1.2);
    if min(diff(mesh.x)) < 0.9*RES || min(diff(mesh.y)) < 0.9*RES
        fprintf('design %d: sliver cell, skipped\n', k); fflush(stdout); continue;
    end
    CSX = DefineRectGrid(CSX, unit, mesh);

    Sim_Path = sprintf('tmp_%s_%d', otag, shard);
    CleanupSimPath(Sim_Path);
    WriteOpenEMS([Sim_Path '/s.xml'], FDTD, CSX);
    RunOpenEMS(Sim_Path, 's.xml', '--numThreads=1');

    port = calcPort(port, Sim_Path, freq);
    s11 = 20*log10(abs(port.uf.ref ./ port.uf.inc));

    fid = fopen(outfile, 'a');
    fprintf(fid, '%d', k); fprintf(fid, ',%.4f', p); fprintf(fid, ',%.3f', s11);
    fprintf(fid, '\n'); fclose(fid);

    fprintf('design %3d/%d done in %5.1f s  (min S11 %6.2f dB)\n', ...
            k, ndes, toc(t0), min(s11)); fflush(stdout);
end
fprintf('sweep complete\n');
