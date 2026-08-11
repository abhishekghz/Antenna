function P = ringpoly(cx, cy, Rout, w, gap, gap_ang, nsides)
% RINGPOLY  Closed 2xN polygon of a split ring (SRR arm).
%   Sweeps the outer boundary from one side of the split to the other, then
%   returns along the inner boundary.  nsides = 0 gives a circular ring,
%   nsides = 6 a hexagonal ring with vertices at gap_ang + k*60 deg.
%
%   cx,cy    ring centre
%   Rout     outer radius (circumradius for a polygon ring)
%   w        trace width
%   gap      split width
%   gap_ang  angular position of the split, degrees
Rmid = Rout - w/2;
dth  = asin(min(0.999, (gap/2)/Rmid)) * 180/pi;
th   = linspace(gap_ang + dth, gap_ang + 360 - dth, 240) * pi/180;
if nsides > 0
    n = nsides;
    pr = @(R,t) R*cos(pi/n) ./ cos(mod(t + pi/n, 2*pi/n) - pi/n);
    ro = pr(Rout, th);
    ri = pr(Rout - w/cos(pi/n), th);
else
    ro = Rout * ones(size(th));
    ri = (Rout - w) * ones(size(th));
end
P = [cx + [ro.*cos(th), fliplr(ri.*cos(th))];
     cy + [ro.*sin(th), fliplr(ri.*sin(th))]];
end
