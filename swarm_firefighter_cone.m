function swarm_firefighter_cone()
% Swarm Firefighting with Spray Cones (single file, no extra toolboxes)
% - Random Gaussian hotspots -> heatmap
% - Peak finding (top-K with non-maximum suppression) without Image Processing Toolbox
% - Robots assigned to hottest peaks; each sprays a cone that reduces heat
% - Separation to avoid collisions + visible bounding boxes

clc; clear; close all;

%% ----- Parameters -----
% Map & heatmap
mapSize     = [60 40];        % meters (W x H)
res         = [160 108];      % pixels (cols x rows)
M           = 6;              % # of initial hotspots
ampRange    = [2.0 4.0];      % hotspot amplitude
sigmaRange  = [2.0 5.0];      % hotspot std dev (m)
global_done_thresh = 0.08;    % stop when max heat < this

% Robots
N           = 10;
dt          = 0.15;
Tmax        = 360;
v_search    = 0.28;
v_task      = 0.45;
bbox        = 0.8;            % meter square
sep_radius  = 1.2*bbox;
k_sep       = 0.9;

% Spray cone (per robot)
cone_range  = 3.5;            % meters
cone_angle  = deg2rad(50);    % radians (full angle)
cone_decay  = 2.0;            % 1/s multiplier in exp(-cone_decay*dt) inside cone
reassign_period = 2.5;        % seconds between target reassignment
peaks_K     = min(N, 5);      % track up to K hottest peaks
nms_radius  = 2.0;            % non-max suppression radius (meters)
min_peak_val= 0.12;           % ignore weak peaks (absolute heat)

rng('shuffle');

%% ----- Grids & helpers -----
[xg, yg] = meshgrid(linspace(0,mapSize(1),res(1)), linspace(0,mapSize(2),res(2)));
pix_dx = xg(1,2)-xg(1,1);  % meters per pixel (x)
pix_dy = yg(2,1)-yg(1,1);  % meters per pixel (y)
nms_rad_pix = max(1, round(nms_radius / mean([pix_dx,pix_dy])));

% Hotspots (for initial H only)
fires = struct('pos',{},'amp',{},'sigma',{});
for m = 1:M
    fires(m).pos   = [rand*mapSize(1); rand*mapSize(2)];
    fires(m).amp   = ampRange(1) + (ampRange(2)-ampRange(1))*rand;
    fires(m).sigma = sigmaRange(1) + (sigmaRange(2)-sigmaRange(1))*rand;
end

% Initial heatmap
H = zeros(size(xg));
for m = 1:M
    dx = xg - fires(m).pos(1);
    dy = yg - fires(m).pos(2);
    s2 = fires(m).sigma^2;
    H = H + fires(m).amp * exp(-0.5*(dx.^2 + dy.^2)/s2);
end
H(H<1e-6) = 0;  % small floor

%% ----- Robots -----
robots = zeros(2,N);
for i = 1:N
    ok = false;
    while ~ok
        p = [rand*mapSize(1); rand*mapSize(2)];
        % start away from strong heat
        [~,~,val] = world_to_pixel(p(1), p(2), xg, yg);
        ok = val < 0.5*max(H(:));
        if ok, robots(:,i) = p; end
    end
end

% Motion memory
u_rw = randn(2,N); u_rw = u_rw ./ max(vecnorm(u_rw),1e-6);
last_refresh = zeros(1,N);

% Targets and assignment
peaks_xy = []; peaks_val = [];
assignments = nan(1,N);          % robot -> peak index
last_assign_t = -inf;

%% ----- Visualization -----
figure('Name','Swarm Firefighting — Spray Cones'); set(gcf,'Color','w');
ax = axes; hold(ax,'on'); axis(ax,'equal');
xlim(ax,[0 mapSize(1)]); ylim(ax,[0 mapSize(2)]);
grid on; box on; xlabel('x [m]'); ylabel('y [m]');

hImg = imagesc(ax, linspace(0,mapSize(1),res(1)), linspace(0,mapSize(2),res(2)), H);
set(ax,'YDir','normal'); colormap(ax, hot); colorbar;
caxis([0, max(H(:))*1.05]);

hRob = plot(ax, robots(1,:), robots(2,:), 'co', 'MarkerFaceColor','c', 'MarkerSize', 6);

%% ----- Main loop -----
for step = 1:ceil(Tmax/dt)
    t = (step-1)*dt;

    % 1) Recompute peaks and reassign periodically
    if (t - last_assign_t) >= reassign_period
        [peaks_xy, peaks_val] = topk_peaks(H, peaks_K, min_peak_val, nms_rad_pix, xg, yg);
        assignments = nan(1,N);
        if ~isempty(peaks_xy)
            rob_free = 1:N;
            % assign one robot to each peak by nearest distance
            for kpk = 1:size(peaks_xy,2)
                if isempty(rob_free), break; end
                d = vecnorm(robots(:,rob_free) - peaks_xy(:,kpk), 2, 1);
                [~,ii] = min(d);
                assignments(rob_free(ii)) = kpk;
                rob_free(ii) = [];
            end
            % send extra robots to strongest peak if any left
            if ~isempty(rob_free)
                best = 1; % peaks are sorted desc
                for r = rob_free
                    assignments(r) = best;
                end
            end
        end
        last_assign_t = t;
    end

    % 2) Desired velocity (task or search)
    V_des = zeros(2,N);
    for i = 1:N
        pk = assignments(i);
        if ~isnan(pk)
            dir = peaks_xy(:,pk) - robots(:,i);
            V_des(:,i) = v_task * safe_unit(dir);
        else
            if t - last_refresh(i) > 4
                u = randn(2,1); u = safe_unit(u);
                u_rw(:,i) = u; last_refresh(i) = t;
            end
            V_des(:,i) = v_search * u_rw(:,i);
        end
    end

    % 3) Separation
    V_sep = zeros(2,N);
    for i = 1:N
        for j = i+1:N
            dvec = robots(:,i) - robots(:,j);
            d = norm(dvec);
            if d < sep_radius && d > 1e-6
                push = k_sep * (1/d - 1/sep_radius) * (dvec/d);
                V_sep(:,i) = V_sep(:,i) + push;
                V_sep(:,j) = V_sep(:,j) - push;
            end
        end
    end

    % 4) Integrate motion (cap speeds)
    for i = 1:N
        v = V_des(:,i) + V_sep(:,i);
        vmax = (~isnan(assignments(i))) * v_task + (isnan(assignments(i))) * v_search;
        n = norm(v); if n > vmax, v = v * (vmax/n); end
        robots(:,i) = robots(:,i) + dt*v;

        % keep in bounds
        robots(1,i) = min(max(robots(1,i), 0), mapSize(1));
        robots(2,i) = min(max(robots(2,i), 0), mapSize(2));
    end

    % 5) Spray cones: reduce H pixels inside each cone
    for i = 1:N
        pk = assignments(i);
        if ~isnan(pk)
            hdir = safe_unit(peaks_xy(:,pk) - robots(:,i));
        else
            hdir = safe_unit(V_des(:,i));  % face walking direction
        end
        if norm(hdir) < 1e-6, continue; end
        H = apply_cone_decay(H, robots(:,i), hdir, cone_angle, cone_range, cone_decay, dt, xg, yg);
    end

    % 6) Visualization updates
    set(hImg, 'CData', H);
    set(hRob, 'XData', robots(1,:), 'YData', robots(2,:));

    % cones & bboxes
    delete(findall(ax,'Tag','bbox')); delete(findall(ax,'Tag','cone'));
    for i = 1:N
        draw_bbox(ax, robots(:,i), bbox);
        pk = assignments(i);
        if ~isnan(pk)
            hdir = safe_unit(peaks_xy(:,pk) - robots(:,i));
        else
            hdir = safe_unit(V_des(:,i));
        end
        draw_cone(ax, robots(:,i), hdir, cone_angle, cone_range);
    end

    % peaks markers
    delete(findall(ax,'Tag','peak'));
    for kpk = 1:size(peaks_xy,2)
        plot(ax, peaks_xy(1,kpk), peaks_xy(2,kpk), 'wo', 'MarkerSize', 6, ...
            'LineWidth', 1.2, 'Tag', 'peak');
    end

    caxis([0, max(0.1, max(H(:))*1.05)]);
    title(ax, sprintf('t = %.1fs | max heat = %.2f | peaks = %d', t, max(H(:)), size(peaks_xy,2)));
    drawnow limitrate;

    % 7) Termination
    if max(H(:)) < global_done_thresh
        fprintf('All fires extinguished (max heat < %.2f) at t = %.1fs.\n', global_done_thresh, t);
        break;
    end
end
end

%% ===== Helper functions =====
function u = safe_unit(v)
n = norm(v); if n < 1e-9, u = [0;0]; else, u = v/n; end
end

function [col, row, val] = world_to_pixel(x, y, X, Y)
% nearest pixel index for world coord
[~, col] = min(abs(X(1,:) - x));
[~, row] = min(abs(Y(:,1) - y));
val = X(1,1); %#ok<NASGU> % unused, kept for signature compatibility
end

function H2 = apply_cone_decay(H, pos, hdir, ang, rng, rate, dt, X, Y)
% Multiply pixels in the cone by exp(-rate*dt)
% Build mask via vector math (no toolboxes)
vx = X - pos(1);
vy = Y - pos(2);
dist = hypot(vx, vy);
inside_r = dist <= rng & dist > 0;

% angle between (vx,vy) and heading
hdx = hdir(1); hdy = hdir(2);
cosang = (vx.*hdx + vy.*hdy) ./ max(dist, 1e-9);
inside_a = acos(max(min(cosang,1),-1)) <= (ang/2);

mask = inside_r & inside_a;
decay = exp(-rate*dt);
H2 = H;
H2(mask) = H2(mask) * decay;
% clamp tiny values to zero
H2(H2<1e-9) = 0;
end

function [peaks_xy, peaks_val] = topk_peaks(H, K, minval, rad_pix, X, Y)
% Find top-K peaks using iterative max + non-max suppression (no IPT)
peaks_xy = [];
peaks_val = [];
Hwork = H;
for k = 1:K
    [vmax, idx] = max(Hwork(:));
    if ~isfinite(vmax) || vmax < minval, break; end
    [r,c] = ind2sub(size(Hwork), idx);
    peaks_val(end+1) = vmax; %#ok<AGROW>
    peaks_xy(:,end+1) = [ X(r,c); Y(r,c) ]; %#ok<AGROW>
    % suppress neighborhood
    rmin = max(1, r - rad_pix); rmax = min(size(Hwork,1), r + rad_pix);
    cmin = max(1, c - rad_pix); cmax = min(size(Hwork,2), c + rad_pix);
    Hwork(rmin:rmax, cmin:cmax) = -inf;
end
end

function draw_bbox(ax, p, side)
h = side/2;
x = [p(1)-h, p(1)+h, p(1)+h, p(1)-h, p(1)-h];
y = [p(2)-h, p(2)-h, p(2)+h, p(2)+h, p(2)-h];
plot(ax, x, y, 'c-', 'LineWidth', 1, 'Tag', 'bbox');
end

function draw_cone(ax, p, hdir, ang, rng)
% outline of cone for visualization only
th = linspace(-ang/2, ang/2, 20);
R = [hdir, [0;0]]; %#ok<NASGU>
% Build 2D rotation matrix from heading vector
theta = atan2(hdir(2), hdir(1));
ca = cos(theta); sa = sin(theta);
R2 = [ca -sa; sa ca];
edge = R2 * [cos(th); sin(th)] * rng;
patch(ax, [p(1), p(1)+edge(1,1), p(1)+edge(1,end)], ...
           [p(2), p(2)+edge(2,1), p(2)+edge(2,end)], ...
           'c', 'FaceAlpha', 0.08, 'EdgeColor', 'c', 'Tag','cone');
end
