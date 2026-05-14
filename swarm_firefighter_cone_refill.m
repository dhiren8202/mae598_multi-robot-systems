function swarm_firefighter_cone_refill()
% Swarm Firefighting with Spray Cones, Water Tanks, and Refill Stations
% - Heatmap from random Gaussian fires
% - Robots spray cones to reduce heat (consumes water)
% - When low on water, robots go to nearest refill station to refill, then rejoin
% - Simple separation + visible bounding boxes and spray cones
% - No toolboxes required

clc; clear; close all;

%% ----- Parameters -----
% Map & heatmap
mapSize     = [60 40];        % meters (W x H)
res         = [160 108];      % pixels (cols x rows)
M           = 6;              % # initial hotspots
ampRange    = [2.0 4.0];      % hotspot amplitude
sigmaRange  = [2.0 5.0];      % hotspot std dev (m)
global_done_thresh = 0.08;    % stop when max heat < this

% Robots
N           = 10;
dt          = 0.15;
Tmax        = 400;
v_search    = 0.26;
v_task      = 0.42;
v_refill    = 0.48;           % move slightly faster to refuel
bbox        = 0.8;            % robot box size (m)
sep_radius  = 1.2*bbox;
k_sep       = 0.9;

% Spray cone (per robot)
cone_range  = 3.5;            % meters
cone_angle  = deg2rad(50);    % radians full-angle
cone_decay  = 2.0;            % intensity decay factor in exp(-cone_decay*dt)

% Water tank
tank_capacity   = 30;         % water units (arbitrary)
spray_rate      = 2.0;        % units/s when spraying
return_threshold= 6.0;        % go refill when tank <= this
refill_radius   = 1.2;        % within this distance, refill happens
refill_rate     = 10.0;       % units/s refill speed

% Assignment / peaks
reassign_period = 2.5;        % seconds between reassignment
peaks_K         = min(N, 5);  % track up to K hottest peaks
nms_radius      = 2.0;        % peak NMS radius (m)
min_peak_val    = 0.12;       % ignore weak peaks (absolute heat)

% Refill stations (choose 2 stations near opposing sides)
stations = [ 5,              mapSize(2)/2; ...
             mapSize(1)-5,   mapSize(2)/2 ]';  % 2xS

rng('shuffle');

%% ----- Grids & initial heat -----
[xg, yg] = meshgrid(linspace(0,mapSize(1),res(1)), linspace(0,mapSize(2),res(2)));
nms_rad_pix = meters_to_pixels(nms_radius, xg, yg);

fires = struct('pos',{},'amp',{},'sigma',{});
for m = 1:M
    fires(m).pos   = [rand*mapSize(1); rand*mapSize(2)];
    fires(m).amp   = ampRange(1) + (ampRange(2)-ampRange(1))*rand;
    fires(m).sigma = sigmaRange(1) + (sigmaRange(2)-sigmaRange(1))*rand;
end

H = zeros(size(xg));
for m = 1:M
    dx = xg - fires(m).pos(1);
    dy = yg - fires(m).pos(2);
    s2 = fires(m).sigma^2;
    H = H + fires(m).amp * exp(-0.5*(dx.^2 + dy.^2)/s2);
end
H(H<1e-6)=0;

%% ----- Robots & state -----
robots = zeros(2,N);
for i = 1:N
    ok=false;
    while ~ok
        p = [rand*mapSize(1); rand*mapSize(2)];
        ok = H(closest_row(p(2),yg), closest_col(p(1),xg)) < 0.6*max(H(:));
        if ok, robots(:,i)=p; end
    end
end

u_rw = randn(2,N); u_rw = u_rw ./ max(vecnorm(u_rw),1e-6);
last_refresh = zeros(1,N);

assignments = nan(1,N);          % robot -> peak index
last_assign_t = -inf;
peaks_xy = []; peaks_val = [];

tank = tank_capacity*ones(1,N);  % water levels
mode = repmat("search",1,N);     % "search"|"fight"|"refill"
target_station = nan(1,N);       % index of station when refilling

%% ----- Visualization -----
figure('Name','Swarm Firefighting — Cones + Tanks + Refill'); set(gcf,'Color','w');
ax = axes; hold(ax,'on'); axis(ax,'equal');
xlim(ax,[0 mapSize(1)]); ylim(ax,[0 mapSize(2)]);
grid on; box on; xlabel('x [m]'); ylabel('y [m]');

hImg = imagesc(ax, linspace(0,mapSize(1),res(1)), linspace(0,mapSize(2),res(2)), H);
set(ax,'YDir','normal'); colormap(ax, hot); colorbar;
caxis([0, max(H(:))*1.05]);
hRob = plot(ax, robots(1,:), robots(2,:), 'wo', 'MarkerFaceColor','w', 'MarkerSize', 6);

% draw stations
plot(ax, stations(1,:), stations(2,:), 'gs', 'MarkerSize', 10, 'MarkerFaceColor','g', 'LineWidth',1.5);

%% ----- Main loop -----
for step = 1:ceil(Tmax/dt)
    t = (step-1)*dt;

    % 1) Recompute top peaks & reassign (skip robots refilling)
    if (t - last_assign_t) >= reassign_period
        [peaks_xy, peaks_val] = topk_peaks(H, peaks_K, min_peak_val, nms_rad_pix, xg, yg);
        freeRob = find(mode ~= "refill");
        assignments(:) = nan;
        if ~isempty(peaks_xy) && ~isempty(freeRob)
            rf = freeRob;
            for kpk = 1:size(peaks_xy,2)
                if isempty(rf), break; end
                d = vecnorm(robots(:,rf) - peaks_xy(:,kpk), 2, 1);
                [~,ii] = min(d);
                r = rf(ii);
                assignments(r) = kpk;
                rf(ii) = [];
            end
            % extra free robots help the strongest peak
            if ~isempty(rf)
                best = 1;
                for r = rf
                    assignments(r) = best;
                end
            end
        end
        last_assign_t = t;
    end

    % 2) Set modes based on tank and assignment
    for i = 1:N
        switch mode(i)
            case "refill"
                if tank(i) >= tank_capacity-1e-6
                    mode(i) = "search"; target_station(i) = nan;
                end
            otherwise
                if tank(i) <= return_threshold
                    mode(i) = "refill";
                    target_station(i) = nearest_station(robots(:,i), stations);
                elseif ~isnan(assignments(i))
                    mode(i) = "fight";
                else
                    mode(i) = "search";
                end
        end
    end

    % 3) Desired velocity from mode
    V_des = zeros(2,N);
    for i = 1:N
        switch mode(i)
            case "fight"
                pk = assignments(i);
                dir = peaks_xy(:,pk) - robots(:,i);
                V_des(:,i) = v_task * safe_unit(dir);
            case "refill"
                s = target_station(i);
                dir = stations(:,s) - robots(:,i);
                V_des(:,i) = v_refill * safe_unit(dir);
            otherwise % "search"
                if t - last_refresh(i) > 4
                    u = randn(2,1); u = safe_unit(u);
                    u_rw(:,i) = u; last_refresh(i) = t;
                end
                V_des(:,i) = v_search * u_rw(:,i);
        end
    end

    % 4) Separation
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

    % 5) Integrate motion
    for i = 1:N
        v = V_des(:,i) + V_sep(:,i);
        vmax = (mode(i)=="fight")*v_task + (mode(i)=="refill")*v_refill + (mode(i)=="search")*v_search;
        n = norm(v); if n > vmax, v = v*(vmax/n); end
        robots(:,i) = robots(:,i) + dt*v;

        robots(1,i) = min(max(robots(1,i), 0), mapSize(1));
        robots(2,i) = min(max(robots(2,i), 0), mapSize(2));
    end

    % 6) Spray cones (decay heat + consume water)
    for i = 1:N
        if mode(i) ~= "fight", continue; end
        if isnan(assignments(i)), continue; end
        if tank(i) <= 0, tank(i) = 0; mode(i) = "refill"; target_station(i)=nearest_station(robots(:,i), stations); continue; end

        pk = assignments(i);
        if size(peaks_xy,2) < pk || any(isnan(peaks_xy(:,pk))), continue; end

        hdir = safe_unit(peaks_xy(:,pk) - robots(:,i));
        if norm(hdir) < 1e-6, continue; end

        % Apply decay to pixels in cone
        H = apply_cone_decay(H, robots(:,i), hdir, cone_angle, cone_range, cone_decay, dt, xg, yg);

        % Consume water
        tank(i) = max(0, tank(i) - spray_rate*dt);
        if tank(i) <= return_threshold
            mode(i) = "refill";
            target_station(i) = nearest_station(robots(:,i), stations);
        end
    end

    % 7) Refill handling
    for i = 1:N
        if mode(i) == "refill"
            s = target_station(i);
            if norm(robots(:,i) - stations(:,s)) <= refill_radius
                tank(i) = min(tank_capacity, tank(i) + refill_rate*dt);
            end
        end
    end

    % 8) Visualization
    set(hImg,'CData',H);
    set(hRob,'XData',robots(1,:),'YData',robots(2,:));
    delete(findall(ax,'Tag','bbox')); delete(findall(ax,'Tag','cone')); delete(findall(ax,'Tag','tank'));
    for i=1:N
        draw_bbox(ax, robots(:,i), bbox);
        % draw cone aligned with mode
        switch mode(i)
            case "fight"
                pk = assignments(i);
                if ~isnan(pk) && size(peaks_xy,2)>=pk
                    hdir = safe_unit(peaks_xy(:,pk) - robots(:,i));
                else
                    hdir = [1;0];
                end
            case "refill"
                hdir = safe_unit(stations(:,target_station(i)) - robots(:,i));
            otherwise
                hdir = safe_unit(V_des(:,i));
        end
        draw_cone(ax, robots(:,i), hdir, cone_angle, cone_range);
        % tank bar
        draw_tank(ax, robots(:,i), tank(i), tank_capacity);
    end

    % peaks markers
    delete(findall(ax,'Tag','peak'));
    for kpk = 1:size(peaks_xy,2)
        plot(ax, peaks_xy(1,kpk), peaks_xy(2,kpk), 'wo', 'MarkerSize', 6, 'LineWidth', 1.2, 'Tag','peak');
    end

    caxis([0, max(0.1, max(H(:))*1.05)]);
    title(ax, sprintf('t=%.1fs | max heat=%.2f | refilling=%d', t, max(H(:)), sum(mode=="refill")));
    drawnow limitrate;

    % 9) Termination
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

function npx = meters_to_pixels(r, X, Y)
% approximate conversion based on average pixel size
dx = mean(diff(X(1,:)));
dy = mean(diff(Y(:,1)));
npx = max(1, round(r / mean([dx,dy])));
end

function H2 = apply_cone_decay(H, pos, hdir, ang, rng, rate, dt, X, Y)
vx = X - pos(1);
vy = Y - pos(2);
dist = hypot(vx, vy);
inside_r = dist <= rng & dist > 0;
hdx = hdir(1); hdy = hdir(2);
cosang = (vx.*hdx + vy.*hdy) ./ max(dist, 1e-9);
inside_a = acos(max(min(cosang,1),-1)) <= (ang/2);
mask = inside_r & inside_a;
decay = exp(-rate*dt);
H2 = H;
H2(mask) = H2(mask) * decay;
H2(H2<1e-9)=0;
end

function [peaks_xy, peaks_val] = topk_peaks(H, K, minval, rad_pix, X, Y)
peaks_xy = []; peaks_val = [];
if all(H(:) < minval), return; end
Hwork = H;
for k = 1:K
    [vmax, idx] = max(Hwork(:));
    if ~isfinite(vmax) || vmax < minval, break; end
    [r,c] = ind2sub(size(Hwork), idx);
    peaks_val(end+1) = vmax; %#ok<AGROW>
    peaks_xy(:,end+1) = [ X(r,c); Y(r,c) ]; %#ok<AGROW>
    rmin = max(1, r - rad_pix); rmax = min(size(Hwork,1), r + rad_pix);
    cmin = max(1, c - rad_pix); cmax = min(size(Hwork,2), c + rad_pix);
    Hwork(rmin:rmax, cmin:cmax) = -inf;
end
end

function s = nearest_station(p, stations)
d = vecnorm(stations - p, 2, 1);
[~, s] = min(d);
end

function draw_bbox(ax, p, side)
h = side/2;
x = [p(1)-h, p(1)+h, p(1)+h, p(1)-h, p(1)-h];
y = [p(2)-h, p(2)-h, p(2)+h, p(2)+h, p(2)-h];
plot(ax, x, y, 'w-', 'LineWidth', 1, 'Tag','bbox');
end

function draw_cone(ax, p, hdir, ang, rng)
theta = atan2(hdir(2), hdir(1));
ca = cos(theta); sa = sin(theta);
R2 = [ca -sa; sa ca];
th = linspace(-ang/2, ang/2, 25);
edge = R2 * [cos(th); sin(th)] * rng;
patch(ax, [p(1), p(1)+edge(1,1), p(1)+edge(1,end)], ...
           [p(2), p(2)+edge(2,1), p(2)+edge(2,end)], ...
           'c', 'FaceAlpha', 0.08, 'EdgeColor', 'c', 'Tag','cone');
end

function draw_tank(ax, p, val, cap)
% tiny vertical bar next to robot showing tank %
h = 2.0; w = 0.25;  % meters
x0 = p(1) + 0.6;    % offset
y0 = p(2) - h/2;
fill(ax, [x0 x0+w x0+w x0], [y0 y0 y0+h y0+h], 'k', 'FaceAlpha', 0.15, 'EdgeColor','k','Tag','tank');
hfill = h * max(0,min(1,val/cap));
fill(ax, [x0 x0+w x0+w x0], [y0 y0 y0+hfill y0+hfill], 'b', 'FaceAlpha', 0.6, 'EdgeColor','b','Tag','tank');
end

function c = closest_col(x, X)
[~,c] = min(abs(X(1,:) - x));
end

function r = closest_row(y, Y)
[~,r] = min(abs(Y(:,1) - y));
end
