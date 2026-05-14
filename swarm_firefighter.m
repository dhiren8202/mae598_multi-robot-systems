function swarm_firefighter()
% Swarm Firefighting on a Heatmap (single file)
% - Multiple random fire hotspots (Gaussian pixels)
% - Robots assign to hottest active fires, move slowly, avoid collisions
% - Extinguish fires by reducing hotspot intensities on proximity
% - Live heatmap visualization + visible robot bounding boxes

clc; clear; close all;

%% ---------- Parameters ----------
% Map & heatmap
mapSize     = [60 40];        % meters (width x height)
res         = [120 80];       % pixels (cols x rows) for the heatmap
M           = 5;              % number of random fire hotspots
ampRange    = [1.5 3.5];      % intensity range of hotspots
sigmaRange  = [2.0 5.0];      % Gaussian size (meters, std dev)
decay_rate  = 0.9;            % multiplicative decay per second when near a robot (0.9 -> 10%/s)
ext_radius  = 2.0;            % extinguish influence radius (meters)

% Robots
N           = 10;             % number of robots
dt          = 0.15;           % time step (s)
Tmax        = 300;            % max sim time (s)
v_search    = 0.30;           % m/s
v_task      = 0.45;           % m/s when heading to a fire
bbox        = 0.8;            % robot bounding box side (m)
sep_radius  = 1.2*bbox;       % start repelling within this range
k_sep       = 0.8;            % separation gain

% Assignment
reassign_period = 3.0;        % seconds between assignment updates
peak_prominence = 0.15;       % minimum normalized heat to consider a peak active
max_targets_used = min(N, M); % cap concurrent targets to number of robots

rng('shuffle');

%% ---------- Heatmap (world) ----------
[xgrid, ygrid] = meshgrid( linspace(0,mapSize(1),res(1)), linspace(0,mapSize(2),res(2)) );

% Random hotspots (truth state)
fires = struct('pos',{},'amp',{},'sigma',{},'alive',{});
for m = 1:M
    fires(m).pos   = [rand*mapSize(1); rand*mapSize(2)];
    fires(m).amp   = ampRange(1) + (ampRange(2)-ampRange(1))*rand;
    fires(m).sigma = sigmaRange(1) + (sigmaRange(2)-sigmaRange(1))*rand;
    fires(m).alive = true;
end

% Build heat field from hotspots
H = build_heat(xgrid,ygrid,fires);

%% ---------- Robots ----------
% Place robots away from strongest hotspots (to avoid instant success)
robots = zeros(2,N);
for i = 1:N
    ok=false;
    while ~ok
        p = [rand*mapSize(1); rand*mapSize(2)];
        dmin = min(vecnorm( reshape([fires.pos],2,[]) - p, 2, 1 ));
        ok = dmin > 8;  % at least 8 m from any hotspot center
        if ok, robots(:,i)=p; end
    end
end

% Motion memory (correlated random walk)
u_rw = 2*rand(2,N)-1; u_rw = u_rw ./ max(vecnorm(u_rw),1e-6);
last_refresh = zeros(1,N);

% Tasking state
targets_active_idx = [];            % indices of fires currently targeted
assignments = nan(1,N);             % robot -> fire index (NaN = unassigned)
last_assign_t = -inf;

%% ---------- Visualization ----------
figure('Name','Swarm Firefighting — Heatmap');
set(gcf,'Color','w');
ax = axes; hold(ax,'on'); axis(ax,'equal');
xlim(ax,[0 mapSize(1)]); ylim(ax,[0 mapSize(2)]);
grid on; box on;

hImg = imagesc(ax, linspace(0,mapSize(1),res(1)), linspace(0,mapSize(2),res(2)), H);
set(ax,'YDir','normal'); colormap(ax, hot); colorbar; caxis([0, max([fires.amp])*1.1]);
title(ax,'Initializing...');
hRob = plot(ax, robots(1,:), robots(2,:), 'co', 'MarkerFaceColor','c', 'MarkerSize', 6);

%% ---------- Main loop ----------
for step = 1:ceil(Tmax/dt)
    t = (step-1)*dt;

    % 1) Periodic assignment to active fires
    if (t - last_assign_t) >= reassign_period
        % Recompute active peaks from current heatmap
        [peaks_xy, peaks_val, peaks_idx] = find_peaks_from_heat(H, xgrid, ygrid, peak_prominence);
        % Map peaks back to closest underlying fire (by center) that is alive
        active_fire_idx = unique(map_peaks_to_fires(peaks_xy, fires));
        active_fire_idx = active_fire_idx(arrayfun(@(i) fires(i).alive, active_fire_idx));
        % Limit how many fires to attack at once
        if numel(active_fire_idx) > max_targets_used
            % pick the strongest (by peak value order)
            [~, order] = sort(peaks_val, 'descend');
            keep = active_fire_idx( ismember(active_fire_idx, map_peaks_to_fires(peaks_xy(:,order(1:max_targets_used)), fires)) );
            % Keep unique and in initial order of strength
            active_fire_idx = unique(keep,'stable');
        end
        targets_active_idx = active_fire_idx;

        % Greedy assignment: nearest robot to each fire, then next nearest, etc.
        assignments = nan(1,N);
        if ~isempty(targets_active_idx)
            rob_free = 1:N;
            for f = targets_active_idx
                if isempty(rob_free), break; end
                d = vecnorm(robots(:,rob_free) - fires(f).pos, 2, 1);
                [~,ii] = min(d);
                assignments(rob_free(ii)) = f;
                rob_free(ii) = [];
            end
            % Optional: send extra robots to the strongest remaining fire
            if ~isempty(rob_free) && ~isempty(targets_active_idx)
                fstrong = targets_active_idx(1);
                for r = rob_free
                    assignments(r) = fstrong;
                end
            end
        end
        last_assign_t = t;
    end

    % 2) Robot desired velocity (task vs search)
    V_des = zeros(2,N);
    for i = 1:N
        fi = assignments(i);
        if ~isnan(fi) && fires(fi).alive
            dir = fires(fi).pos - robots(:,i);
            V_des(:,i) = v_task * safe_unit(dir);
        else
            % slow, correlated random walk for search
            if t - last_refresh(i) > 4
                u = 2*rand(2,1)-1; u = safe_unit(u);
                u_rw(:,i) = u; last_refresh(i) = t;
            end
            V_des(:,i) = v_search * u_rw(:,i);
        end
    end

    % 3) Separation to avoid collisions (bounding-box idea)
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

    % 4) Integrate motion with caps and noise (very small process noise)
    Sigma_p = (0.004^2)*eye(2);
    for i = 1:N
        v = V_des(:,i) + V_sep(:,i);
        vmax = (~isnan(assignments(i)) && fires(assignments(i)).alive) * v_task + ...
               ( isnan(assignments(i)) || ~fires(assignments(i)).alive) * v_search;
        n = norm(v); if n > vmax, v = v*(vmax/n); end
        robots(:,i) = robots(:,i) + dt*v + chol(Sigma_p)*randn(2,1);

        % keep in bounds
        robots(1,i) = min(max(robots(1,i), 0), mapSize(1));
        robots(2,i) = min(max(robots(2,i), 0), mapSize(2));
    end

    % 5) Extinguish fires when robots are near
    for f = 1:M
        if ~fires(f).alive, continue; end
        % any robot within ext_radius reduces the amplitude multiplicatively
        d = vecnorm(robots - fires(f).pos, 2, 1);
        closeCnt = sum(d < ext_radius);
        if closeCnt > 0
            fires(f).amp = fires(f).amp * (decay_rate^(dt*closeCnt)); % faster with more robots
            if fires(f).amp < 0.1
                fires(f).amp = 0;
                fires(f).alive = false;
            end
        end
    end

    % 6) Rebuild heat map from updated fires (fast enough for these sizes)
    H = build_heat(xgrid,ygrid,fires);

    % 7) Visualization update
    set(hImg,'CData',H);
    set(hRob,'XData',robots(1,:),'YData',robots(2,:));
    % draw bounding boxes
    delete(findall(ax,'Tag','bbox')); % clear previous boxes
    for i=1:N, draw_bbox(ax, robots(:,i), bbox); end
    % mark active/assigned fires
    delete(findall(ax,'Tag','firemark'));
    for f = 1:M
        if fires(f).alive
            plot(ax, fires(f).pos(1), fires(f).pos(2), 'wo', 'MarkerSize', 6, ...
                 'LineWidth', 1.5, 'Tag','firemark');
        else
            plot(ax, fires(f).pos(1), fires(f).pos(2), 'g*', 'MarkerSize', 8, ...
                 'LineWidth', 1.2, 'Tag','firemark');
        end
    end

    aliveCount = sum([fires.alive]);
    title(ax, sprintf('t = %.1fs  |  Fires alive: %d/%d', t, aliveCount, M));
    drawnow limitrate;

    % 8) Termination
    if aliveCount == 0
        fprintf('All fires extinguished at t = %.1fs.\n', t);
        break;
    end
end
end

%% ===== Helper functions =====
function H = build_heat(X,Y,fires)
% Sum of Gaussians
H = zeros(size(X));
for m = 1:numel(fires)
    if ~fires(m).alive || fires(m).amp<=0, continue; end
    dx = X - fires(m).pos(1);
    dy = Y - fires(m).pos(2);
    s2 = fires(m).sigma^2;
    H = H + fires(m).amp * exp(-0.5*(dx.^2 + dy.^2)/s2);
end
% small floor to avoid numeric noise
H(H<1e-6) = 0;
end

function [peaks_xy, peaks_val, peaks_idx] = find_peaks_from_heat(H, X, Y, minfrac)
% Find local maxima on heat field and return their world coords
Hn = H; Hn = Hn ./ max(1e-9, max(Hn(:)));  % normalize 0..1
BW = imregionalmax(Hn);
BW(Hn < minfrac) = 0;
[idx_r, idx_c] = find(BW);
peaks_idx = [idx_r, idx_c];
peaks_val = arrayfun(@(r,c) H(r,c), idx_r, idx_c);
% sort by intensity
[peaks_val, order] = sort(peaks_val, 'descend');
idx_r = idx_r(order); idx_c = idx_c(order);
% map to world coords via X,Y grids
peaks_xy = [ X(sub2ind(size(X), idx_r, idx_c)).'; ...
             Y(sub2ind(size(Y), idx_r, idx_c)).' ];
end

function fire_idx = map_peaks_to_fires(peaks_xy, fires)
% Map each peak to the nearest fire center (index)
if isempty(peaks_xy), fire_idx = []; return; end
centers = reshape([fires.pos],2,[]);
fire_idx = zeros(1, size(peaks_xy,2));
for k = 1:size(peaks_xy,2)
    d = vecnorm(centers - peaks_xy(:,k), 2, 1);
    [~, j] = min(d);
    fire_idx(k) = j;
end
end

function u = safe_unit(v)
n = norm(v); if n < 1e-9, u = [0;0]; else, u = v/n; end
end

function draw_bbox(ax, p, side)
% Visual bounding box centered at p
h = side/2;
x = [p(1)-h, p(1)+h, p(1)+h, p(1)-h, p(1)-h];
y = [p(2)-h, p(2)-h, p(2)+h, p(2)+h, p(2)-h];
plot(ax, x, y, 'c-', 'LineWidth', 1, 'Tag','bbox');
end
