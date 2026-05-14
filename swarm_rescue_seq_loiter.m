function swarm_rescue_seq_loiter()
% Swarm Rescue (sequential targets + loiter ring, single file)
% - Slow robots
% - Random targets each run
% - Sequential engagement: target 1 -> fuse -> done -> target 2 -> ...
% - Bounding boxes visible + separation to prevent collisions
% - Loiter ring around the current fused estimate to avoid pileups

clc; clear; close all;

%% -------- Parameters --------
N = 10;                    % robots
M = 3;                     % number of targets (humans)
mapSize = [50 50];         % meters
dt = 0.1; T = 240;         % step, horizon (seconds)

% Motion (slow)
v_search   = 0.28;         % m/s
v_converge = 0.50;         % m/s

% Loiter behavior around active target estimate
loiter_radius = 3.0;       % meters
loiter_omega  = 0.6;       % rad/s
phase = 2*pi*rand(1,N);    % per-robot phase offset

% Sensing & estimation
detectRadius = 7.0;        % detection range (smaller to avoid instant detection)
Sigma_z = (0.6^2)*eye(2);  % measurement noise
Sigma_p = (0.01^2)*eye(2); % small process noise for motion
tau_cov = 0.20;            % stop criterion per target: trace(P) < tau_cov
k_min   = 5;               % require at least this many reports before accepting

% Collision avoidance via separation (bounding-box idea)
bbox = 0.8;                % side length (m) — drawn around each robot
sep_radius = 1.2*bbox;     % repel within this distance
k_sep = 0.8;               % repulsion gain

rng('shuffle');            % different targets each run

%% -------- World init --------
targets = [rand(1,M)*mapSize(1); rand(1,M)*mapSize(2)];  % random targets

% Place robots away from targets to avoid instant detection
robots = zeros(2,N);
for i = 1:N
    ok = false;
    while ~ok
        p = [rand*mapSize(1); rand*mapSize(2)];
        d = vecnorm(targets - p, 2, 1);
        ok = all(d > 15); % keep >15 m from every target initially
        if ok, robots(:,i) = p; end
    end
end

% Per-target accumulators for WLS
Wsum = repmat(zeros(2), 1, 1, M);
bsum = repmat(zeros(2,1), 1, 1, M);
xhat = nan(2,M); P = repmat(nan(2), 1, 1, M);
reportCount = zeros(1,M);

% Sequential state
current = 1;               % index of target being sought/engaged
engaging = false;          % becomes true after first detection of current

% Correlated random-walk memory
u_rw = 2*rand(2,N)-1; u_rw = u_rw ./ max(vecnorm(u_rw),1e-6);
last_refresh = zeros(1,N);

%% -------- Visualization --------
figure('Name','Swarm Rescue — Sequential Targets + Loiter');
hold on; axis equal; xlim([0 mapSize(1)]); ylim([0 mapSize(2)]);
grid on; box on;

%% -------- Main loop --------
for k = 1:round(T/dt)
    t = (k-1)*dt;

    % If all targets processed, stop
    if current > M
        title(sprintf('All %d targets localized. Done at t=%.1fs', M, t));
        fprintf('SUCCESS: All %d targets localized by t=%.1fs.\n', M, t);
        break;
    end

    %% 1) Compute goals (loiter ring around current estimate if engaging)
    goals = nan(2,N);
    if engaging && all(isfinite(P(:,:,current)), 'all')
        for i=1:N
            theta_i = loiter_omega*t + phase(i);
            offset  = loiter_radius * [cos(theta_i); sin(theta_i)];
            goals(:,i) = xhat(:,current) + offset;
        end
    end

    %% 2) Desired velocity (search or converge-to-loiter)
    V_des = zeros(2,N);
    for i = 1:N
        if ~any(isnan(goals(:,i)))
            dir = goals(:,i) - robots(:,i);
            V_des(:,i) = v_converge * safe_unit(dir);
        else
            % correlated random walk, refresh heading every ~4 s
            if t - last_refresh(i) > 4
                u = 2*rand(2,1)-1; u = safe_unit(u);
                u_rw(:,i) = u; last_refresh(i) = t;
            end
            V_des(:,i) = v_search * u_rw(:,i);
        end
    end

    %% 3) Separation (avoid collisions)
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

    %% 4) Integrate motion with caps and noise
    for i = 1:N
        v = V_des(:,i) + V_sep(:,i);
        vmax = (~isnan(goals(1,i))) * v_converge + (isnan(goals(1,i))) * v_search;
        n = norm(v); if n > vmax, v = v * (vmax/n); end
        robots(:,i) = robots(:,i) + dt*v + chol(Sigma_p)*randn(2,1);

        % keep within map
        robots(1,i) = min(max(robots(1,i), 0), mapSize(1));
        robots(2,i) = min(max(robots(2,i), 0), mapSize(2));
    end

    %% 5) Detections ONLY for the CURRENT target (sequential behavior)
    detectionsThisStep = 0;
    for i = 1:N
        if norm(robots(:,i) - targets(:,current)) < detectRadius
            z = targets(:,current) + chol(Sigma_z)*randn(2,1);
            Wi = inv(Sigma_z); yi = Wi*z;
            Wsum(:,:,current) = Wsum(:,:,current) + Wi;
            bsum(:,:,current) = bsum(:,:,current) + yi;
            [xhat(:,current), P(:,:,current)] = wls_from_accum(Wsum(:,:,current), bsum(:,:,current));
            reportCount(current) = reportCount(current) + 1;
            engaging = true;
            detectionsThisStep = detectionsThisStep + 1;
        end
    end

    %% 6) Check resolve condition for CURRENT target
    if engaging && reportCount(current) >= k_min && trace(P(:,:,current)) < tau_cov
        fprintf('Target %d localized at [%.2f, %.2f] (trace=%.3f) at t=%.1fs with %d reports.\n', ...
            current, xhat(1,current), xhat(2,current), trace(P(:,:,current)), t, reportCount(current));
        % advance to next target
        current = current + 1;
        engaging = false;
        % (loiter phases stay the same; new target will trigger new loiter center)
    end

    %% 7) Visualization
    cla; hold on;

    % draw all targets (truth)
    for m = 1:M
        if m < current
            plot(targets(1,m), targets(2,m), 'g*', 'MarkerSize', 10, 'LineWidth', 1.5);
        elseif m == current
            plot(targets(1,m), targets(2,m), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
        else
            plot(targets(1,m), targets(2,m), 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
        end
    end

    % draw robots + bounding boxes
    for i=1:N
        plot(robots(1,i), robots(2,i), 'bo', 'MarkerFaceColor','b', 'MarkerSize', 6);
        draw_bbox(robots(:,i), bbox);
    end

    % draw current estimate & ellipse + loiter ring guide
    if current <= M && engaging && all(isfinite(P(:,:,current)), 'all')
        plot(xhat(1,current), xhat(2,current), 'k+', 'MarkerSize', 10, 'LineWidth', 2);
        draw_ellipse_95(xhat(:,current), P(:,:,current));
        % ring guide (optional)
        th = linspace(0,2*pi,200);
        ring = xhat(:,current) + loiter_radius*[cos(th); sin(th)];
        plot(ring(1,:), ring(2,:), 'k--');
    end

    title(sprintf('t=%.1fs | current target: %d/%d | reports on current: %d', ...
        t, min(current,M), M, reportCount(min(current,M))));
    drawnow limitrate;
end

% If loop ended without finishing, print status
if current <= M
    fprintf('Stopped at t=%.1fs with %d/%d targets localized.\n', T, current-1, M);
end
end

%% -------- Helper functions --------
function u = safe_unit(v)
n = norm(v); if n < 1e-9, u = [0;0]; else, u = v/n; end
end

function [xhat, P] = wls_from_accum(Wsum, bsum)
% From accumulated information: Wsum = Σ Wi, bsum = Σ Wi*zi
xhat = Wsum \ bsum;
P = inv(Wsum);
end

function draw_ellipse_95(mu, P)
% 95% confidence ellipse for 2D Gaussian
S = (P + P')/2;                 % symmetrize
[V, D] = eig(S);
lam = max(diag(D), 1e-12);
R = V * diag(sqrt(lam));
th = linspace(0, 2*pi, 200);
circle = [cos(th); sin(th)];
E = R * circle * sqrt(5.991) + mu; % chi2inv(0.95,2) = 5.991
plot(E(1,:), E(2,:), 'k-', 'LineWidth', 1.5);
end

function draw_bbox(p, side)
% Draw a square bounding box centered at p
h = side/2;
x = [p(1)-h, p(1)+h, p(1)+h, p(1)-h, p(1)-h];
y = [p(2)-h, p(2)-h, p(2)+h, p(2)+h, p(2)-h];
plot(x, y, 'b-'); % robot's bounding box
end
